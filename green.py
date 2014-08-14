import numpy as np
import pyta.defaults as defaults
import solver
import pyta.lead
import pyta.consts
import copy

class Green(solver.Solver):
    """ Abstract Base Class for energy resolved Green's function

        invar:
        1) leads
            list of Leads
            overwrite, append

        parameters:
        1) size

        outvar:
        1) green_ret
        2) green_gr
        3) green_lr
        4) spectral
        5) transmission

    """

    def __init__(self, 
            #Parameters
            size):
        """
        Base class constructor only invoked by derived classes.

        Arguments:

        size (integer, parameter)
            specify the size of the system

        """

        #Invar
        self.leads = list()

        #Param
        self.size = size

        #Outvar
        self.green_ret = None
        self.green_gr = None
        self.green_lr = None
        self.transmission = None

        super(Green, self).__init__()

    def get_green_ret(self):
        if self.green_ret is None:
            self._do_green_ret()
        return self.green_ret

    def get_green_lr(self):
        if self.green_lr is None:
            self._do_green_lr()
        return self.green_lr

    def get_green_gr(self):
        if self.green_gr is None:
            self._do_green_gr()
        return self.green_gr

    def _do_green_ret(self):
        raise RuntimeError('Base Class Green has no _do_green_ret method')

    def _do_green_gr(self):
        """Calculate equilibrium Green's function.
        If the lesser is available avoid direct calculation of sigma_gr."""
        if self.green_lr is not None:
            spectral = self.get('spectral')
            green_lr = self.get('green_lr')
            self.green_gr = green_lr - 1j*spectral
        else:
            sigma_gr = np.matrix(np.zeros((self.size, self.size), dtype=np.complex128))
            self._add_leads(sigma_gr, 'sigma_gr')
            self.green_gr = self.get_green_ret() * sigma_gr * self.get_green_ret().H
        return

    def _do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.matrix(np.zeros((self.size, self.size), dtype=np.complex128))
        self._add_leads(sigma_lr, 'sigma_lr')
        self.green_lr = self.get_green_ret() * sigma_lr * self.get_green_ret().H
        return

    def set_leads(self, leads, mode='replace'):
        """Add a Leads set"""
        if mode == 'replace':
            assert(type(leads) == list)
            self.leads = leads
        if mode == 'remove':
            assert(isinstance(leads, pyta.lead.Lead))
            if leads in self.leads:
                self.leads.remove(leads)
        if mode == 'append':
            assert(isinstance(leads, pyta.lead.Lead))
            assert(leads not in self.leads)
            self.leads.append(leads)
        self.cleandep_leads()
        return

    def cleandep_leads(self):
        self.green_ret = None
        self.green_gr = None
        self.green_lr = None


    def get_spectral(self):
        """Get spectral function A = j(G^{r} - G^{a})"""
        green_ret = self.get_green_ret()
        spectral = 1j * (green_ret - green_ret.H)
        return spectral

    def _add_leads(self, mat, varname):
        """Subtract all the leads contributions from matrix mat.
         Result in place. """
        for lead in self.leads:
            sigma = lead.get(varname)
            assert(sigma.shape[0]==sigma.shape[1])
            size = sigma.shape[0]
            pos = lead.get('position')
            mat[pos: pos+size, pos:pos+size] += sigma
        return

    def _resize_lead_matrix(self, lead, varname):
        """Resize a lead matrix in a matrix with the shape of the Green.
        Elements out of the lead are set to zero.
        Variable to be converted must be given as string."""
        leadmat = lead.get(varname)
        vartype = np.result_type(leadmat)
        mat = np.zeros((self.size, self.size), dtype=vartype)
        leadsize = leadmat.shape[0]
        pos = lead.get('position')
        mat[pos: pos+leadsize, pos:pos+leadsize] = leadmat
        return mat


    def get_transmission(self, leads = None):
        """Calculate the transmission in place for a given couple of leads,
        specified as an iterable. If leads are not specified, use lead 0 and 1
        if set.
        It implements the Landauer Caroli formula"""
        if leads is not None:
            assert(len(leads)==2)
            lead1=leads[0]
            lead2=leads[1]
        else:
            lead1=self.leads[0]
            lead2=self.leads[1]
        gamma1 = np.zeros((self.size, self.size))
        pos = lead1.get('position')
        size = lead1.get('size')
        gamma1[pos: pos+size, pos:pos+size]+=lead1.get('gamma')
        gamma2 = np.zeros((self.size, self.size))
        pos = lead2.get('position')
        size = lead2.get('size')
        gamma2[pos: pos+size, pos:pos+size]+=lead1.get('gamma')
        green_ret = self.get_green_ret()
        trans = (np.trace(gamma1 * green_ret * gamma2 * green_ret.H))
        return trans

    def get_meirwingreen(self, lead=None):
        assert(lead is not None)
        glr = self.get('green_lr')
        ggr = self.get('green_gr')
        sgr = self._resize_lead_matrix(lead, 'sigma_gr')
        slr = self._resize_lead_matrix(lead, 'sigma_lr')
        current = ((pyta.consts.e / pyta.consts.h_eVs) *
                    np.trace(slr*ggr - sgr*glr))
        return current

    def get_dattacurrent(self, lead=None):
        """Calculate the current using the Datta version of Meir Wingreen:
            I = Sigma_lesser * A - Gamma * G_lesser"""
        assert(lead is not None)
        green_n = -1.0j * self.get('green_lr')
        sigma_n = -1.0j * self._resize_lead_matrix(lead, 'sigma_lr')
        gamma = self._resize_lead_matrix(lead, 'gamma')
        spectral = self.get('spectral')
        current = ((pyta.consts.e / pyta.consts.h_eVs) *
                    np.trace(sigma_n * spectral - gamma * green_n))
        return current

    def get_occupation(self):
        """Calculate the occupation by comparing the lesser green function
        and the spectral function. """
        diag1 = self.get_green_lr()
        diag2 = self.get_spectral()
        occupation = (np.imag(diag1/ diag2))
        return occupation


class ElGreen(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self, 
            #Parameters
            ham, over = None):
        """ElGreen is initialized by specifying an Hamiltonian as numpy.matrix.
        Optionally, overlap can be specified.
        If overlap is not specified, an orthogonal basis is assumed.

        See base class Green

        invar:
        1) energy
            mode: overwrite

        2) leads
            set of Leads
            mode: overwrite, append

        parameters:
        1)  ham
            system hamiltonian, numpy.matrix
        2)  overlap (optional)
            overlap matrix, real numpy.matrix

        """

        #Param
        #========================================================
        assert(type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert(type(over) == np.matrixlib.defmatrix.matrix)
        self.ham = ham
        size = len(self.ham)
        if over is None:
            self.over = np.matrix(np.eye(size))

        #Invar
        #=================================
        self.energy = None
        #=================================

        #Base constructor
        super(ElGreen, self).__init__(size)

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.cleandep_energy()
            #Update energy point in all leads where set_energy is defined
            self._spread_energy(energy)
        return

    def cleandep_energy(self):
        self.green_ret = None
        self.green_lr = None
        self.green_gr = None


    def _spread_energy(self, energy):
        "Distribute energy in leads"
        for lead in self.leads:
            try:
                lead.set_energy(energy)
            except AttributeError:
                pass
        return

    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        sigma = np.matrix(np.zeros((self.size, self.size), dtype=np.complex128))
        esh = self.energy * self.over - self.ham
        self._add_leads(sigma, 'sigma_ret')
        esh -= sigma
        self.green_ret = esh.I

        return self.green_ret


class PhGreen(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""
    
    def __init__(self, 
            #Param
            spring, mass=None):
        """PhGreen is initialized by specifying a coupling spring constant
        matrix as numpy.matrix.
        Optionally, masses can be specified.
        Masses must be specified as a diagonal numpy.matrix
        If masses are not specified, an identity matrix is assumed.

        See base class Green

        invar:
        1) frequency
           mode: overwrite

        2) leads
            set of Leads
            mode: overwrite, append

        parameters:
        1)  sprint
            spring constants, numpy.matrix
        2)  mass (optional)
            mass matrix, real numpy.matrix

        """

        #Param
        #=======================================
        self.spring = spring
        self.mass = mass
        assert(type(spring) == np.matrixlib.defmatrix.matrix)
        if not (mass is None):
            assert(type(spring) == np.matrixlib.defmatrix.matrix)
        self.spring = spring
        size = len(self.spring)
        if mass is None:
            self.mass = np.matrix(np.eye(size))
        #======================================

        #Invar
        #=======================================
        self.frequency = None
        #=======================================
        
        #Base constructor
        super(PhGreen, self).__init__(size)

    def set_frequency(self, frequency):
        """Set energy point"""
        if frequency != self.frequency:
            self.frequency = frequency
            self.cleandep_frequency()
            self._spread_frequency(frequency)

    def cleandep_frequency(self):
        self.green_ret = None
        self.green_lr = None
        self.green_gr = None
        return

    def _spread_frequency(self, frequency):
        "Distribute energy in leads"
        for lead in self.leads:
            try:
                lead.set_frequency(frequency)
            except AttributeError:
                pass
            else:
                lead.set_frequency(frequency)
        return


    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        esh = self.frequency * self.frequency * self.mass - self.spring
        self._add_lead_sigma(esh)
        self.green_ret = esh.I

        return self.green_ret


class SCCMixer():
    """A class to link self-consistently two inter-dependent solvers.
    The parameters are passed through constructor. When the constructor is
    invoked, the solvers must be still independent. Internally, a set method
    is invoked. Self-consistency is then checked on the quantities specified
    in input (more quantities can be converging at once).

    Example: an instance foo_A of a solver class A depends on a solver class B.
    The instance foo_B of the solver class B depends on A. The SCC is considered
    solved when A.whatever is converged.

    WITHOUT setting A.set('foo_B', B) and B.set('foo_') we call
    >> SCCMixer = scc(A,B,varname='whatever',tol=1e-10,maxiter=1000)
    >> scc.do()

    After the cycle is finished, A and B contain the SCC solutions."""
    def __init__(self, solver_a, solver_b, varname, tol,
                 maxiter=1000,
                 niter = None):
        """
        solver_a, solver_b : instances of solver class to be linked
        b_in_a_name, a_in_b_name : variable names to be used to set instance b
        varname :   string or list of strings with variables to be verified
                    for SCC convergence
        tol : numerical tolerance
        maxiter : maximum number of iterations
        niter : if specified, exit anyway after niter iterations.
                In this case ignore tolerance
        """

        #Param
        self.solver_a = solver_a
        self.solver_b = solver_b
        #varname is always stored as list
        if not (type(varname) == str or type(varname) == list):
            raise ValueError('varname must be a string or a list of strings')
        if type(varname) == list:
            for var in varname:
                if not type(var) == str:
                    raise ValueError('varname must be a list of strings')
            self.varname = varname
        if type(varname == str):
            self.varname = [varname]
        self.tol = tol
        self.maxiter = maxiter

    def do(self):
        """
        Run the scc calculation. When done, the solvers are
        linked and contain the SCC solutions
        """
        solver_b = self.solver_b
        solver_a = self.solver_a
        local_a = copy.deepcopy(solver_a)
        # Get all the values for convergence check, as a list
        var_a = [solver_a.get(var) for var in self.varname]
        raise RuntimeError('SCCMixer class not yet finished')
        return




class SCBA():
    """A class to solve Self Consistent Born Approximation Loop"""
    def __init__(self, 
        #Params
        greensolver, selfener, tol = defaults.scbatol, maxiter=1000,
        task='both'):
        """ greensolver and selfener are the solver 
        to be plugged in the loop.
        Task specify whether we loop on the equilibrium ('eq') or the Keldysh
        ('keldysh') or both"""

        #Param
        self.green = greensolver
        self.selfener = selfener
        self.tol = tol
        self.maxiter = maxiter
        self.task = task



    def do(self):
        """Link the green and self energy solvers and run the scba loop"""

        selfener = self.selfener
        green = self.green
        local = copy.deepcopy(green)

        if self.task == 'both':
            #1st iteration: green without scba self energy and 1st order
            #Born self energy
            green_buf = green.get('green_ret')
            green_buf_lr = green.get('green_lr')
            local.set('leads', selfener, mode='append')
            green_local = local.get('green_ret')
            green_local_lr = local.get('green_lr')
            green.set('leads', selfener, mode='append')
            #Now append the scba
            for ind, scba in enumerate(range(self.maxiter)):
                err1 = (green_local - green_buf).max()
                err2 = (green_local_lr - green_buf_lr).max()
                if (abs(err1)<self.tol) and (abs(err2)<self.tol):
                    return
                selfener.set('greensolver', local)
                green.cleandep('leads')
                green_buf = green.get('green_ret')
                green_buf_lr = green.get('green_lr')
                selfener.set('greensolver', green)
                local.cleandep('leads')
                green_local = local.get('green_ret')
                green_local_lr = local.get('green_lr')
            raise RuntimeError('SCBA loop not converged')

        if self.task == 'equilibrium':
            #1st iteration: green without scba self energy and 1st order
            #Born self energy
            green_buf = green.get('green_ret')
            local.set('leads', selfener, mode='append')
            green_local = local.get('green_ret')
            green.set('leads', selfener, mode='append')
            #Now append the scba
            for ind, scba in enumerate(range(self.maxiter)):
                err1 = (green_local - green_buf).max()
                print('SCBA loop', ind, err1)
                if (abs(err1)<self.tol):
                    return
                selfener.set('greensolver', local)
                green.cleandep('leads')
                green_buf = green.get('green_ret')
                selfener.set('greensolver', green)
                local.cleandep('leads')
                green_local = local.get('green_ret')
            raise RuntimeError('SCBA loop not converged')


        if self.task == 'keldysh':
            green_buf_lr = green.get('green_lr')
            for ind, scba in enumerate(range(self.maxiter)):
                green_buf_lr = self.green.get_green_lr()
                self.selfener.set_eqgreen(self.green.get_eqgreen())
                self.selfener.set_green_lr(self.green.get_green_lr())
                self.green.set_lead(self.selfener)
                green_after = self.green.get_eqgreen()
                green_after_lr = self.green.get_green_lr()
                err2 = (green_after_lr - green_buf_lr).max()
                if (abs(err2)<self.tol):
                    return
            raise RuntimeError('SCBA loop not converged')
        else:
            raise ValueError('Unknown run mode for SCBA mixer')
