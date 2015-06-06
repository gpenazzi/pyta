import numpy as np
from pyta import defaults
from pyta import mathutils
from pyta import solver
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

    def _reset(self):
        self.green_ret = None
        self.green_lr = None
        self.green_gr = None

    def get_green_ret(self):
        """
        Returns:
            np.matrix: Retarded Green's function
        """
        if self.green_ret is None:
            self._do_green_ret()
        return self.green_ret

    def get_green_lr(self):
        """
        Returns:
        np.matrix: Lesser Green's function
        """
        if self.green_lr is None:
            self._do_green_lr()
        return self.green_lr

    def get_green_gr(self):
        """
        Returns:
            np.matrix: Greater Green's function
        """
        if self.green_gr is None:
            self._do_green_gr()
        return self.green_gr

    def _do_green_ret(self):
        raise RuntimeError('Base Class Green has no _do_green_ret method')

    def _do_green_gr(self):
        """Calculate equilibrium Green's function.
        If the lesser is available avoid direct calculation of sigma_gr."""
        if self.green_lr is not None:
            spectral = self.get_spectral()
            green_lr = self.get_green_lr()
            self.green_gr = green_lr - 1j * spectral
        else:
            sigma_gr = np.asmatrix(
                np.zeros((self.size, self.size), dtype=np.complex128))
            self._add_leads(sigma_gr, "sigma_gr")
            self.green_gr = (self.get_green_ret * sigma_gr *
                             self.get_green_ret.H)
        return

    def _do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.asmatrix(
            np.zeros((self.size, self.size), dtype=np.complex128))
        self._add_leads(sigma_lr, "sigma_lr")
        green_ret = self.get_green_ret()
        self.green_lr = green_ret * sigma_lr * green_ret.H
        return

    def set_leads(self, leads):
        self.leads = leads

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
            assert (sigma.shape[0] == sigma.shape[1])
            size = sigma.shape[0]
            pos = lead.get('position')
            mat[pos: pos + size, pos:pos + size] = \
                mat[pos: pos + size, pos:pos + size] + sigma
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
        mat[pos: pos + leadsize, pos:pos + leadsize] = leadmat
        return mat


    def get_transmission(self, leads=None):
        """Calculate the transmission in place for a given couple of leads,
        specified as an iterable. If leads are not specified, use lead 0 and 1
        if set.
        It implements the Landauer Caroli formula"""
        if leads is not None:
            assert (len(leads) == 2)
            lead1 = leads[0]
            lead2 = leads[1]
        else:
            lead1 = self.leads[0]
            lead2 = self.leads[1]
        gamma1 = np.zeros((self.size, self.size), dtype=np.complex128)
        pos = lead1.get('position')
        size = lead1.get('size')
        gamma1[pos: pos + size, pos:pos + size] += lead1.get('gamma')
        gamma2 = np.zeros((self.size, self.size), dtype=np.complex128)
        pos = lead2.get('position')
        size = lead2.get('size')
        gamma2[pos: pos + size, pos:pos + size] += lead2.get('gamma')
        green_ret = self.get_green_ret
        trans = (np.trace(gamma1 * green_ret * gamma2 * green_ret.H))
        return trans

    def get_meirwingreen(self, lead=None):
        """
        Calculate the total current in a specified lead
        by applying the Meir-Wirgreen formula

        Args:
            lead (Lead): reference lead where the current is evaluated
                equation
        Returns:
            float: value of current in given energy point
        """
        assert (lead is not None)
        glr = self.get_green_lr()
        ggr = self.get_green_gr()
        sgr = self._resize_lead_matrix(lead, 'sigma_gr')
        slr = self._resize_lead_matrix(lead, 'sigma_lr')
        const = 1.0  #pyta.consts.e / pyta.consts.h_eVs)
        current = const * np.trace(slr * ggr - sgr * glr)
        return current

    def get_dattacurrent(self, lead=None):
        """Calculate the current using the Datta version of Meir Wingreen:
            I = Sigma_lesser * A - Gamma * G_lesser

        Args:
            lead (Lead): reference lead where the current is evaluated
                equation
        Returns:
            float: value of current in given energy point
        """
        assert (lead is not None)
        green_n = -1.0j * self.get_green_lr()
        sigma_n = -1.0j * self._resize_lead_matrix(lead, 'sigma_lr')
        gamma = self._resize_lead_matrix(lead, 'gamma')
        spectral = self.get_spectral()
        current = ((pyta.consts.e / pyta.consts.h_eVs) *
                   np.trace(sigma_n * spectral - gamma * green_n))
        return current

    def get_occupation(self):
        """Calculate the occupation by comparing the lesser green function
        and the spectral function. """
        diag1 = self.get_green_lr()
        diag2 = self.get_spectral()
        occupation = (np.imag(diag1 / diag2))
        return occupation

    def scba(self, lead, mode='equilibrium', niter=None, maxiter=None,
             tolerance=None, alpha=1.0):
        """
        Perform Born Approximation mixing with respect to self energy contained
        in lead specified in input
        """

        if mode == 'equilibrium':
            green_varname = 'green_ret'
            sigma_varname = 'sigma_ret'
        elif mode == 'keldysh':
            green_varname = 'green_lr'
            sigma_varname = 'sigma_lr'
        else:
            raise AttributeError('Unknown mode: must be equilibrium or keldysh')

        #Initialize the virtual lead self-energy to zero
        lead.set(green_varname, np.zeros((self.size, self.size)))

        def func1(var1):
            ## Note: var1 here is dummy because it is automatically retrievable
            ## everytime a lead.get is invoked in func2
            self._reset()
            return self.get(green_varname)

        def func2(var2):
            ## Here var2 in input must be the right green's function and the
            ## return value is the corresponding sigma
            lead.set(green_varname, var2)
            return lead.get(sigma_varname)

        ## G0 calculated here (self-energy initialized to zero)
        var1_guess = self.get(green_varname)
        mathutils.linear_mixer(func1, func2, var1_guess, niter=niter,
                               tolerance=tolerance,
                               alpha=alpha, maxiter=maxiter)


class ElGreen(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self,
                 #Parameters
                 ham, over=None, delta=0.0):
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
        3)  delta (optional)
            if no leads are present, it is used for shifting poles

        """

        #Param
        #========================================================
        assert (type(ham) == np.matrixlib.defmatrix.matrix)
        if over is not None:
            assert (type(over) == np.matrixlib.defmatrix.matrix)
            self.over = over
        self.ham = ham

        #Consistency check (yes, it happened to wrongly set a non hermitian
        # hamiltonian...good luck spotting that without a check)
        if (np.abs(ham - ham.H) > 1e-7 ).any():
            raise ValueError(
                'Error in Green parameter. The Hamiltonian is not hermitian')

        size = len(self.ham)
        if over is None:
            self.over = np.asmatrix(np.eye(size))

        self.delta = delta
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
            self._reset()
            #Update energy point in all leads where set_energy is defined
            self._spread_energy(energy)
        return

    def _spread_energy(self, energy):
        "Distribute energy in leads"
        for lead in self.leads:
            try:
                lead.set('energy', energy)
            except AttributeError:
                pass
        return

    def get_local_currents(self):
        """
        Calculate the matrices of local currents for all orbitals in the system
        """
        green_lr = self.get_green_lr()
        ham = self.ham
        over = self.over
        en = self.energy
        lc = np.real(np.multiply(2. * (ham - en * over), green_lr))
        print('lc', lc[1, 2], 'energy ', en, 'gl', green_lr[1, :])
        return lc

    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        sigma = np.asmatrix(
            np.zeros((self.size, self.size), dtype=np.complex128))
        esh = self.energy * self.over - self.ham
        self._add_leads(sigma, 'sigma_ret')
        ## Note: I add an imaginary part, to avoid problem if I have no
        ## leads. Anyway it is small respect to self energy. If you don't want
        ## it, set delta to 0.0 in constructor
        esh = esh - sigma + 1j*self.delta * self.over
        self.green_ret = esh.I

        return self.green_ret


class PhGreen(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""

    def __init__(self,
                 #Param
                 spring, mass=None, delta=0.0):
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
        assert (type(spring) == np.matrixlib.defmatrix.matrix)
        if not (mass is None):
            assert (type(spring) == np.matrixlib.defmatrix.matrix)
        self.spring = spring
        size = len(self.spring)
        if mass is None:
            self.mass = np.asmatrix(np.eye(size))
        #======================================

        #Invar
        #=======================================
        self.frequency = None
        self.delta = delta
        #=======================================

        #Base constructor
        super(PhGreen, self).__init__(size)

    def set_frequency(self, frequency):
        """Set energy point"""
        if frequency != self.frequency:
            self.frequency = frequency
            self._reset()
            self._spread_frequency(frequency)

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
        esh = esh + 1j*self.delta * self.over
        self.green_ret = esh.I

        return self.green_ret


## Deprecated code. Waiting for my decision to delete it

#class LinearMixerSCC():
#    """A class to link self-consistently two inter-dependent solvers.
#    The parameters are passed through constructor. When the constructor is
#    invoked, the solvers must be still already linked.
#
#    IMPORTANT: the solver A is the target one, it is necessary that the value
#    of varname_a is defined (to some initial guess) and is not None, otherwise
#    you will end up in an undefined recursion.
#
#    Example: solver A and B have interdependent quantities A.a and B.b
#    A.a(0) must have a finite value, then the mixer will compute
#
#    B.b(0) = f(A.a(0))
#    A.a(1) = (1-alpha)*A.a(0) + alpha*f(B.b(1))
#    B.b(1) = f(A.a(1))
#    ...
#
#    WITHOUT setting A.set('foo_B', B) and B.set('foo_') we call
#    >> SCCMixer = scc(A,B,varname='whatever',tol=1e-10,maxiter=1000)
#    >> scc.do()
#
#    After the cycle is finished, A and B contain the SCC solutions."""
#
#    def __init__(self, solver_a, varname_a,
#                 solver_b, varname_b,
#                 maxiter, tolerance,
#                 alpha=1.0, niter=None, errfunc=None):
#
#        """
#        Args
#            solver_a (Solver): main solver (convergence is checked on this)
#            varname_a: any variable from solver_a supporting product and sum
#            solver_b (Solver): second solver
#            varname_b: any variable from solver_b supporting product and sum
#            mixing_parameter (float): linear mixer weight. 1.0 corresponds to
#                no mixing with the old solution.
#            tolerance (float): exit tolerance condition
#            errfun: function used to evaluate the error. If not specified,
#                    err=max(abs(a(n+1)-a(n)))
#
#        Returns
#            LinearMixerSCC instance
#
#        """
#
#        self.solver_a = solver_a
#        self.solver_b = solver_b
#        self.varname_a = varname_a
#        self.varname_b = varname_b
#        self.tolerance = tolerance
#        self.maxiter = maxiter
#        self.niter = niter
#        self.alpha = alpha
#        self.errfunc = errfunc
#
#
#    def solve(self):
#        """
#        Solve the SCC problem with linear mixer
#        """
#        for n in range(self.maxiter):
#
#            error = self._do_single_iteration()
#            if error < self.tolerance or n == self.niter:
#                return
#
#
#    def _do_single_iteration(self):
#        """
#        Update the solvers from step n to step n+1
#        """
#        #Note: we need to keep the copy of the variable to mix and check
#        # convergence
#        var_a_n = np.copy(self.solver_a.get(self.varname_a))
#        ## Refresh solver B
#        self.solver_b.set(self.varname_b, None)
#        self.solver_b.get(self.varname_b)
#        ## Refresh A and mix solver A n+1
#        self.solver_a.set(self.varname_a, None)
#        var_a_new = self.solver_a.get(self.varname_a)
#        var_a_np1 = (1 - self.alpha) * var_a_n + self.alpha * var_a_new
#        self.solver_a.set(self.varname_a, var_a_np1)
#        if self.errfunc is None:
#            error = np.max(np.abs(var_a_np1 - var_a_n))
#        else:
#            pass
#
#        return
#

#class SCCMixer():
#    """A class to link self-consistently two inter-dependent solvers.
#    The parameters are passed through constructor. When the constructor is
#    invoked, the solvers must be still independent. Internally, a set method
#    is invoked. Self-consistency is then checked on the quantities specified
#    in input (more quantities can be converging at once).
#
#    Example: an instance foo_A of a solver class A depends on a solver class B.
#    The instance foo_B of the solver class B depends on A. The SCC is
#    considered solved when A.whatever is converged.
#
#    WITHOUT setting A.set('foo_B', B) and B.set('foo_') we call
#    >> SCCMixer = scc(A,B,varname='whatever',tol=1e-10,maxiter=1000)
#    >> scc.do()
#
#    After the cycle is finished, A and B contain the SCC solutions."""
#
#    def __init__(self, solver_a, solver_b, varname,
#                 b_in_a_name, a_in_b_name, tol,
#                 b_in_a_kwargs={}, a_in_b_kwargs={},
#                 maxiter=1000,
#                 niter=None,
#                 mixer=None,
#                 relative_tol=True,
#                 stats=False):
#        """
#        solver_a, solver_b : instances of solver class to be linked
#        b_in_a_name, b_in_a_kwargs : variable names and optional set arguments
#                                    to be used to clean dependencies of
#                                    instance b in solver a and set instance
#                                    b in solver a
#        a_in_b_name, a_in_b_kwargs : variable names and optional set arguments
#                                    to be used to set instance a in solver b
#        varname :   string or list of strings with variables to be verified
#                    for SCC convergence
#        tol : numerical tolerance
#        maxiter : maximum number of iterations
#        niter : if specified, exit anyway after niter iterations.
#                In this case ignore tolerance
#        mixer: None -> no mixer, A has always the recalculated value
#               {'type': 'linear', 'weight': 0.5} linear mixer A1 = weight*A1 + (1-weight)*A0
#        relative_tol : if True, the relative tolerance is considered
#        stats: if Yes, produce some output statistics
#        """
#
#        #Param
#        self.solver_a = solver_a
#        self.solver_b = solver_b
#        #varname is always stored as list
#        if not (type(varname) == str or type(varname) == list):
#            raise ValueError('varname must be a string or a list of strings')
#        if type(varname) == list:
#            for var in varname:
#                if not type(var) == str:
#                    raise ValueError('varname must be a list of strings')
#            self.varname = varname
#        if type(varname) == str:
#            self.varname = [varname]
#        self.a_in_b_name = a_in_b_name
#        self.b_in_a_name = b_in_a_name
#        self.a_in_b_kwargs = a_in_b_kwargs
#        self.b_in_a_kwargs = b_in_a_kwargs
#        #Tolerance is a list of double with the size of varname, or a single double
#        if not (type(tol) == float or type(tol) == list or type(tol) == np.ndarray):
#            raise ValueError('tol must be a string or a list of strings')
#        if type(tol) == list or type(tol) == np.ndarray:
#            for val in tol:
#                if not type(val) == float:
#                    raise ValueError('tol must be a list of float or a numpy.array')
#            self.tol = tol
#        if type(tol) == float:
#            self.tol = [tol]
#        assert (len(self.tol) == len(self.varname))
#        self.maxiter = maxiter
#        self.niter = niter
#        self.mixer = mixer
#        self.relative_tol = relative_tol
#        self.stats = stats
#
#    def do(self):
#        """
#        Run the scc calculation. When done, the solvers are
#        linked and contain the SCC solutions
#        """
#        solver_b = self.solver_b
#        solver_a = self.solver_a
#
#        # I assume that solver_a has not solver_b linked. We check if true, otherwise
#        # we remove it
#        if solver_b in solver_a.get(self.b_in_a_name):
#            solver_a.set(self.b_in_a_name, solver_b, mode='remove')
#        # We need to buffer the results of solver_a in a local solver to keep
#        # the result of previous iteration.
#        # Alternate implementation: make a deepcopy and you can avoid to explicitly call get(var)
#        local_a = copy.copy(solver_a)
#        for var in self.varname:
#            local_a.get(var)
#        # ----------------------------------------------------------------------------------------
#        solver_a.set(self.b_in_a_name, solver_b, **self.b_in_a_kwargs)
#
#        # If filled, it contains maximum value and diff for each iterations for each variable
#        if self.stats:
#            out_stats = np.ndarray(shape=(self.maxiter, len(self.varname) * 2))
#        else:
#            out_stats = None
#
#        for n in range(self.maxiter):
#            #Calculate Bn = f(An-1)
#            #print('n',n)
#            solver_b.set(self.a_in_b_name, local_a, **self.a_in_b_kwargs)
#            solver_a.cleandep(self.b_in_a_name)
#            if self.mixer is not None:
#                if self.mixer['type'] == 'linear':
#                    w = self.mixer['weight']
#                    assert (w <= 1.0 and w > 0.0)
#                    for var in self.varname:
#                        # TODO: the mixer does not work, Glesser diverges in SCBA
#                        # print('local max',(np.absolute(local_a.get(var)).max()))
#                        solver_a.set(var, solver_a.get(var) * w + (1.0 - w) * local_a.get(var))
#            #print('var ', var,' solver_a', solver_a.get(var), ' local ', local_a.get(var))
#
#            #Calculate difference between previous and current state
#            if self.relative_tol:
#                diff = np.array(
#                    [(np.absolute(solver_a.get(var) - local_a.get(var))).max() /
#                     (np.absolute(solver_a.get(var))).max()
#                     for var in self.varname])
#            else:
#                diff = np.array(
#                    [(np.absolute(solver_a.get(var) - local_a.get(var))).max()
#                     for var in self.varname])
#
#            #Build some statistics, if needed
#            if self.stats:
#                out_stats[n, ::2] = diff
#                maxval = np.array([(np.absolute(solver_a.get(var))).max() for var in self.varname])
#                out_stats[n, 1::2] = maxval
#            if not self.stats:
#                out_stats = None
#
#            #Verify exit condition
#            if self.niter is not None:
#                if n == self.niter - 1:
#                    if self.stats:
#                        return out_stats[:n, :]
#                    else:
#                        return
#            else:
#                if np.all(diff < np.array(self.tol)):
#                    if self.stats:
#                        return out_stats[:n, :]
#                    else:
#                        return
#            if n == self.maxiter - 1:
#                raise RuntimeError('Maximum number of iterations reached in SCCMixer')
#            local_a = copy.copy(solver_a)
#
#        return out_stats
#
#
#class SCBA():
#    """A class to solve Self Consistent Born Approximation Loop"""
#
#    def __init__(self,
#                 #Params
#                 greensolver, selfener, tol=defaults.scbatol, maxiter=1000,
#                 task='both', niter=None, mixer_parameter=1.0, stats=False):
#        """ greensolver and selfener are the solver
#        to be plugged in the loop.
#        Task specify whether we loop on the equilibrium ('eq') or the Keldysh
#        ('keldysh') or both.
#        If the tolerance is a single scalar, the same tolerance is applied to both """
#
#        #Param
#        self.greensolver = greensolver
#        self.selfener = selfener
#        self.tol = tol
#        self.maxiter = maxiter
#        self.task = task
#        self.niter = niter
#        self.task = task
#        self.mixer_parameter = mixer_parameter
#        self.stats = stats
#
#
#    def solve(self):
#        """Link the green and self energy solvers and run the scba loop"""
#
#        eq = False
#        keldysh = False
#        if self.task == 'keldysh':
#            keldysh = True
#        elif self.task == 'eq':
#            eq = True
#        elif self.task == 'both':
#            keldysh = True
#            eq = True
#        else:
#            raise ValueError('Unknown task. Task must be keldysh, eq or both')
#
#        if eq:
#            leads = self.greensolver.get('leads')
#            if self.selfener in leads:
#                leads.remove(self.selfener)
#                self.greensolver.set('leads', leads)
#            green_ret = self.greensolver.get('green_ret')
#            self.greensolver.set('leads', leads + [self.selfener])
#            self.greensolver.set('green_ret', green_ret)
#            sccmixer = LinearMixerSCC(self.greensolver, 'green_ret',
#                                      self.selfener, 'sigma_ret', self.maxiter,
#                                      self.tol, niter=self.niter)
#            sccmixer.solve()
#
#        if keldysh:
#            leads = self.greensolver.get('leads')
#            if self.selfener in leads:
#                leads.remove(self.selfener)
#                self.greensolver.set('leads', leads)
#            green_gr = self.greensolver.get('green_lr')
#            leads = self.greensolver.get('leads')
#            self.greensolver.set('leads', leads + [self.selfener])
#            self.greensolver.set('green_lr', green_gr)
#            sccmixer = LinearMixerSCC(self.greensolver, 'green_lr',
#                                      self.selfener, 'sigma_lr', self.maxiter,
#                                      self.tol, alpha=self.mixer_parameter,
#                                      niter=self.niter)
#            sccmixer.solve()
#
#        return
