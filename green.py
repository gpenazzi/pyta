import numpy as np
import pyta.defaults as defaults
import solver
import pyta.lead
import copy

class Green(solver.Solver):
    """A class for equilibrium Green's function

        invar:
        1) leads
            list of Leads
            overwrite, append

        parameters:
        1) size

        outvar:
        1) eqgreen
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
        self.eqgreen = None
        self.green_gr = None
        self.green_lr = None
        self.transmission = None

    def get_eqgreen(self):
        if self.eqgreen is None:
            self._do_eqgreen()
        return self.eqgreen

    def get_green_lr(self):
        if self.green_lr is None:
            self._do_green_lr()
        return self.green_lr

    def get_green_gr(self):
        if self.green_gr is None:
            self._do_green_gr()
        return self.green_gr

    def _do_eqgreen(self):
        raise RuntimeError('Base Class Green has no _do_eqgreen method')

    def _do_green_gr(self):
        """Calculate equilibrium Green's function"""
        sigma_gr = np.matrix(np.zeros((self.size, self.size), dtype=np.complex128))
        self._add_leads(sigma_gr, 'sigma_gr')
        self.green_gr = self.get_eqgreen() * sigma_gr * self.get_eqgreen().H
        return

    def _do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.matrix(np.zeros((self.size, self.size), dtype=np.complex128))
        self._add_leads(sigma_lr, 'sigma_lr')
        self.green_lr = self.get_eqgreen() * sigma_lr * self.get_eqgreen().H
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
        self.eqgreen = None
        self.green_gr = None
        self.green_lr = None


    def get_spectral(self):
        """Get spectral function A = j(G^{r} - G^{a})"""
        eqgreen = self.get_eqgreen()
        spectral = 1j * (eqgreen - eqgreen.H)
        return spectral

    def _add_leads(self, mat, varname):
        """Subtract all the leads contributions from matrix mat.
         Result in place. """
        for lead in self.leads:
            sigma = lead.get(varname)
            assert(sigma.shape[0]==sigma.shape[1])
            size = sigma.shape[0]
            pos = lead.get('position')
            mat[pos: pos+size, pos:pos+size] -= sigma
        return


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
        eqgreen = self.get_eqgreen()
        trans = (np.trace(gamma1 * eqgreen * gamma2 * eqgreen.H))
        return trans


    def get_occupation(self):
        """Calculate the occupation by comparing the lesser green function
        and the spectral function. """
        diag1 = self.get_green_lr()
        diag2 = self.get_spectral()
        occupation = (np.imag(diag1/ diag2))
        return occupation

class GreenFermion(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self, 
            #Parameters
            ham, over = None):
        """GreenFermion is initialized by specifying an Hamiltonian as numpy.matrix.
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
        Green.__init__(self, size)

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.cleandep_energy()
            #Update energy point in all leads where set_energy is defined
            self._spread_energy(energy)
        return

    def cleandep_energy(self):
        self.eqgreen = None
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

    def _do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        esh = self.energy * self.over - self.ham
        self._add_leads(esh, 'sigma')
        self.eqgreen = esh.I

        return self.eqgreen


class GreenPhonon(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""
    
    def __init__(self, 
            #Param
            spring, mass=None):
        """GreenPhonon is initialized by specifying a coupling spring constant
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
        Green.__init__(self, size)

    def set_frequency(self, frequency):
        """Set energy point"""
        if frequency != self.frequency:
            self.frequency = frequency
            self.cleandep_frequency()
            self._spread_frequency(frequency)

    def cleandep_frequency(self):
        self.eqgreen = None
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


    def _do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        esh = self.frequency * self.frequency * self.mass - self.spring
        self._add_lead_sigma(esh)
        self.eqgreen = esh.I

        return self.eqgreen


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
            green_buf = green.get('eqgreen')
            green_buf_lr = green.get('green_lr')
            local.set('leads', selfener, mode='append')
            green_local = local.get('eqgreen')
            green_local_lr = local.get('green_lr')
            green.set('leads', selfener, mode='append')
            #Now append the scba
            for ind, scba in enumerate(range(self.maxiter)):
                err1 = (green_local - green_buf).max()
                err2 = (green_local_lr - green_buf_lr).max()
                if (abs(err1)<self.tol) and (abs(err2)<self.tol):
                    green.set('leads', selfener, mode='remove')
                    return
                selfener.set('greensolver', local)
                green.cleandep('leads')
                green_buf = green.get('eqgreen')
                green_buf_lr = green.get('green_lr')
                selfener.set('greensolver', green)
                local.cleandep('leads')
                green_local = local.get('eqgreen')
                green_local_lr = local.get('green_lr')
            raise RuntimeError('SCBA loop not converged')

        if self.task == 'eq':
            green_buf = green.get('eqgreen')
            for ind, scba in enumerate(range(self.maxiter)):
                green_buf = self.green.get_eqgreen()
                self.selfener.set_eqgreen(self.green.get_eqgreen())
                self.green.set_lead(self.selfener)
                green_after = self.green.get_eqgreen()
                err1 = (green_after - green_buf).max()
                if (abs(err1)<self.tol):
                    return
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
