import numpy as np
import pyta.defaults as defaults
import solver

class Green(solver.Solver):
    """A class for equilibrium Green's function"""
    def __init__(self, 
            #Parameters
            size):
        """
        Base class constructor only invoked by derived classes.

        invar:
        1) leads
            set of Leads
            overwrite, append

        parameters:
        1) size

        outvar:
        1) eqgreen
        2) green_gr
        3) green_lr
        4) spectral

        """

        #Invar
        self.leads = set()

        #Param
        self.size = size

        #Outvar
        self.eqgreen = None
        self.green_gr = None
        self.green_lr = None

    def _do_green_gr(self):
        """Calculate equilibrium Green's function"""
        sigma_gr = np.matrix(np.zeros((self._size, self._size), dtype=np.complex128))
        for lead in self._leads:
            sigma_gr = sigma_gr + lead.get_sigma_gr(resize = self.size)
        self.green_gr = self.get_eqgreen() * sigma_gr * self.get_eqgreen().H

        return self.green_gr

    def _do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.matrix(np.zeros((self._size, self._size), dtype=np.complex128))
        for lead in self._leads:
            sigma_lr = sigma_lr + lead.get_sigma_lr(resize = self.size)
        self.green_lr = self.get_eqgreen() * sigma_lr * self.get_eqgreen().H

        return self._green_lr


    def set_leads(self, leads, mode):
        """Add a Leads set"""
        assert(type(leads) == set)
        if mode == 'replace':
            self._leads = leads
        if mode == 'append':
            self.leads = self._leads | leads
        self.cleandep_leads()
        return

    def cleandep_leads(self):
        self.eqgreen = None
        self.green_gr = None
        self.green_lr = None

    def get_eqgreen(self, optargs):
        """Get equilibrium Green's function. If a green's function is already
        existing, it's returned without calculating again assumed that the
        energy has not changed."""

        if self.eqgreen is None:
            self._do_eqgreen()
        assert(not self.eqgreen is None)
        return self.eqgreen

    def get_green_gr(self):
        """Get greater Green's function. If a green's function is already
        existing, it's returned without calculating again assumed that the
        energy has not changed."""

        if self.green_gr is None:
            self._do_green_gr()
        assert(not self.green_gr is None)
        return self.green_gr

    def get_green_lr(self):
        """Get lesser Green's function. If a green's function is already
        existing, it's returned without calculating again assumed that the
        energy has not changed."""

        if self._green_lr is None:
            self._do_green_lr()
        assert(not self.green_lr is None)
        return self._reen_lr

    def get_spectral(self):
        """Get spectral function A = j(G^{r} - G^{a})"""
        eqgreen = self.get_eqgreen()
        spectral = 1j * (eqgreen - eqgreen.H)
        return spectral


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
           overwrite

        2) leads
            set of Leads
            overwrite, append

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

    def set_energy(self, energy, mode):
        """Set energy point"""
        assert(mode == 'replace')
        if energy != self.energy:
            self.energy = energy
            self.cleandep_energy()
            #Update energy point in all leads where set_energy is defined
            self._spread_energy(energy)

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
            else:
                lead.set_energy(energy)


    def _do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        es_h = self.energy * self.over - self.ham
            
        for lead in self.leads:
            es_h = es_h - lead.get_sigma(resize = self.size)
        self.eqgreen = es_h.I

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
        If masses are not specified, an identity matrix is assumed."""

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

    def set_frequency(self, frequency, mode):
        """Set energy point"""
        assert(mode == 'replace')
        if frequency != self.frequency:
            self.frequency = frequency
            self.cleandep_frequency()
            self._spread_frequency(frequency)

    def cleandep_frequency(self):
        self.eqgreen = None
        self.green_lr = None
        self.green_gr = None


    def _spread_frequency(self, frequency):
        "Distribute energy in leads"
        for lead in self.leads:
            try:
                lead.set_frequency(frequency)
            except AttributeError:
                pass
            else:
                lead.set_frequency(frequency)


    def do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        tmp = self.frequency * self.frequency * self.mass - self.spring
            
        for lead in self._leads:
            tmp = tmp - lead.get_sigma(resize = self.size)
        self.eqgreen = tmp.I

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
        self._green = greensolver
        self._selfener = selfener
        self._tol = tol
        self._maxiter = maxiter
        self._task = task

    def do(self):
        """Run the scba loop"""
        if self._task == 'both':
            for ind, scba in enumerate(range(self._maxiter)):
                green_buf = self._green.get_eqgreen()
                green_buf_lr = self._green.get_green_lr()
                self._selfener.set_eqgreen(self._green.get_eqgreen())
                self._selfener.set_green_lr(self._green.get_green_lr())
                self._selfener.set_green_gr(self._green.get_green_gr())
                self._green.set_lead(self._selfener)
                green_after = self._green.get_eqgreen()
                green_after_lr = self._green.get_green_lr()
                err1 = (green_after - green_buf).max()
                err2 = (green_after_lr - green_buf_lr).max()
                if (abs(err1)<self._tol) and (abs(err2)<self._tol):
                    return
            raise RuntimeError('SCBA loop not converged')
        if self._task == 'eq':
            for ind, scba in enumerate(range(self._maxiter)):
                green_buf = self._green.get_eqgreen()
                self._selfener.set_eqgreen(self._green.get_eqgreen())
                self._green.set_lead(self._selfener)
                green_after = self._green.get_eqgreen()
                err1 = (green_after - green_buf).max()
                if (abs(err1)<self._tol):
                    return
            raise RuntimeError('SCBA loop not converged')
        if self._task == 'keldysh':
            for ind, scba in enumerate(range(self._maxiter)):
                green_buf_lr = self._green.get_green_lr()
                self._selfener.set_eqgreen(self._green.get_eqgreen())
                self._selfener.set_green_lr(self._green.get_green_lr())
                self._green.set_lead(self._selfener)
                green_after = self._green.get_eqgreen()
                green_after_lr = self._green.get_green_lr()
                err2 = (green_after_lr - green_buf_lr).max()
                if (abs(err2)<self._tol):
                    return
            raise RuntimeError('SCBA loop not converged')
        
