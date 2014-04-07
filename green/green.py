import numpy as np
import pyta.core.defaults as defaults

class Green:
    """A class for equilibrium Green's function"""
    def __init__(self, 
            #Constant
            size,
            #Independent variables
            leads = None):
        """Base class constructor only invoked by derived classes."""

        #Invar
        if leads is None:
            self._leads = set()
        else:
            assert(type(leads) == list or type(leads) == set)
            self._leads = set(leads)
        
        #Param
        self._size = size

        #Outvar
        self._eqgreen = None
        self._green_gr = None
        self._green_lr = None

    def do_green_gr(self):
        """Calculate equilibrium Green's function"""
        sigma_gr = np.matrix(np.zeros((self._size, self._size), dtype=np.complex128))
        for lead in self._leads:
            sigma_gr = sigma_gr + lead.get_sigma_gr(resize = self._size)
        self._green_gr = self.get_eqgreen() * sigma_gr * self.get_eqgreen().H

        return self._green_gr

    def do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.matrix(np.zeros((self._size, self._size), dtype=np.complex128))
        for lead in self._leads:
            sigma_lr = sigma_lr + lead.get_sigma_lr(resize = self._size)
        self._green_lr = self.get_eqgreen() * sigma_lr * self.get_eqgreen().H

        return self._green_lr

    def set_leads(self, leads):
        """Add a Leads set"""
        assert(type(leads) == set)
        self._leads = leads
        self._eqgreen = None
        self._green_gr = None
        self._green_lr = None

    def set_lead(self, lead):
        """Add a Lead"""
        self._leads.add(lead)
        self._eqgreen = None
        self._green_gr = None
        self._green_lr = None

    def get_eqgreen(self):
        """Get equilibrium Green's function. If a green's function is already
        existing, it's returned without calculating again assumed that the
        energy has not changed."""

        if self._eqgreen is None:
            self.do_eqgreen()
        assert(not self._eqgreen is None)
        return self._eqgreen

    def get_green_gr(self):
        """Get greater Green's function. If a green's function is already
        existing, it's returned without calculating again assumed that the
        energy has not changed."""

        if self._green_gr is None:
            self.do_green_gr()
        assert(not self._green_gr is None)
        return self._green_gr

    def get_green_lr(self):
        """Get lesser Green's function. If a green's function is already
        existing, it's returned without calculating again assumed that the
        energy has not changed."""

        if self._green_lr is None:
            self.do_green_lr()
        assert(not self._green_lr is None)
        return self._green_lr

    def get_spectral(self):
        """Get spectral function A = j(G^{r} - G^{a})"""
        eqgreen = self.get_eqgreen()
        spectral = 1j * (eqgreen - eqgreen.H)
        return spectral


class GreenFermion(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self, 
            #Param
            ham, over = None,
            #Invar
            energy = 0.0, leads=None):
        """GreenFermion is initialized by specifying an Hamiltonian as numpy.matrix.
        Optionally, overlap can be specified.
        If overlap is not specified, an orthogonal basis is assumed."""

        #Param
        #========================================================
        assert(type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert(type(over) == np.matrixlib.defmatrix.matrix)
        self._ham = ham
        size = len(self._ham)
        if over is None:
            self._over = np.matrix(np.eye(size))

        #Invar
        #=================================
        self._energy = energy
        #=================================

        #Base constructor
        Green.__init__(self, size, leads)

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self._energy:
            self._energy = energy
            self._eqgreen = None
            self._green_lr = None
            self._green_gr = None
            #Update energy point in all leads where set_energy is defined
            for lead in self._leads:
                try:
                    lead.set_energy(energy)
                except AttributeError:
                    pass
                else:
                    lead.set_energy(energy)

    def do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        es_h = self._energy * self._over - self._ham
            
        for lead in self._leads:
            es_h = es_h - lead.get_sigma(resize = self._size)
        self._eqgreen = es_h.I

        return self._eqgreen


class GreenPhonon(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""
    
    def __init__(self, 
            #Param
            spring, mass=None,
            #Invar
            freq=None, leads=None):
        """GreenPhonon is initialized by specifying a coupling spring constant
        matrix as numpy.matrix.
        Optionally, masses can be specified.
        Masses must be specified as a diagonal numpy.matrix
        If masses are not specified, an identity matrix is assumed."""

        #Param
        #=======================================
        self._spring = spring
        self._mass = mass
        assert(type(spring) == np.matrixlib.defmatrix.matrix)
        if not (mass is None):
            assert(type(spring) == np.matrixlib.defmatrix.matrix)
        self._spring = spring
        size = len(self._spring)
        if mass is None:
            self._mass = np.matrix(np.eye(size))
        #======================================

        #Invar
        #=======================================
        self._freq = None
        #=======================================
        
        #Base constructor
        Green.__init__(self, size, leads)

    def set_freq(self, freq):
        """Set energy point"""
        if freq != self._freq:
            self._freq = freq
            self._eqgreen = None
            self._green_lr = None
            self._green_gr = None
            #Update energy point in all leads
            for lead in self._leads:
                try:
                    lead.set_freq(freq)
                except AttributeError:
                    pass
                else:
                    lead.set_freq(freq)

    def do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        tmp = self._freq * self._freq * self._mass - self._spring
            
        for lead in self._leads:
            tmp = tmp - lead.get_sigma(resize = self._size)
        self._eqgreen = tmp.I

        return self._eqgreen


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
        if self._task == 'eq':
            for ind, scba in enumerate(range(self._maxiter)):
                green_buf = self._green.get_eqgreen()
                self._selfener.set_eqgreen(self._green.get_eqgreen())
                self._green.set_lead(self._selfener)
                green_after = self._green.get_eqgreen()
                err1 = (green_after - green_buf).max()
                if (abs(err1)<self._tol):
                    return
        if self._task == 'keldysh':
            for ind, scba in enumerate(range(self._maxiter)):
                green_buf_lr = self._green.get_green_lr()
                self._selfener.set_eqgreen(self._green.get_eqgreen())
                self._selfener.set_green_lr(self._green.get_green_lr())
                self._green.set_lead(self._selfener)
                self._green.reset()
                green_after = self._green.get_eqgreen()
                green_after_lr = self._green.get_green_lr()
                err2 = (green_after_lr - green_buf_lr).max()
                if (abs(err2)<self._tol):
                    return
        
