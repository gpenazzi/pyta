import numpy as np
import pyta.core.defaults as defaults

def decorate(n, pos, mat):
    """Returned a nxn square decorated matrix where the matrix is placed
    according to lead position"""
    if mat.shape == (n, n):
        return mat
    else:
        assert(mat.shape[0] == mat.shape[1])
        size = mat.shape[0]
        tmp = np.matrix(np.zeros((n, n)), dtype=mat.dtype)
        tmp[pos:pos + size, pos:pos + size] = mat[:, :]
        return tmp


class Green:
    """A class for equilibrium Green's function"""
    def __init__(self, ham, over=None, delta=defaults.delta, particle=
            "fermion"):
        """EqGreen is initialized by specifying an Hamiltonian as numpy.matrix.
        Optionally, overlap and imaginary delta can be specified."""

        assert(type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert(type(over) == np.matrixlib.defmatrix.matrix)
        self._ham = ham
        self._n = len(self._ham)
        if over is None:
            self._over = np.matrix(np.eye(self._n))
        self._leads = dict()

        self._active_leads = None

        #Internal variables
        self._eqgreen = None
        self._green_gr = None
        self._green_lr = None

    def get_ham(self):
        return self._ham

    def get_over(self):
        return self._over

    def do_green_gr(self):
        """Calculate equilibrium Green's function"""
        sigma_gr = np.matrix(np.zeros((self._n, self._n), dtype=np.complex128))
        for key in self._active_leads:
            lead = self._leads[key]
            sigma_gr = (sigma_gr +
                        decorate(self._n, lead.get_position(),
                        lead.get_sigma_gr()))
        self._green_gr = self.get_eqgreen() * sigma_gr * self.get_eqgreen().H

        return self._green_gr

    def do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.matrix(np.zeros((self._n, self._n), dtype=np.complex128))
        for key in self._active_leads:
            lead = self._leads[key]
            sigma_lr = (sigma_lr +
                        decorate(self._n, lead.get_position(),
                        lead.get_sigma_lr()))
        self._green_lr = self.get_eqgreen() * sigma_lr * self.get_eqgreen().H

        return self._green_lr

    def add_lead(self, lead):
        """Add a Lead"""
        self._leads[lead.get_name()] = lead

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

    def set_active_leads(self, leads):
        self._active_leads = leads


class GreenFermion(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self, ham, over=None, delta=defaults.delta):
        
        Green.__init__(self, ham, over=over, delta=defaults.delta)
        self._energy = None

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self._energy:
            self._energy = energy
            self._eqgreen = None
            self._green_lr = None
            self._green_gr = None
            #Update energy point in all leads
            for key in self._leads:
                (self._leads[key]).set_energy(energy)

    def do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        es_h = self._energy * self._over - self._ham
            
        for key in self._active_leads:
            lead = self._leads[key]
            es_h = es_h - decorate(self._n, lead.get_position(),
                                   lead.get_sigma())
        self._eqgreen = es_h.I

        return self._eqgreen



class GreenPhonon(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""
    
    def __init__(self, ham, over=None, delta=defaults.delta):
        Green.__init__(self, ham, over=over, delta=defaults.delta)

        self._freq = None

    def set_freq(self, freq):
        """Set energy point"""
        if freq != self._freq:
            self._freq = freq
            self._eqgreen = None
            self._green_lr = None
            self._green_gr = None
            #Update energy point in all leads
            for key in self._leads:
                (self._leads[key]).set_freq(freq)

    def do_eqgreen(self):
        """Calculate equilibrium Green's function"""
        es_h = self._freq * self._freq * self._over - self._ham
            
        for key in self._active_leads:
            lead = self._leads[key]
            es_h = es_h - decorate(self._n, lead.get_position(),
                                   lead.get_sigma())
        self._eqgreen = es_h.I

        return self._eqgreen
