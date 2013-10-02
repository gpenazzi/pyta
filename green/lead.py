import numpy as np
import pyta.core.defaults as defaults
import pyta.stats as stats
import pyta.core.consts as consts

#I use this function to decorate matrices
def resize_matrix(n, pos, mat):
    """Resize mat as nxn matrix but include the matrix mat starting from position pos"""
    if mat.shape == (n, n):
        return mat
    else:
        assert(mat.shape[0] == mat.shape[1])
        size = mat.shape[0]
        tmp = np.matrix(np.zeros((n, n)), dtype=mat.dtype)
        tmp[pos:pos + size, pos:pos + size] = mat[:, :]
        return tmp

class Lead:
    """A base class for managing and building up real and virtual leads
    (inscattering and outscattering sources)."""

    def __init__(self, 
            #Constants
            position, size):
        """
        position (int): index of interacting device layer"""

        #Constants
        #=====================================
        self._position = position
        self._size = size
        #=====================================

        #Dependent variables
        #Note: not all the dependent variables are set in runtime as a member
        #some subclass calculates them on-the-fly directly during every get
        #(gamma is a typical example gamma = sigma - sigma.H)
        #================================
        self._sigma = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._gamma = None
        #================================
    
    def get_sigma(self, resize = None):
        if self._sigma is None:
            self._do_sigma()
        if not resize is None:
            return resize_matrix(resize, self._position, self._sigma)
        else:
            return self._sigma

    def get_sigma_gr(self, resize = None):
        if self._sigma_gr is None:
            self._do_sigma_gr()
        if not resize is None:
            return resize_matrix(resize, self._position, self._sigma_gr)
        else:
            return self._sigma_gr

    def get_sigma_lr(self, resize = None):
        if self._sigma_lr is None:
            self._do_sigma_lr()
        if not resize is None:
            return resize_matrix(resize, self._position, self._sigma_lr)
        else:
            return self._sigma_lr

    def get_gamma(self, resize = None):
        if self._gamma is None:
            self._do_gamma()
        if not resize is None:
            return resize_matrix(resize, self._position, self._gamma)
        else:
            return self._gamma

class MRDephasing(Lead):
    """A Lead modelling Momentum relaxing dephasing"""

    def __init__(self, 
            #Constants
            #Independent variables
            deph, eqgreen=None, green_gr=None):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()"""
        
        #Constants
        #================================
        #================================

        #Independent variables
        #================================
        assert(type(deph) == np.ndarray)
        self._deph = deph
        self._green_gr = green_gr
        self._eqgreen = eqgreen
        #================================

        #Base constructors 
        position = 0
        size = deph.size
        Lead.__init__(self, position, size)

    def set_deph(self, deph):
        """Set a new dephasing parameter"""
        assert(type(deph) == np.ndarray)
        assert(deph.size == self._size)
        self._sigma = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._deph = deph
            
    def set_eqgreen(self, eqgreen):
        """Set an equilibrium Green's function"""
        self._sigma = None
        self._eqgreen = eqgreen
        self._size = self._eqgreen.shape[0]

    def set_green_gr(self, green_gr):
        """Set a non-equilibrium Green's function (Greater)"""
        self._sigma_gr = None
        self._green_gr = green_gr
        self._size = self._green_gr.shape[0]

    def set_green_lr(self, green_lr):
        """Set a non-equilibrium Green's function (Lesser)"""
        self._sigma_lr = None
        self._green_lr = green_lr
        self._size = self._green_lr.shape[0]

    def _do_sigma(self):
        """Calculate the retarded self energy"""
        assert(not self._eqgreen is None)
        tmp = np.matrix(np.eye(self._size), dtype=np.complex128)
        #Note: * is an elementwise operator for ndarray types 
        np.fill_diagonal(tmp, np.multiply(self._eqgreen.diagonal(), self._deph))
        self._sigma = tmp

    def _do_sigma_gr(self):
        """Calculate the retarded self energy"""
        assert(not self._green_gr is None)
        tmp = np.matrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self._green_gr.diagonal(), self._deph))
        self._sigma_gr = tmp 

    def _do_sigma_lr(self):
        """Calculate the retarded self energy"""
        assert(not self._green_lr is None)
        tmp = np.matrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self._green_lr.diagonal(), self._deph))
        self._sigma_lr = tmp


class MCDephasing(Lead):
    """A Lead modelling Momentum conserving dephasing"""

    def __init__(self, 
            #Constants
            #Independent variables
            deph, eqgreen=None, green_gr=None):
        """Only the dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()"""

        #Constants
        #================================
        #================================

        #Independent variables
        #================================
        assert(type(deph) == np.array)
        self._deph = deph
        self._green_gr = green_gr
        self._eqgreen = eqgreen
        self._green_lr = green_lr
        #================================

        #Base constructors 
        position = 0
        size = deph.size()
        Lead.__init__(self, position, size)

    def set_deph(self, deph):
        """Set a new dephasing parameter"""
        assert(type(deph) == np.array)
        assert(deph.size == self._size)
        self._sigma = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._deph = deph

    def set_eqgreen(self, eqgreen):
        """Set an equilibrium Green's function"""
        self._sigma = None
        self._eqgreen = eqgreen
        self._size = self._eqgreen.shape[0]

    def set_green_gr(self, green_gr):
        """Set a non-equilibrium Green's function (Greater)"""
        self._sigma_gr = None
        self._green_gr = green_gr
        self._size = self._green_gr.shape[0]

    def _do_sigma(self):
        """Calculate the retarded self energy"""
        assert(not self._eqgreen is None)
        self._sigma = self._eqgreen * self._deph

    def _do_sigma_gr(self):
        """Calculate the greater self energy"""
        assert(not self._green_gr is None)
        self._sigma_gr = self._green_gr * self._deph

    def _do_sigma_lr(self):
        """Calculate the greater self energy"""
        assert(not self._green_lr is None)
        self._sigma_lr = self._green_gr.H * self._deph


class PhysicalLead(Lead):
    """A class derived from Lead for the description of physical contacts"""

    def __init__(self,
            #Constants
            position, size, pl_size,
            #Independent variables
            mu=0.0, temp=0.0, delta=defaults.delta):
        """A PhysicalLead object describe a semi-infinite periodic lead
        within the Open Boundary
        Steady State picture. 
        Derived class for Fermion and Bosons are implemented, this base class
        should not be explicitely instanced.
        position (int): index of interacting device layer
        mu (float): chemical potential
        We always mean by convention the
        coupling device-contact, i.e. Hdc"""

        #Constants
        #================================
        self._pl_size = pl_size
        #================================
        
        #Independent variables
        #================================
        self._mu = mu
        self._temp = temp
        self._delta = delta
        #================================
        
        #Base constructors
        Lead.__init__(self, position, size)

    def set_mu(self, mu):
        """Set a chemical potential, for nonequilibrium self energy"""
        self._mu = mu

    def set_temp(self, temp):
        """Set temperature, for nonequilibrium self energy"""
        self._temp = temp

    def get_gamma(self, resize = None):
        "Return \Gamma=j(\Sigma^{r} - \Sigma^{a})"""
        gamma = 1.j * (self.get_sigma() - self.get_sigma().H)
        if not resize is None:
            return resize_matrix(resize, self._position, gamma)
        else:
            return gamma

class PhysicalLeadFermion(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""
    def __init__(self, 
            #Constants
            position, ham, ham_t, ham_ld, over=None,
                 over_t=None, over_ld=None, 
            #Independent variables
                 energy = 0.0, mu=0.0, temp=0.0, delta=defaults.delta):
        """
        The following quantities must be specified:

        ham (np.matrix): Hamiltonian for a single layer
        over (np.matrix), optional: Overlap for a single layer
        ham_t (np.matrix): coupling matrix between single layers
        over_t (np.matrix): overlap in the coupling block between single
        layers.If none, it's set tozero
        ham_ld (np.matrix): coupling between lead and device
        over_ld (np.matrix): overlap in device-lead coupling
        position (int): index of interacting device layer
        mu (float): chemical potential

        We always mean by convention the
        coupling device-contact, i.e. Hcd 
        For the contact we specify coupling between first and second layer, i.e.
        H10 (same for the overlap, if any)"""
        

        #Constants
        #==========================================================
        self._ham = ham
        self._ham_t = ham_t
        self._ham_ld = ham_ld
        self._over = over
        self._over_t = over_t
        self._over_ld = over_ld
        #Some check
        assert(type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert(type(over) == np.matrixlib.defmatrix.matrix)
        #H must be a square matrix
        assert(self._ham.shape[0] == self._ham.shape[1])

        #Set defaults
        pl_size = self._ham.shape[0]
        if not over:
            self._over = np.matrix(np.eye(pl_size))
        if not over_t:
            self._over_t = np.matrix(np.zeros(self._ham_t.shape))
        if not over_ld:
            self._over_ld = np.matrix(np.zeros(self._ham_ld.shape))
        #===========================================================

        #Independent variables
        #============================
        self._energy = energy
        #============================
   
        #Base constructor
        size = self._ham_t.shape[0]
        PhysicalLead.__init__(self, position, size, pl_size, mu=mu, 
                temp=temp, delta=delta)

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self._energy:
            self._energy = energy
            self._sigma = None
            self._sigma_lr = None
            self._sigma_gr = None
    
    def _do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.
        This implementation follows Lopez-Sancho

        Note: modified from ASE implementation"""

        self._delta
        z = self._energy + self._delta * 1j
        #TODO: Verify this!!
        d_00 = z * self._over.H - self._ham.H
        d_11 = d_00.copy()
        d_10 = z * self._over_t - self._ham_t
        d_01 = d_10.H
        delta = tol + 1
        while delta > tol:
            a = np.linalg.solve(d_11, d_01)
            b = np.linalg.solve(d_11, d_10)
            d_01_dot_b = np.dot(d_01, b)
            d_00 -= d_01_dot_b
            d_11 -= np.dot(d_10, a)
            d_11 -= d_01_dot_b
            d_01 = -np.dot(d_01, a)
            d_10 = -np.dot(d_10, b)
            delta = abs(d_01).max()

        return d_00

    def _do_sigma(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        z = self._energy 
        tau_ld = z * self._over_ld - self._ham_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = z * self._over_ld.H - self._ham_ld.H
        self._sigma = np.dot(tau_dl, a_ld)
        return self._sigma

    def _do_sigma_lr(self, resize = None):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        self._sigma_lr = (stats.fermi(self._energy, self._mu, temppot=self._temp) *
                    1j * self.get_gamma())

    def _do_sigma_gr(self, resize = None):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        self._sigma_gr = ((stats.fermi(self._energy, self._mu, temppot=self._temp) 
                    - 1.0) * 1j * self.get_gamma())


class PhysicalLeadPhonon(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""
    def __init__(self, 
            #Constants
            position, spring, spring_t, spring_ld, mass=None,
            #Independent variables
            freq = 1.0, temp=0.0,delta=defaults.delta):

        #Constants
        #======================================
        self._spring = spring
        self._mass = mass
        self._spring_t = spring_t
        self._spring_ld = spring_ld
        #PL size n x n
        self._pl_size = self._spring.shape[0]
        #Interaction layer size tn x tm
        self._size = self._spring_t.shape[0]
        #Some checks
        assert(type(spring) == np.matrixlib.defmatrix.matrix)
        if mass:
            assert(type(mass) == np.matrixlib.defmatrix.matrix)
        #H must be a square matrix
        assert(self._spring.shape[0] == self._spring.shape[1])
        pl_size = self._spring.shape[0]
        if not mass:
            self._mass = np.matrix(np.eye(pl_size))
        #======================================
    
        #Independent variables
        #===================================
        self._freq = freq
        #===================================

        #Base constructor
        size = self._spring_t.shape[0]
        mu = 0.0
        PhysicalLead.__init__(self, position, size, pl_size, mu=mu, 
                temp=temp, delta=delta)
    
    def set_freq(self, freq):
        """Set frequency point"""
        if freq != self._freq:
            self._freq = freq
            self._sigma = None
            self._sigma_lr = None
            self._sigma_lr = None

    def _do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.

        Note: from ASE implementation
        slightly adapted for phonons
        
        Note: frequencies are given in fs^-1, energies in eV"""

        z = self._freq * self._freq + self._delta * 1j
        #TODO: Verify this!!
        d_00 = z * self._mass - self._spring
        d_11 = d_00.copy()
        d_10 = - self._spring_t
        d_01 = - self._spring_t.H
        delta = tol + 1
        while delta > tol:
            a = np.linalg.solve(d_11, d_01)
            b = np.linalg.solve(d_11, d_10)
            d_01_dot_b = np.dot(d_01, b)
            d_00 -= d_01_dot_b
            d_11 -= np.dot(d_10, a)
            d_11 -= d_01_dot_b
            d_01 = -np.dot(d_01, a)
            d_10 = -np.dot(d_10, b)
            delta = abs(d_01).max()
        return d_00

    def _do_sigma(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        z = self._freq * self._freq
        tau_ld = self._spring_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = self._spring_ld.H
        self._sigma = np.dot(tau_dl, a_ld)
        return self._sigma

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        energy = self._freq * consts.hbar_eV_fs
        self._sigma_lr = ((stats.bose(energy, self._mu, temppot=self._temp)) * 
                    (-1j) * self.get_gamma())

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        energy = self._freq * consts.hbar_eV_fs
        self._sigma_gr = ((stats.bose(energy, self._mu, temppot=self._temp)
                    + 1.0) * (-1j) * self.get_gamma())
