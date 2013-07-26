import numpy as np
import pyta.core.defaults as defaults
import pyta.stats as stats
import pyta.core.consts as consts

class Lead:
    """A base class for managing and building up real and virtual leads
    (inscattering and outscattering sources)."""

    def __init__(self, name, position):
        """A name to associate to the lead is always needed (e.g. 'left',
        'phonon')
        position (int): index of interacting device layer"""

        self._type = None
        self._name = name
        self._position = position
        #Size of the self energy. Must be correctly set by subclass
        self._size = None

    def get_name(self):
        """Return a unique name for the lead"""
        return self._name

    def set_energy(self, energy):
        """By default set_energy is a dummy function, not all the self energy
        are explicitely energy dependent (WideBand, momentum relaxing etc.)"""
        return 0
    
    def set_freq(self, freq):
        """By default set_freq is a dummy function, not all the self energy
        are explicitely frequency dependent (WideBand, momentum relaxing etc.)"""
        return 0

    def get_size(self):
        assert(not self._size is None)
        return self._size

    def get_position(self):
        assert(not self._position is None)
        return self._position


class MRDephasing(Lead):
    """A Lead modelling Momentum relaxing dephasing"""

    def __init__(self, name, deph, eqgreen=None, green_gr=None):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
    to get the self energies using set_eqgreen() and set_neqgreen()"""
        position = 0
        Lead.__init__(self, name, position)
        self._sigma = None
        self._green_gr = green_gr
        self._eqgreen = eqgreen
        self._deph = deph
        self._size = None
        self._sigma_gr = None
        self._sigma_lr = None

    def set_deph(self, deph):
        """Set a new dephasing parameter"""
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

    def do_sigma(self):
        """Calculate the retarded self energy"""
        assert(not self._eqgreen is None)
        tmp = np.matrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, self._eqgreen.diagonal())
        self._sigma = tmp * self._deph

    def do_sigma_gr(self):
        """Calculate the retarded self energy"""
        assert(not self._green_gr is None)
        tmp = np.matrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, self._green_gr.diagonal())
        self._sigma_gr = tmp * self._deph

    def do_sigma_lr(self):
        """Calculate the retarded self energy"""
        assert(not self._green_lr is None)
        tmp = np.matrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, self._green_lr.diagonal())
        self._sigma_lr = tmp * self._deph

    def get_sigma(self):
        if self._sigma is None:
            self.do_sigma()
        return self._sigma

    def get_sigma_gr(self):
        if self._sigma_gr is None:
            self.do_sigma_gr()
        return self._sigma_gr

    def get_sigma_lr(self):
        if self._sigma_lr is None:
            self.do_sigma_lr()
        return self._sigma_lr


class MCDephasing(Lead):
    """A Lead modelling Momentum conserving dephasing"""

    def __init__(self, name, deph, eqgreen=None, green_gr=None):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()"""
        position = 0
        Lead.__init__(self, name, position)
        self._sigma = None
        self._green_gr = green_gr
        self._eqgreen = eqgreen
        self._deph = deph
        self._size = None
        self._sigma_gr = None
        self._sigma_lr = None

    def set_deph(self, deph):
        """Set a new dephasing parameter"""
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

    def do_sigma(self):
        """Calculate the retarded self energy"""
        assert(not self._eqgreen is None)
        self._sigma = self._eqgreen * self._deph

    def do_sigma_gr(self):
        """Calculate the greater self energy"""
        assert(not self._green_gr is None)
        self._sigma_gr = self._green_gr * self._deph

    def do_sigma_lr(self):
        """Calculate the greater self energy"""
        assert(not self._green_lr is None)
        self._sigma_lr = self._green_lr * self._deph

    def get_sigma(self):
        if self._sigma is None:
            self.do_sigma()
        return self._sigma

    def get_sigma_gr(self):
        if self._sigma_gr is None:
            self.do_sigma_gr()
        return self._sigma_gr

    def get_sigma_lr(self):
        if self._sigma_lr is None:
            self.do_sigma_lr()
        return self._sigma_lr


class PhysicalLead(Lead):
    """A class derived from Lead for the description of physical contacts"""

    def __init__(self, name, position, mu=None, temp=0.0,
                 delta=defaults.delta):
        """A PhysicalLead object describe a semi-infinite periodic lead
        within the Open Boundary
        Steady State picture. 
        Derived class for Fermion and Bosons are implemented, this base class
        should not be explicitely instanced.
        position (int): index of interacting device layer
        mu (float): chemical potential
        We always mean by convention the
        coupling device-contact, i.e. Hdc"""

        #Input variables
        self._mu = mu
        self._temp = temp
        self._delta = delta
        
        Lead.__init__(self, name, position)

        #These variables are set to none if they have not been calculated,
        #or set to
        #none if something changed and their value is not valid anymore (es.
        #different energy point)
        self._sigma = None

        #These must be assigned by derived classes
        self._pl_size = None
        self._size = None

    def set_mu(self, mu):
        """Set a chemical potential, for nonequilibrium self energy"""
        self._mu = mu

    def set_temp(self, temp):
        """Set temperature, for nonequilibrium self energy"""
        self._temp = temp

    def get_sigma(self):
        if self._sigma is None:
            self._do_sigma()
        return self._sigma

    def get_size(self):
        """Get size of Sigma"""
        return self._size

    def get_gamma(self):
        "Return \Gamma=j(\Sigma^{r} - \Sigma^{a})"""
        return 1.j * (self.get_sigma() - self.get_sigma().H)


class PhysicalLeadFermion(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""
    def __init__(self, name, ham, ham_t, ham_ld, position, over=None,
                 over_t=None, over_ld=None, mu=0.0, temp=0.0,
                 delta=defaults.delta):
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
        
        tmp_mu = mu
        tmp_temp = temp
        tmp_delta = delta
        PhysicalLead.__init__(self, name, position, mu=tmp_mu, 
                temp=tmp_temp, delta=tmp_delta)

        #Input variables
        self._ham = ham
        self._over = over
        self._ham_t = ham_t
        self._over_t = over_t
        self._ham_ld = ham_ld
        self._over_ld = over_ld
        #PL size n x n
        self._pl_size = self._ham.shape[0]
        #Interaction layer size tn x tm
        self._size = self._ham_t.shape[0]

        #Some check
        assert(type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert(type(over) == np.matrixlib.defmatrix.matrix)
        #H must be a square matrix
        assert(self._ham.shape[0] == self._ham.shape[1])

        #Set defaults
        if not over:
            self._over = np.matrix(np.eye(self._pl_size))
        if not over_t:
            self._over_t = np.matrix(np.zeros(self._ham_t.shape))
        if not over_ld:
            self._over_ld = np.matrix(np.zeros(self._ham_ld.shape))

        #Local variables
        self._energy = None
    
    def set_energy(self, energy):
        """Set energy point"""
        if energy != self._energy:
            self._energy = energy
            self._sigma = None
    
    def do_invsurfgreen(self, tol=defaults.surfgreen_tol):
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
        a_ld = np.linalg.solve(self.do_invsurfgreen(), tau_ld)
        tau_dl = z * self._over_ld.H - self._ham_ld.H
        self._sigma = np.dot(tau_dl, a_ld)
        return self._sigma

    def get_sigma_gr(self):
        """Calculate the Sigma greater"""
        assert(not self._mu is None)
        return ((stats.fermi(self._energy, self._mu, temppot=self._temp) - 1.0)
                * (1j) * self.get_gamma())

    def get_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        return (stats.fermi(self._energy, self._mu, temppot=self._temp) *
                    1j * self.get_gamma())
            
    def get_inscattering(self):
        """Calculate the inscattering Sigma (Datta notation)"""
        assert(not self._mu is None)
        return (stats.fermi(self._energy, self._mu, temppot=self._temp) *
                            self.get_gamma())

    def get_outscattering(self):
        """Calculate the outscattering Sigma (Datta notation)"""
        assert(not self._mu is None)
        if self._particle == "fermion":
            return ((1.0 - stats.fermi(self._energy, self._mu,
                     temppot=self._temp)) * self.get_gamma())


class PhysicalLeadPhonon(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""
    def __init__(self, name, spring, spring_t, spring_ld, position, mass=None,
                 temp=0.0,delta=defaults.delta):

        tmp_temp = temp
        tmp_mu = 0.0
        tmp_delta = delta
        PhysicalLead.__init__(self, name, position, mu=tmp_mu, 
                temp=tmp_temp, delta=tmp_delta)

        #Local variable
        self._freq = None
        
        #Input variables
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
        if not mass:
            self._mass = np.matrix(np.eye(self._pl_size))
    
    def set_freq(self, freq):
        """Set frequency point"""
        if freq != self._freq:
            self._freq = freq
            self._sigma = None
    
    def do_invsurfgreen(self, tol=defaults.surfgreen_tol):
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
        a_ld = np.linalg.solve(self.do_invsurfgreen(), tau_ld)
        tau_dl = self._spring_ld.H
        self._sigma = np.dot(tau_dl, a_ld)
        return self._sigma

    def get_sigma_gr(self):
        """Calculate the Sigma greater"""
        assert(not self._mu is None)
        energy = self._freq * consts.hbar_eV_fs
        return ((stats.bose(energy, self._mu, temppot=self._temp) +
                 1.0) * (-1j) * self.get_gamma())

    def get_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        energy = self._freq * consts.hbar_eV_fs
        return ((stats.bose(energy, self._mu, temppot=self._temp)) * 
                    (-1j) * self.get_gamma())

