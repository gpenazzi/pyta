import numpy as np
import pyta.core.defaults as defaults
import pyta.stats as stats


class Lead:
    """A base class for managing and building up real and virtual leads
    (inscattering and outscattering sources)."""

    def __init__(self, name, position, delta=defaults.delta):
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

    def __init__(self, name, ham, ham_t, ham_dl, position, over=None,
                 over_t=None, over_dl=None, mu=None, temp=0.0,
                 delta=defaults.delta):
        """A PhysicalLead object describe a semi-infinite periodic lead
        within the Open Boundary
        Steady State picture. The following quantities must be specified:

        ham (np.matrix): Hamiltonian for a single layer
        over (np.matrix), optional: Overlap for a single layer
        ham_t (np.matrix): coupling matrix between single layers
        over_t (np.matrix): overlap in the coupling block between single
        layers.If none, it's set tozero
        ham_dl (np.matrix): coupling between lead and device
        over_lld (np.matrix): overlap in device-lead coupling
        position (int): index of interacting device layer
        mu (float): chemical potential
        We always mean by convention the
        coupling device-contact, i.e. Hdc"""
        self._ham = ham
        self._over = over
        self._pl_size = None
        self._size = None
        self._ham_t = ham_t
        self._over_t = over_t
        self._ham_dl = ham_dl
        self._over_dl = over_dl
        self._energy = None
        self._mu = mu
        self._temp = temp
        self._delta = delta
        Lead.__init__(self, name, position, delta)

        assert(type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert(type(over) == np.matrixlib.defmatrix.matrix)
        self._ham = ham
        #PL size n x n
        self._pl_size = self._ham.shape[0]
        #H must be a square matrix
        assert(self._ham.shape[0] == self._ham.shape[1])
        if not over:
            self._over = np.matrix(np.eye(self._pl_size))
        self._ham_t = ham_t
        #Interaction layer size tn x tm
        self._size = self._ham_t.shape[0]
        if not over_t:
            self._over_t = np.matrix(np.zeros(self._ham_t.shape))
        if not over_dl:
            self._over_dl = np.matrix(np.zeros(self._ham_dl.shape))
        self._delta = delta

        #These variables are set to none if they have not been calculated,
        #or set to
        #none if something changed and their value is not valid anymore (es.
        #different energy point)
        self._sigma = None

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self._energy:
            self._energy = energy
            self._sigma = None

    def set_mu(self, mu):
        """Set a chemical potential, for nonequilibrium self energy"""
        self._mu = mu

    def set_temp(self, temp):
        """Set temperature, for nonequilibrium self energy"""
        self._temp = temp

    def do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.

        Note: from ASE implementation"""

        self._delta
        z = self._energy + self._delta * 1j
        #TODO: Verify this!!
        v_00 = z * self._over.H - self._ham.H
        v_11 = v_00.copy()
        v_10 = z * self._over_t - self._ham_t
        v_01 = z * self._over_t.H - self._ham_t.H
        delta = tol + 1
        while delta > tol:
            a = np.linalg.solve(v_11, v_01)
            b = np.linalg.solve(v_11, v_10)
            v_01_dot_b = np.dot(v_01, b)
            v_00 -= v_01_dot_b
            v_11 -= np.dot(v_10, a)
            v_11 -= v_01_dot_b
            v_01 = -np.dot(v_01, a)
            v_10 = -np.dot(v_10, b)
            delta = abs(v_01).max()

        return v_00

    def _do_sigma(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        z = self._energy + self._delta * 1.j
        tau_dl = z * self._over_dl - self._ham_dl
        a_dl = np.linalg.solve(self.do_invsurfgreen(), tau_dl)
        tau_ld = z * self._over_dl.T.conj() - self._ham_dl.T.conj()
        self._sigma = np.dot(tau_ld, a_dl)

        return self._sigma

    def get_sigma_gr(self):
        """Calculate the Sigma greater"""
        assert(not self._mu is None)
        return ((1.0 - stats.fermi(self._energy, self._mu, temppot=self._temp))
                * (-1j) * self.get_gamma())

    def get_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert(not self._mu is None)
        sigma_lr = (stats.fermi(self._energy, self._mu, temppot=self._temp) *
                    1j * self.get_gamma())
        return sigma_lr

    def get_inscattering(self):
        """Calculate the inscattering Sigma (Datta notation)"""
        assert(not self._mu is None)
        return stats.fermi(self._energy, self._mu) * self.get_gamma()

    def get_outscattering(self):
        """Calculate the outscattering Sigma (Datta notation)"""
        assert(not self._mu is None)
        return (1.0 - stats.fermi(self._energy, self._mu)) * self.get_gamma()

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

