import numpy as np
import solver
import pyta.defaults as defaults
import pyta.dist as dist
import pyta.consts as consts

#I use this function to decorate matrices
#def resize_matrix(n, pos, mat):
#    """Resize mat as nxn matrix but include the matrix mat starting from position pos"""
#    if mat.shape == (n, n):
#        return mat
#    else:
#        assert(mat.shape[0] == mat.shape[1])
#        size = mat.shape[0]
#        tmp = np.matrix(np.zeros((n, n)), dtype=mat.dtype)
#        tmp[pos:pos + size, pos:pos + size] = mat[:, :]
#        return tmp


class Lead(solver.Solver):
    """A base class for managing and building up real and virtual leads. """

    def __init__(self,
                 #Param
                 position):
        """
        Base class (virtual like)

        :param position:
        outvar:
        1) sigma
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        (see derived classes)

        parameters:
        1) position
            index of interacting layer (int)

        internal:
        2) size
            index of system to be attached to

        """

        #Param
        #=====================================
        self.position = position
        #=====================================

        #Outvar
        #Note: not all the output variables are set in runtime as a member
        #some subclass calculates them on-the-fly directly during every get
        #(gamma is a typical example gamma = sigma - sigma.H)
        #================================
        self.sigma = None
        self.sigma_gr = None
        self.sigma_lr = None
        self.gamma = None
        #================================

        solver.Solver.__init__(self)

    def get_gamma(self):
        "Return \Gamma=j(\Sigma^{r} - \Sigma^{a})"""
        sigma = self.get('sigma')
        gamma = 1.j * (sigma - sigma.H)
        return gamma


class MRDephasing(Lead):
    """A Lead modelling Momentum relaxing dephasing"""

    def __init__(self):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()

        outvar:
        1) sigma
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) deph
        dephasing parameter (numpy.array, float)
        2) eqgreen
        equilibrium green's function for calculation of self energy
        3) green_gr
        Keldysh green's function for calculation of self energy

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self.deph = None
        #================================

        #Base constructors 
        position = 0
        self.size = deph.size
        Lead.__init__(self, position)

    def set_deph(self, deph):
        """Set a new dephasing parameter"""
        assert (type(deph) == np.ndarray)
        assert (deph.size == self.size)
        self.cleandep_deph()
        self.deph = deph
        return

    def cleandep_deph(self):
        self.sigma = None
        self.sigma_gr = None
        self.sigma_lr = None
        return

    def set_eqgreen(self, eqgreen):
        """Set an equilibrium Green's function"""
        self.eqgreen = eqgreen
        self.size = self.eqgreen.shape[0]
        self.cleandep_eqgreen()
        return

    def cleandep_eqgreen(self):
        self.sigma = None
        return

    def set_green_gr(self, green_gr):
        """Set a non-equilibrium Green's function (Greater)"""
        self.green_gr = green_gr
        self.size = self.green_gr.shape[0]
        self.cleandep_green_gr()
        return

    def cleandep_green_gr(self):
        self.sigma_gr = None
        return

    def set_green_lr(self, green_lr):
        """Set a non-equilibrium Green's function (Lesser)"""
        self.green_lr = green_lr
        self.size = self.green_lr.shape[0]
        self.cleandep_green_lr()
        return

    def cleandep_green_lr(self):
        self.sigma_lr = None
        return

    def _do_sigma(self):
        """Calculate the retarded self energy"""
        assert (not self.eqgreen is None)
        tmp = np.matrix(np.eye(self.size), dtype=np.complex128)
        #Note: * is an elementwise operator for ndarray types 
        np.fill_diagonal(tmp, np.multiply(self.eqgreen.diagonal(), self.deph))
        self.sigma = tmp

    def _do_sigma_gr(self):
        """Calculate the retarded self energy"""
        assert (not self.green_gr is None)
        tmp = np.matrix(np.eye(self.size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self.green_gr.diagonal(), self.deph))
        self.sigma_gr = tmp

    def _do_sigma_lr(self):
        """Calculate the retarded self energy"""
        assert (not self.green_lr is None)
        tmp = np.matrix(np.eye(self.size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self.green_lr.diagonal(), self.deph))
        self.sigma_lr = tmp


class MCDephasing(Lead):
    """A Lead modelling Momentum conserving dephasing"""

    def __init__(self):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()

        outvar:
        1) sigma
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) deph
        dephasing parameter (numpy.array, float)
        2) eqgreen
        equilibrium green's function for calculation of self energy
        3) green_gr
        Keldysh green's function for calculation of self energy

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self.deph = None
        self.green_gr = None
        self.eqgreen = None
        #================================

        #Base constructors
        position = 0
        self.size = deph.size
        Lead.__init__(self, position)

    def set_deph(self, deph):
        """Set a new dephasing parameter"""
        assert (type(deph) == np.ndarray)
        assert (deph.size == self.size)
        self.cleandep_deph()
        self._deph = deph
        return

    def cleandep_deph(self):
        self._sigma = None
        self._sigma_gr = None
        self._sigma_lr = None
        return

    def set_eqgreen(self, eqgreen):
        """Set an equilibrium Green's function"""
        self.eqgreen = eqgreen
        self.size = self.eqgreen.shape[0]
        self.cleandep_eqgreen()
        return

    def cleandep_eqgreen(self):
        self.sigma = None
        return

    def set_green_gr(self, green_gr):
        """Set a non-equilibrium Green's function (Greater)"""
        self.green_gr = green_gr
        self.size = self.green_gr.shape[0]
        self.cleandep_green_gr()
        return

    def cleandep_green_gr(self):
        self.sigma_gr = None
        return

    def set_green_lr(self, green_lr):
        """Set a non-equilibrium Green's function (Lesser)"""
        self.green_lr = green_lr
        self.size = self.green_lr.shape[0]
        self.cleandep_green_lr()
        return

    def cleandep_green_lr(self):
        self.sigma_lr = None
        return

    def _do_sigma(self):
        """Calculate the retarded self energy"""
        assert (not self.eqgreen is None)
        self._sigma = self.eqgreen * self._deph

    def _do_sigma_gr(self):
        """Calculate the greater self energy"""
        assert (not self.green_gr is None)
        self._sigma_gr = self.green_gr * self._deph

    def _do_sigma_lr(self):
        """Calculate the greater self energy"""
        assert (not self.green_lr is None)
        self._sigma_lr = self.green_gr.H * self._deph


class PhysicalLead(Lead):
    """A class derived from Lead for the description of physical contacts"""

    def __init__(self,
                 #Param
                 position, delta=defaults.delta):
        """A PhysicalLead object describe a semi-infinite periodic lead
        within the Open Boundary
        Steady State picture. 
        Derived class for Fermion and Bosons are implemented, this base class
        should not be explicitely instanced.
        position (int): index of interacting device layer
        mu (float): chemical potential
        We always mean by convention the
        coupling device-contact, i.e. Hdc"""

        #Param
        #================================

        #Invar
        #================================
        self.delta = delta
        #================================

        #Base constructors
        Lead.__init__(self, position)

    def set_mu(self, mu):
        """Set a chemical potential, for nonequilibrium self energy"""
        self.mu = mu
        self.cleandep_mu()
        return

    def set_temp(self, temp):
        """Set temperature, for non equilibrium self energy"""
        self.temp = temp
        self.cleandep_temp()
        return

    def cleandep_mu(self):
        self.sigma_gr = None
        self.sigma_lr = None
        return

    def cleandep_temp(self):
        self.sigma_gr = None
        self.sigma_lr = None
        return


class PhysicalLeadFermion(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""

    def __init__(self,
                 #Param
                 position, ham, ham_t, ham_ld, over=None,
                 over_t=None, over_ld=None, delta=defaults.delta):
        """

        parameters:

        1) ham (np.matrix): Hamiltonian for a single layer
        2) over (np.matrix), optional: Overlap for a single layer
        3) ham_t (np.matrix): coupling matrix between single layers
        4) over_t (np.matrix): overlap in the coupling block between single
        5) layers.If none, it's set to zero
        6) ham_ld (np.matrix): coupling between lead and device
        7) over_ld (np.matrix): overlap in device-lead coupling
        8) position (int): index of interacting device layer

        invar:
        1) energy (float)
        2) mu (float)
        chemical potential
        3) temp
        temperature

        outvar:
        1) sigma
        2) sigma_gr
        3) sigma_lr
        4) gamma

        We always mean by convention the
        coupling device-contact, i.e. Hcd 
        For the contact we specify coupling between first and second layer, i.e.
        H10 (same for the overlap, if any)"""

        #Param
        #==========================================================
        self.ham = ham
        self.ham_t = ham_t
        self.ham_ld = ham_ld
        self.over = over
        self.over_t = over_t
        self.over_ld = over_ld
        #Some check
        assert (type(ham) == np.matrixlib.defmatrix.matrix)
        if over:
            assert (type(over) == np.matrixlib.defmatrix.matrix)
        #H must be a square matrix
        assert (self.ham.shape[0] == self.ham.shape[1])

        #Set defaults
        self.pl_size = self.ham.shape[0]
        if over is None:
            self.over = np.matrix(np.eye(self.pl_size))
        if over_t is None:
            self.over_t = np.matrix(np.zeros(self.ham_t.shape))
        if over_ld is None:
            self.over_ld = np.matrix(np.zeros(self.ham_ld.shape))
        #===========================================================

        #Invar
        #============================
        self.energy = None
        #============================

        #Base constructor
        self.size = self.ham_ld.shape[0]
        PhysicalLead.__init__(self, position, delta=delta)

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.cleandep_energy()
        return

    def cleandep_energy(self):
        self.sigma = None
        self.sigma_lr = None
        self.sigma_gr = None
        return

    def _do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.
        This implementation follows Lopez-Sancho

        Note: modified from ASE implementation"""

        z = self.energy + self.delta * 1j
        #TODO: Verify this!!
        d_00 = z * self.over.H - self.ham.H
        d_11 = d_00.copy()
        d_10 = z * self.over_t - self.ham_t
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
        z = self.energy
        tau_ld = z * self.over_ld - self.ham_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = z * self.over_ld.H - self.ham_ld.H
        self.sigma = np.dot(tau_dl, a_ld)
        return self.sigma

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_lr = (dist.fermi(self.energy, self.mu, temppot=self.temp) *
                         1j * self.get_gamma())

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_gr = ((dist.fermi(self.energy, self.mu, temppot=self.temp)
                          - 1.0) * 1j * self.get_gamma())


# noinspection PyArgumentList
class PhysicalLeadPhonon(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""

    def __init__(self,
                 #Param
                 position, spring, spring_t, spring_ld, mass=None,
                 delta=defaults.delta):

        #Param
        #======================================
        self.spring = spring
        self.mass = mass
        self.spring_t = spring_t
        self.spring_ld = spring_ld
        #PL size n x n
        self.pl_size = self.spring.shape[0]
        #Interaction layer size tn x tm
        self.size = self.spring_t.shape[0]
        #Some checks
        assert (type(spring) == np.matrixlib.defmatrix.matrix)
        if mass:
            assert (type(mass) == np.matrixlib.defmatrix.matrix)
        #H must be a square matrix
        assert (self.spring.shape[0] == self.spring.shape[1])
        pl_size = self.spring.shape[0]
        if not mass:
            self.mass = np.matrix(np.eye(pl_size))
        #======================================

        #Invar
        #===================================
        self.frequency = None
        #===================================

        #Base constructor
        self.size = self.spring_ld.shape[0]
        PhysicalLead.__init__(self, position,
                              delta=delta)

    def set_frequency(self, frequency):
        """Set frequency point"""
        if frequency != self.frequency:
            self.frequency = frequency
            self.cleandep_frequency()
        return

    def cleandep_frequency(self):
        self.sigma = None
        self.sigma_lr = None
        self.sigma_gr = None
        return

    def _do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.

        Note: from ASE implementation
        slightly adapted for phonons
        
        Note: frequencies are given in fs^-1, energies in eV"""

        z = self.frequency * self.frequency + self.delta * 1j
        #TODO: Verify this!!
        d_00 = z * self.mass - self.spring
        d_11 = d_00.copy()
        d_10 = - self.spring_t
        d_01 = - self.spring_t.H
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
        tau_ld = self.spring_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = self.spring_ld.H
        self.sigma = np.dot(tau_dl, a_ld)
        return

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        energy = self.frequency * consts.hbar_eV_fs
        self.sigma_lr = ((dist.bose(energy, self.mu, temppot=self.temp)) *
                         (-1j) * self.get_gamma())
        return

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        energy = self.frequency * consts.hbar_eV_fs
        self.sigma_gr = ((dist.bose(energy, self.mu, temppot=self.temp)
                          + 1.0) * (-1j) * self.get_gamma())
        return


class WideBandFermion(PhysicalLead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""

    def __init__(self,
                 #Param
                 position, dos, ham_ld, over_ld=None,
                 delta=defaults.delta):
        """
        The following quantities must be specified:

        dos (float): Density of states of the lead (ev^-1atom^-1)
        over_ld (np.matrix): overlap in device-lead coupling
        position (int): index of interacting device layer
        mu (float): chemical potential

        We always mean by convention the
        coupling device-contact, i.e. Hcd 
        For the contact we specify coupling between first and second layer, i.e.
        H10 (same for the overlap, if any)"""

        #Param
        #==========================================================
        self.dos = dos
        self.ham_ld = ham_ld
        self.over_ld = over_ld

        #Set defaults
        self.pl_size = self.ham_ld.shape[0]
        if not over_ld:
            self.over_ld = np.matrix(np.zeros(self.ham_ld.shape))
        #===========================================================

        #Invar
        #============================
        self.energy = None
        #============================

        #Base constructor
        size = self.ham_ld.shape[0]
        PhysicalLead.__init__(self, position, size, delta=delta)

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self._energy:
            self._energy = energy
            self.cleandep_energy()
        return

    def cleandep_energy(self):
        self.sigma = None
        self.sigma_lr = None
        self.sigma_gr = None
        return

    def _do_sigma(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        z = self.energy
        tau_ld = z * self.over_ld - self.ham_ld
        a_ld = 1j * 2.0 * np.pi * np.dot(tau_ld, self.dos)
        tau_dl = z * self.over_ld.H - self.ham_ld.H
        self.sigma = np.dot(tau_dl, a_ld)
        return

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_lr = (dist.fermi(self.energy, self.mu, temppot=self.temp) *
                         1j * self.get_gamma())
        return

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_gr = ((dist.fermi(self.energy, self.mu, temppot=self.temp)
                          - 1.0) * 1j * self.get_gamma())
        return
