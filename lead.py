import numpy as np
from pyta import solver
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
    """ Abstract Class.
        A base class for managing and building up real and virtual leads. """

    def __init__(self,
                 #Param
                 position):
        """
        Base class (virtual like)

        :param position:
        outvar:
        1) sigma_ret
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

        #Invar

        #Param
        #=====================================
        self.position = position
        #=====================================

        #Outvar
        #Note: not all the output variables are set in runtime as a member
        #some subclass calculates them on-the-fly directly during every get
        #(gamma is a typical example gamma = sigma_ret - sigma_ret.H)
        #================================
        self.sigma_ret = None
        self.sigma_gr = None
        self.sigma_lr = None
        self.gamma = None
        #================================

        super(Lead, self).__init__()

    def get_gamma(self):
        "Return \Gamma=j(\Sigma^{r} - \Sigma^{a})"""
        sigma = self.get('sigma_ret')
        gamma = 1.j * (sigma - sigma.H)
        return gamma

    def get_occupation(self):
        """Calculate the occupation by comparing the lesser green function
        and the spectral function. """
        diag1 = self.get('sigma_lr')
        diag2 = self.get('gamma')
        occupation = (np.imag(diag1 / diag2))
        return occupation

    def reset(self):
        """
        Set all the output variables to undefined state
        """


class MRDephasing(Lead):
    """A virtual Lead modelling Momentum relaxing dephasing"""

    def __init__(self,
                 #Invar
                 green_ret=None,
                 green_lr=None,
                 coupling=None,
                 rotation=None):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()

        outvar:
        1) sigma_ret
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) deph
        dephasing parameter (numpy.array, float)
        2) green_ret
        Reference retarded Green's function
        3) green_lr
        Reference lesser Green's function
        4) over
        We may need to apply a rotation Sigma*rotation for consistency with
        rotations

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self.coupling = coupling
        self.green_ret = green_ret
        self.green_lr = green_lr
        self.rotation = rotation
        #================================

        #Internal
        self._has_rotation = False
        if self.coupling is not None:
            self.size = len(self.coupling)
            if self.rotation is None:
                self.rotation = np.asmatrix(np.eye(self.size))
                self._has_rotation = True

        #Base constructors
        position = 0
        super(MRDephasing, self).__init__(position)

    def set_coupling(self, coupling):
        """Set a new dephasing parameter"""
        assert (type(coupling) == np.ndarray)
        self.sigma_ret = None
        self.sigma_gr = None
        self.sigma_lr = None
        self.coupling = coupling
        self.size = len(coupling)
        self.rotation = np.asmatrix(np.eye(self.size))
        return

    def set_rotation(self, rotation):
        """
         Set a transformation matrix (for non-orthogonal cases)
        """
        self.rotation = rotation
        self._has_rotation = True

    def set_green_ret(self, green_ret):
        self.sigma_ret = None
        self.green_ret = green_ret

    def set_green_lr(self, green_lr):
        self.sigma_lr = None
        self.green_lr = green_lr

    def _do_sigma_ret(self):
        """Calculate the retarded self energy"""
        if self._has_rotation:
            ## This implementation was equivalent to the one above using
            ## rotation = overlap^1/2. I could not find out under which assumption,
            ## it doesn't seem to be mathematically equivalent. However this last one
            ## can be easily demonstrated to be correct
            tmp = (self.rotation * self.green_ret * self.rotation).diagonal()
            tmp2 = np.multiply(tmp, self.coupling)
            tmp = np.asmatrix(np.eye(self.size), dtype=np.complex128)
            np.fill_diagonal(tmp, tmp2)
            self.sigma_ret = self.rotation * tmp * self.rotation

            ## Approximate transformation (first order expansion of the above)
            #tmp = np.asmatrix(np.eye(self.size), dtype=np.complex128)
            #np.fill_diagonal(tmp, np.multiply((self.green_ret * self.rotation).diagonal(),
            #                                  self.coupling))
            #self.sigma_ret = 0.5 * tmp * self.rotation + 0.5 * self.rotation * tmp

        else:
            tmp = np.asmatrix(np.eye(self.size), dtype=np.complex128)
            np.fill_diagonal(tmp, np.multiply(self.green_ret.diagonal(),
                                              self.coupling))
            self.sigma_ret = tmp
        return

    def _do_sigma_gr(self):
        """Calculate the retarded self energy"""
        gamma = self.get('gamma')
        sigma_lr = self.get('sigma_lr')
        self.sigma_gr = sigma_lr - 1j * gamma
        return

    def _do_sigma_lr(self):
        """Calculate the retarded self energy"""
        tmp = np.asmatrix(np.eye(self.size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self.green_lr.diagonal(),
                                          self.coupling))
        self.sigma_lr = tmp
        return


class MCDephasing(Lead):
    """A virtual Lead modelling Momentum conserving dephasing"""

    def __init__(self):
        """Only the name and dephasing intensity needed at first.
        You can provide the equilibrium or the Keldysh Green's function
        to get the self energies using set_eqgreen() and set_neqgreen()

        outvar:
        1) sigma_ret
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) coupling
        coupling strength (numpy.array, float)
        2) greensolver
        A Green solver which provides all the needed Green's function

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self.coupling = None
        self.green_gr = None
        self.green_ret = None
        #================================

        #Base constructors
        position = 0
        self.size = 0
        super(MCDephasing, self).__init__(position)

    def set_coupling(self, coupling):
        """Set a new dephasing parameter"""
        assert (type(coupling) == np.ndarray)
        assert (coupling.size == self.size)
        self._sigma = None
        self._sigma_gr = None
        self._sigma_lr = None
        self.coupling = coupling
        return

    def set_greensolver(self, green):
        """Set a non-equilibrium Green's function (Greater)"""
        self.greensolver = green
        self.sigma_gr = None
        self.sigma_lr = None
        self.sigma_ret = None
        self.size = green.size
        return

    def _do_sigma_ret(self):
        """Calculate the retarded self energy"""
        green_ret = self.greensolver.get('green_ret')
        self.sigma_ret = np.dot(green_ret, self.coupling)

    def _do_sigma_gr(self):
        """Calculate the greater self energy"""
        green_gr = self.greensolver.get('green_gr')
        self.sigma_gr = np.dot(green_gr, self.coupling)

    def _do_sigma_lr(self):
        """Calculate the greater self energy"""
        green_lr = self.greensolver.get('green_lr')
        self.sigma_lr = np.dot(green_lr, self.coupling)


class ElLead(Lead):
    """A class derived from Lead for the description of electron bath, in
    the case of Fermion Green's functions"""

    def __init__(self,
                 #Param
                 position, ham, ham_t, ham_ld, over=None,
                 over_t=None, over_ld=None, temperature=0.0,
                 delta=defaults.delta):
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
        3) temperature
        temperature

        outvar:
        1) sigma_ret
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
        if ((ham - ham.H) > 1e-10).any():
            raise ValueError(
                'Error in Lead parameter. The Hamiltonian is not hermitian')
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
            self.over = np.asmatrix(np.eye(self.pl_size))
        if over_t is None:
            self.over_t = np.asmatrix(np.zeros(self.ham_t.shape))
        if over_ld is None:
            self.over_ld = np.asmatrix(np.zeros(self.ham_ld.shape))
        #===========================================================

        #Invar
        #============================
        self.energy = None
        self.mu = 0.0
        self.delta = delta
        self.temperature = temperature
        #============================

        #Base constructor
        self.size = self.ham_ld.shape[0]
        super(ElLead, self).__init__(position)

    def set_temperature(self, temperature):
        """Set temperature, for non equilibrium self energy"""
        self.temperature = temperature
        self.sigma_gr = None
        self.sigma_lr = None
        return

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.sigma_ret = None
            self.sigma_lr = None
            self.sigma_gr = None
        return

    def set_mu(self, mu):
        """Set a chemical potential, for nonequilibrium self energy"""
        self.mu = mu
        self.sigma_gr = None
        self.sigma_lr = None
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
        d_00 = z * self.over.conj().T - self.ham.conj().T
        d_11 = d_00.copy()
        d_10 = z * self.over_t - self.ham_t
        d_01 = z * self.over_t.conj().T - self.ham_t.conj().T
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

    def _do_sigma_ret(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        z = self.energy
        tau_ld = z * self.over_ld - self.ham_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = z * self.over_ld.H - self.ham_ld.H
        self.sigma_ret = np.dot(tau_dl, a_ld)
        return self.sigma_ret

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_lr = (
            dist.fermi(self.energy, self.mu, temppot=self.temperature) *
            1j * self.get_gamma())

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_gr = (
            (dist.fermi(self.energy, self.mu, temppot=self.temperature)
             - 1.0) * 1j * self.get_gamma())


# noinspection PyArgumentList
class PhLead(Lead):
    """A class derived from Lead for the description of phonon lead"""

    def __init__(self,
                 #Param
                 position, spring, spring_t, spring_ld, mass=None,
                 temperature=0.0, delta=defaults.delta):

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
            self.mass = np.asmatrix(np.eye(pl_size))
        #======================================

        #Invar
        #===================================
        self.frequency = None
        self.delta = delta
        self.temperature = temperature
        #===================================

        #Base constructor
        self.size = self.spring_ld.shape[0]
        super(PhLead, self).__init__(position)

    def set_temperature(self, temperature):
        """Set temperature, for non equilibrium self energy"""
        self.temperature = temperature
        self.sigma_gr = None
        self.sigma_lr = None
        return

    def set_frequency(self, frequency):
        """Set frequency point"""
        if frequency != self.frequency:
            self.frequency = frequency
            self.sigma_ret = None
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
        d_01 = - self.spring_t.conj().T
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

    def _do_sigma_ret(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        tau_ld = self.spring_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = self.spring_ld.conj().T
        self.sigma_ret = np.dot(tau_dl, a_ld)
        return

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        energy = self.frequency * consts.hbar_eV_fs
        self.sigma_lr = ((dist.bose(energy, temppot=self.temperature)) *
                         (-1j) * self.get_gamma())
        return

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        energy = self.frequency * consts.hbar_eV_fs
        self.sigma_gr = ((dist.bose(energy, self.mu, temppot=self.temperature)
                          + 1.0) * (-1j) * self.get_gamma())
        return


class ElWideBand(Lead):
    """A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions"""

    def __init__(self,
                 #Param
                 position, dos, ham_ld, over_ld=None,
                 delta=defaults.delta,
                 #Invar
                 temperature=0.0, mu=None):
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
        if over_ld is None:
            self.over_ld = np.zeros(self.ham_ld.shape)
        #===========================================================

        #Invar
        #============================
        self.energy = None
        self.mu = mu
        self.delta = delta
        self.temperature = temperature
        #============================

        #Base constructor
        self.size = self.ham_ld.shape[0]
        super(ElWideBand, self).__init__(position)

    def set_temperature(self, temperature):
        """Set temperature, for non equilibrium self energy"""
        self.temperature = temperature
        self.sigma_gr = None
        self.sigma_lr = None
        return

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.sigma_ret = None
            self.sigma_lr = None
            self.sigma_gr = None
        return

    def _do_sigma_ret(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        z = self.energy
        dos_mat = np.zeros((self.pl_size, self.pl_size), dtype=np.complex128)
        np.fill_diagonal(dos_mat, self.dos)
        tau_ld = z * self.over_ld - self.ham_ld
        a_ld = -1j * np.pi * np.dot(tau_ld, self.dos)
        self.sigma_ret = np.dot(a_ld, tau_ld.H)
        return

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_lr = (
            dist.fermi(self.energy, self.mu, temppot=self.temperature) *
            1j * self.get_gamma())
        return

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_gr = (
            (dist.fermi(self.energy, self.mu, temppot=self.temperature)
             - 1.0) * 1j * self.get_gamma())
        return


class ElWideBandGamma(Lead):
    """A completely phenomenological virtual lead with coupling given by a
    single parameter Gamma and purely imaginary self-energies"""

    def __init__(self,
                 #Param
                 position, size, coupling,
                 #Invar
                 temperature=0.0, mu=None):
        """
        The following quantities must be specified:

        position (int): index of interacting device layer (lower index)
        size (int): size of matrix. The model is diagonal
        mu (float): chemical potential

        We always mean by convention the
        coupling device-contact, i.e. Hcd
        For the contact we specify coupling between first and second layer, i.e.
        H10 (same for the overlap, if any)"""

        #Param
        #==========================================================
        self.coupling = coupling

        #Set defaults
        self.pl_size = size
        #===========================================================

        #Invar
        #============================
        self.energy = None
        self.mu = mu
        self.temperature = temperature
        #============================

        #Base constructor
        self.size = size
        super(ElWideBandGamma, self).__init__(position)

    def set_temperature(self, temperature):
        """Set temperature, for non equilibrium self energy"""
        self.temperature = temperature
        self.sigma_gr = None
        self.sigma_lr = None
        return

    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.sigma_ret = None
            self.sigma_lr = None
            self.sigma_gr = None
        return

    def _do_sigma_ret(self):
        """Calculate the equilibrium retarded self energy \Sigma^{r}."""
        gamma_mat = np.matrix(np.zeros((self.pl_size, self.pl_size),
                                       dtype=np.complex128))
        np.fill_diagonal(gamma_mat, self.coupling)
        self.sigma_ret = 1j * gamma_mat / 2.
        return

    def _do_sigma_lr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_lr = (
            dist.fermi(self.energy, self.mu, temppot=self.temperature) *
            1j * self.get_gamma())
        return

    def _do_sigma_gr(self):
        """Calculate the Sigma lesser"""
        assert (not self.mu is None)
        self.sigma_gr = (
            (dist.fermi(self.energy, self.mu, temppot=self.temperature)
             - 1.0) * 1j * self.get_gamma())
        return
