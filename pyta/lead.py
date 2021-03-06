"""
Contains classes to calculate Self Energies
"""
import numpy as np
from pyta import solver
import pyta.defaults as defaults
import pyta.dist as dist
import pyta.consts as consts


class Lead(solver.Solver):
    """
    Abstract Class.
    A base class for managing and building up real and virtual leads.
    """

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
        self._position = position
        #=====================================

        #Outvar
        #Note: not all the output variables are set in runtime as a member
        #some subclass calculates them on-the-fly directly during every get
        #(gamma is a typical example gamma = sigma_ret - sigma_ret.H)
        #================================
        self._sigma_ret = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._gamma = None
        #================================

        super(Lead, self).__init__()

    @property
    def position(self):
        """
        Return:
        position (int): first index where the system interact with the self
                energy or, equivalently, the index where self energies are
                plugged
        """
        return self._position

    @property
    def gamma(self):
        """
        Return \Gamma=j(\Sigma^{r} - \Sigma^{a})
        """
        sigma = self.sigma_ret
        gamma = 1.j * (sigma - sigma.conj().T)
        return gamma

    @property
    def occupation(self):
        """
        Calculate the occupation by comparing the lesser green function
        and the spectral function.
        """
        diag1 = np.diag(self.sigma_lr)
        diag2 = np.diag(self.gamma)
        occupation = np.imag(np.divide(diag1, diag2))
        return occupation

    @property
    def sigma_ret(self):
        """
        Output variable getter. See class doc.
        """
        if self._sigma_ret is None:
            self._do_sigma_ret()
        return self._sigma_ret

    @property
    def sigma_gr(self):
        """
        Output variable getter. See class doc.
        """
        if self._sigma_gr is None:
            self._do_sigma_gr()
        return self._sigma_gr

    @property
    def sigma_lr(self):
        """
        Output variable getter. See class doc.
        """
        if self._sigma_lr is None:
            self._do_sigma_lr()
        return self._sigma_lr

    def reset(self):
        """
        Set all the output variables to undefined state
        """
        self._sigma_ret = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._gamma = None

    # Virtual classes
    def _do_sigma_ret(self):
        """
        Virtual class method
        """
        raise NotImplementedError()

    def _do_sigma_gr(self):
        """
        Virtual class method
        """
        raise NotImplementedError()

    def _do_sigma_lr(self):
        """
        Virtual class method
        """
        raise NotImplementedError()


class MRDephasing(Lead):
    """A virtual Lead modelling Momentum relaxing dephasing"""

    def __init__(self,
                 #Invar
                 green_ret=None,
                 green_lr=None,
                 coupling=None):
        """
        Only the name and dephasing intensity needed at first.
        You have to provide the equilibrium and the Keldysh Green's function.

        Note: if the coupling is

        outvar:
        1) sigma_ret
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) deph
        dephasing parameter (numpy.array)
        2) green_ret
        Reference retarded Green's function
        3) green_lr
        Reference lesser Green's function
        4) over

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self._coupling = coupling
        self._green_ret = green_ret
        self._green_lr = green_lr
        #================================

        #Internal
        if self._coupling is not None:
            self._size = len(self.coupling)

        #Base constructors
        position = 0
        super(MRDephasing, self).__init__(position)

    @property
    def coupling(self):
        """
        Returns:
            coupling ()
        """
        return self._coupling

    @coupling.setter
    def coupling(self, value):
        """Set a new dephasing parameter"""
        assert type(value) == np.ndarray
        self._sigma_ret = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._coupling = value
        self._size = len(value)
        return

    @property
    def green_ret(self):
        """
        Returns:
            green_ret (numpy.ndarray): green retarded used to build the self
                energy
        """
        return self._green_ret

    @green_ret.setter
    def green_ret(self, value):
        """
        Args:
            green_ret (numpy.ndarray): set the retarded green function used to
                build the self energy
        """
        self._sigma_ret = None
        self._green_ret = value

    @property
    def green_lr(self):
        """
        Returns:
            green_ret (numpy.ndarray): green retarded used to build the self
                energy
        """
        return self._green_lr

    @green_lr.setter
    def green_lr(self, value):
        """
        Args:
            green_lr (numpy.ndarray): set the lesser green function used to
                build the self energy
        """
        self._sigma_lr = None
        self._green_lr = value

    def _do_sigma_ret(self):
        """
        Calculate the retarded self energy
        """
        tmp = np.asmatrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self.green_ret.diagonal(),
                                          self._coupling))
        self._sigma_ret = tmp
        return

    def _do_sigma_gr(self):
        """
        Calculate the greater self energy
        """
        gamma = self.gamma
        sigma_lr = self.sigma_lr
        self._sigma_gr = sigma_lr - 1j * gamma
        return

    def _do_sigma_lr(self):
        """
        Calculate the lesser self energy
        """
        tmp = np.asmatrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self.green_lr.diagonal(),
                                          self.coupling))
        self._sigma_lr = tmp
        return


class MCDephasing(Lead):
    """A virtual Lead modelling Momentum conserving dephasing"""

    def __init__(self,
                 #Invar
                 green_ret=None,
                 green_lr=None,
                 coupling=None):
        """
        Only the name and dephasing intensity needed at first.
        You have to provide the equilibrium and the Keldysh Green's function.

        Note: if the coupling is

        outvar:
        1) sigma_ret
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) deph
        dephasing parameter (double)
        2) green_ret
        Reference retarded Green's function
        3) green_lr
        Reference lesser Green's function
        4) over

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self._coupling = coupling
        self._green_ret = green_ret
        self._green_lr = green_lr
        #================================

        #Base constructors
        position = 0
        super(MCDephasing, self).__init__(position)

    @property
    def coupling(self):
        """
        Returns:
            coupling ()
        """
        return self._coupling

    @property
    def size(self):
        """
        Returns:
            size: system size
        """
        return self._green_ret.shape[0]

    @coupling.setter
    def coupling(self, value):
        """Set a new dephasing parameter"""
        self._sigma_ret = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._coupling = value
        return

    @property
    def green_ret(self):
        """
        Returns:
            green_ret (numpy.ndarray): green retarded used to build the self
                energy
        """
        return self._green_ret

    @green_ret.setter
    def green_ret(self, value):
        """
        Args:
            green_ret (numpy.ndarray): set the retarded green function used to
                build the self energy
        """
        self._sigma_ret = None
        self._green_ret = value

    @property
    def green_lr(self):
        """
        Returns:
            green_ret (numpy.ndarray): green retarded used to build the self
                energy
        """
        return self._green_lr

    @green_lr.setter
    def green_lr(self, value):
        """
        Args:
            green_lr (numpy.ndarray): set the lesser green function used to
                build the self energy
        """
        self._sigma_lr = None
        self._green_lr = value

    def _do_sigma_ret(self):
        """
        Calculate the retarded self energy
        """
        self._sigma_ret = self.coupling * self.green_ret
        return

    def _do_sigma_gr(self):
        """
        Calculate the greater self energy
        """
        gamma = self.gamma
        sigma_lr = self.sigma_lr
        self._sigma_gr = sigma_lr - 1j * gamma
        return

    def _do_sigma_lr(self):
        """
        Calculate the lesser self energy
        """
        self._sigma_lr = self.coupling * self.green_lr
        return


class MRDephasingClose(Lead):
    """
    A virtual Lead modelling Momentum relaxing dephasing. It assume a
    constant coupling and evaluate the lesser through a close form equation.
    See Cresti, J.Phys.Cond.Matter. 18 (2006) 10059 for more details.
    """

    def __init__(self,
                 #Invar
                 green_ret=None,
                 sigma_lr_leads=None,
                 coupling=None):
        """
        Only the name and dephasing intensity needed at first.
        You have to provide the equilibrium and the Keldysh Green's function.

        Note: if the coupling is

        outvar:
        1) sigma_ret
        2) sigma_gr
        3) sigma_lr
        4) gamma

        invar:
        1) coupling
        dephasing parameter (float)
        2) green_ret
        Reference retarded Green's function
        3) sigma_lr_leads
        Lead only self energy of the system

        internal:
        2) size
        index of system to be attached to

        """

        #Param
        #================================
        #================================

        #Invar
        #================================
        self._coupling = coupling
        self._green_ret = green_ret
        self._sigma_lr_leads = sigma_lr_leads
        #================================

        #Internal
        if self._green_ret is not None:
            self._size = self._green_ret.shape[0]

        #Base constructors
        position = 0
        super(MRDephasingClose, self).__init__(position)

    @property
    def coupling(self):
        """
        Input variable getter. See class doc
        """
        return self._coupling

    @coupling.setter
    def coupling(self, value):
        """
        Input variable getter. See class doc.
        """
        self._sigma_ret = None
        self._sigma_gr = None
        self._sigma_lr = None
        self._coupling = value
        return

    @property
    def green_ret(self):
        """
        Input variable getter. See class doc.
        """
        return self._green_ret

    @green_ret.setter
    def green_ret(self, value):
        """
        Input variable setter. See class doc.
        """
        self._sigma_ret = None
        self._sigma_lr = None
        self._sigma_gr = None
        self._size = value.shape[0]
        self._green_ret = value

    @property
    def sigma_lr_leads(self):
        """
        Input variable setter. See class doc.
        """
        return self._sigma_lr_leads

    @sigma_lr_leads.setter
    def sigma_lr_leads(self, value):
        """
        Input variable setter. See class doc.
        """
        self._sigma_lr = None
        self._sigma_gr = None
        self._sigma_lr_leads = value

    def _do_sigma_ret(self):
        """
        Calculate the retarded self energy
        """
        tmp = np.asmatrix(np.eye(self._size), dtype=np.complex128)
        np.fill_diagonal(tmp, np.multiply(self.green_ret.diagonal(),
                                          self._coupling))
        self._sigma_ret = tmp
        return

    def _do_sigma_gr(self):
        """
        Calculate the greater self energy
        """
        gamma = self.gamma
        sigma_lr = self.sigma_lr
        self._sigma_gr = sigma_lr - 1j * gamma
        return

    def _do_sigma_lr(self):
        """
        Calculate the lesser self energy
        """
        q_mat = np.multiply(self._green_ret, self.green_ret.conj().T)
        tmp1 = self._coupling * np.linalg.inv(
            np.eye(self._size) - self._coupling * q_mat)
        tmp2 = np.dot(np.dot(self._green_ret, self._sigma_lr_leads),
                      self._green_ret.conj().T)
        self._sigma_lr = np.zeros(shape=(self._size, self._size),
                                  dtype=np.complex128)
        np.fill_diagonal(self._sigma_lr, np.dot(tmp1, np.diag(tmp2)))
        return



class ElLead(Lead):
    """
    A class derived from Lead for the description of electron bath, in
    the case of Fermion Green's functions.
    """

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
        if ((ham - ham.conj().T) > 1e-10).any():
            raise ValueError(
                'Error in Lead parameter. The Hamiltonian is not hermitian')
        self.ham_t = ham_t
        self.ham_ld = ham_ld
        self.over = over
        self.over_t = over_t
        self.over_ld = over_ld

        #Set defaults
        self.pl_size = self.ham.shape[0]
        if over is None:
            self.over = np.eye(self.pl_size)
        if over_t is None:
            self.over_t = np.zeros(self.pl_size)
        if over_ld is None:
            self.over_ld = np.zeros(self.pl_size)
        #===========================================================

        #Invar
        #============================
        self._energy = None
        self._mu = 0.0
        self._delta = delta
        self._temperature = temperature
        #============================

        #Base constructor
        self.size = self.ham_ld.shape[0]
        super(ElLead, self).__init__(position)

    @property
    def temperature(self):
        """
        Input variable getter. See class doc.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """
        Input variable setter. See class doc.
        """
        self._temperature = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def energy(self):
        """
        Input variable getter. See class doc.
        """
        return self._energy

    @energy.setter
    def energy(self, value):
        """
        Input variable setter. See class doc.
        """
        if value != self._energy:
            self._energy = value
            self._sigma_ret = None
            self._sigma_lr = None
            self._sigma_gr = None
        return

    @property
    def mu(self):
        """
        Input variable getter. See class doc.
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        """
        Input variable setter. See class doc.
        """
        self._mu = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def distribution(self):
        """
        Return the value of Fermi distribution
        """
        return dist.fermi(self._energy, self.mu, temppot=self._temperature)

    def _do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """
        Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.
        This implementation follows Lopez-Sancho

        Credit: modified from ASE implementation
        """

        z = self._energy + self._delta * 1j
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
        """
        Calculate the equilibrium retarded self energy \Sigma^{r}.
        """
        z = self._energy
        tau_ld = z * self.over_ld - self.ham_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = z * self.over_ld.conj().T - self.ham_ld.conj().T
        self._sigma_ret = np.dot(tau_dl, a_ld)

    def _do_sigma_lr(self):
        """
        Calculate the Sigma lesser
        """
        assert not self.mu is None
        self._sigma_lr = (
            self.distribution * 1j * self.gamma)

    def _do_sigma_gr(self):
        """
        Calculate the Sigma lesser
        """
        assert not self.mu is None
        self._sigma_gr = (
            (self.distribution - 1.0) * 1j * self.gamma)



# noinspection PyArgumentList
class PhLead(Lead):
    """
    A class derived from Lead for the description of phonon lead
    """

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
        #assert (type(spring) == np.matrixlib.defmatrix.matrix)
        #if mass:
        #    assert (type(mass) == np.matrixlib.defmatrix.matrix)
        #H must be a square matrix
        assert self.spring.shape[0] == self.spring.shape[1]
        pl_size = self.spring.shape[0]
        if not mass:
            self.mass = np.eye(pl_size)
        #======================================

        #Invar
        #===================================
        self._frequency = None
        self.delta = delta
        self._temperature = temperature
        #===================================

        #Base constructor
        self.size = self.spring_ld.shape[0]
        super(PhLead, self).__init__(position)

    @property
    def temperature(self):
        """
        Input variable getter. See class doc.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """
        Input variable setter. See class doc.
        """
        self._temperature = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def frequency(self):
        """
        Input variable getter. See class doc.
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """
        Input variable setter. See class doc.
        """
        if value != self._frequency:
            self._frequency = value
            self._sigma_ret = None
            self._sigma_lr = None
            self._sigma_gr = None
        return

    def _do_invsurfgreen(self, tol=defaults.surfgreen_tol):
        """
        Calculate the INVERSE of surface green's function
        by means of decimation
        algorithm Guinea F, Tejedor C, Flores F and Louis E 1983 Effective
        two-dimensional Hamiltonian at surfacesPhys. Rev.B 28 4397.

        Credit: from ASE implementation slightly adapted for phonons

        Note: frequencies are given in fs^-1, energies in eV
        """

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
        """
        Calculate the equilibrium retarded self energy \Sigma^{r}.
        """
        tau_ld = self.spring_ld
        a_ld = np.linalg.solve(self._do_invsurfgreen(), tau_ld)
        tau_dl = self.spring_ld.conj().T
        self._sigma_ret = np.dot(tau_dl, a_ld)
        return

    def _do_sigma_lr(self):
        """
        Calculate the Sigma lesser
        """
        energy = self._frequency * consts.hbar_eV_fs
        self._sigma_lr = ((dist.bose(energy, temppot=self._temperature)) *
                          (-1j) * self.gamma)
        return

    def _do_sigma_gr(self):
        """
        Calculate the Sigma lesser
        """
        energy = self._frequency * consts.hbar_eV_fs
        self._sigma_gr = ((dist.bose(energy, temppot=self._temperature) +
                           1.0) * (-1j) * self.gamma)
        return


class ElWideBand(Lead):
    """
    A class derived from Lead for the description of physical contacts, in
    the case of Fermion Green's functions
    """

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
        self._energy = None
        self._mu = mu
        self.delta = delta
        self._temperature = temperature
        #============================

        #Base constructor
        self.size = self.ham_ld.shape[0]
        super(ElWideBand, self).__init__(position)

    @property
    def temperature(self):
        """
        Input variable getter. See class doc.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """
        Input variable setter. See class doc.
        """
        self._temperature = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def energy(self):
        """
        Input variable getter. See class doc.
        """
        return self._energy

    @energy.setter
    def energy(self, value):
        """
        Input variable setter. See class doc.
        """
        if value != self._energy:
            self._energy = value
            self._sigma_ret = None
            self._sigma_lr = None
            self._sigma_gr = None
        return

    @property
    def mu(self):
        """
        Input variable getter. See class doc.
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        """
        Input variable setter. See class doc.
        """
        self._mu = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def distribution(self):
        """
        Return the value of Fermi distribution
        """
        return dist.fermi(self._energy, self.mu, temppot=self._temperature)

    def _do_sigma_ret(self):
        """
        Calculate the equilibrium retarded self energy \Sigma^{r}.
        """
        z = self.energy
        dos_mat = np.zeros((self.pl_size, self.pl_size), dtype=np.complex128)
        np.fill_diagonal(dos_mat, self.dos)
        tau_ld = z * self.over_ld - self.ham_ld
        a_ld = -1j * np.pi * np.dot(tau_ld, self.dos)
        self._sigma_ret = np.dot(a_ld, tau_ld.H)
        return

    def _do_sigma_lr(self):
        """
        Calculate the Sigma lesser
        """
        assert not self._mu is None
        self._sigma_lr = (
            self.distribution * 1j * self.gamma)
        return

    def _do_sigma_gr(self):
        """
        Calculate the Sigma lesser
        """
        assert not self._mu is None
        self._sigma_gr = (
            (self.distribution - 1.0) * 1j * self.gamma)
        return


class ElWideBandGamma(Lead):
    """
    A completely phenomenological virtual lead with coupling given by a
    single parameter Gamma and purely imaginary self-energies
    """

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
        self._coupling = coupling

        #Set defaults
        self.size = size
        #===========================================================

        #Invar
        #============================
        self._energy = None
        self._mu = mu
        self._temperature = temperature
        #============================

        #Base constructor
        super(ElWideBandGamma, self).__init__(position)

    @property
    def temperature(self):
        """
        Input variable getter. See class doc.
        """
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        """
        Input variable setter. See class doc.
        """
        self._temperature = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def energy(self):
        """
        Input variable getter. See class doc.
        """
        return self._energy

    @energy.setter
    def energy(self, value):
        """
        Input variable setter. See class doc.
        """
        if value != self._energy:
            self._energy = value
            self._sigma_ret = None
            self._sigma_lr = None
            self._sigma_gr = None
        return

    @property
    def mu(self):
        """
        Input variable getter. See class doc.
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        """
        Input variable setter. See class doc.
        """
        self._mu = value
        self._sigma_gr = None
        self._sigma_lr = None
        return

    @property
    def distribution(self):
        """
        Return the value of Fermi distribution
        """
        return dist.fermi(self._energy, self.mu, temppot=self._temperature)

    def _do_sigma_ret(self):
        """
        Calculate the equilibrium retarded self energy \Sigma^{r}.
        """
        gamma = np.zeros((self.size, self.size), dtype=np.complex128)
        np.fill_diagonal(gamma, self._coupling)
        self._sigma_ret = -1j * gamma / 2.
        return

    def _do_sigma_lr(self):
        """
        Calculate the Sigma lesser
        """
        assert not self.mu is None
        self._sigma_lr = (
            self.distribution * 1j * self.gamma)
        return

    def _do_sigma_gr(self):
        """
        Calculate the Sigma lesser
        """
        assert not self.mu is None
        self._sigma_gr = (
            (self.distribution - 1.0) * 1j * self.gamma)
        return
