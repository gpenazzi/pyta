"""
Contains solver classes solving the equilibrium and non equilibrium Green's
function steady state transport, in the commo "NEGF" modeling scheme.
"""
import numpy as np
from pyta import mathutils
from pyta import solver
#import pyta.consts


class Green(solver.Solver):
    """ Abstract Base Class for energy resolved Green's function

        invar:
        1) leads
            list of Leads
            overwrite, append

        parameters:
        1) size

        outvar:
        1) green_ret
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
        self._size = size

        #Outvar
        self._green_ret = None
        self._green_gr = None
        self._green_lr = None

        super(Green, self).__init__()

    def reset(self):
        """
        Set all output variables to undefined state
        """
        self._green_ret = None
        self._green_lr = None
        self._green_gr = None

    @property
    def green_ret(self):
        """
        Ouput variable getter. See class doc.
        """
        if self._green_ret is None:
            self._do_green_ret()
        return self._green_ret

    @property
    def green_lr(self):
        """
        Output variable getter. See class doc.
        """
        if self._green_lr is None:
            self._do_green_lr()
        return self._green_lr

    @property
    def green_gr(self):
        """
        Output variable getter. See class doc.
        """
        if self._green_gr is None:
            self._do_green_gr()
        return self._green_gr

    def _do_green_ret(self):
        """
        Virtual method: solve the Green retarded and fills a
        self._green_ret member
        """
        raise RuntimeError('Base Class Green has no _do_green_ret method')

    def _do_green_gr(self):
        """Calculate equilibrium Green's function.
        If the lesser is available avoid direct calculation of sigma_gr."""
        if self.green_lr is not None:
            self._green_gr = self.green_lr - 1j * self.spectral
        else:
            sigma_gr = np.zeros((self._size, self._size), dtype=np.complex128)
            self._add_leads(sigma_gr, "sigma_gr")
            gr = self.green_ret
            self._green_gr = np.dot(np.dot(gr, sigma_gr), gr.conj().T)
        return

    def _do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.asmatrix(
            np.zeros((self._size, self._size), dtype=np.complex128))
        self._add_leads(sigma_lr, "sigma_lr")
        green_ret = self.green_ret
        self._green_lr = np.dot(np.dot(green_ret, sigma_lr), green_ret.conj().T)
        return

    @property
    def spectral(self):
        """Get spectral function A = j(G^{r} - G^{a})"""
        return 1j * (self.green_ret - self.green_ret.conj().T)

    def _add_leads(self, mat, attribute):
        """Subtract all the leads contributions from matrix mat.
         Result in place. """
        for lead in self.leads:
            sigma = getattr(lead, attribute)
            assert sigma.shape[0] == sigma.shape[1]
            size = sigma.shape[0]
            pos = lead.position
            mat[pos: pos + size, pos:pos + size] = \
                mat[pos: pos + size, pos:pos + size] + sigma
        return

    def resize_leads_matrix(self, leads, varname):
        """
        An utility giving as output a matrix quantity defined in the leads as
        a full matrix (system size), summing them up if more leads are defined.
        Note, this use only the size of the Green and the position of the Lead,
        therefore it works also on leads not included in the instance.

        Typically needed to extract total self energies, total Gamma etc.

        Args:
            leads (Lead type or list of Leads): list of leads we want to resize
            varname (string): the attributes to be summed up

        Returns:
            out (numpy.ndarray): output summed up and decorate matrix
        """
        size = self._size
        leadmat = getattr(leads[0], varname)
        vartype = np.result_type(leadmat)
        mat = np.zeros((size, size), dtype=vartype)
        for lead in leads:
            leadmat = getattr(lead, varname)
            leadsize = leadmat.shape[0]
            pos = lead.position
            mat[pos: pos + leadsize, pos:pos + leadsize] += leadmat
        return mat

    def transmission(self, leads=None):
        """Calculate the transmission in place for a given couple of leads,
        specified as an iterable. If leads are not specified, use lead 0 and 1
        if set.
        It implements the Landauer Caroli formula"""
        if leads is not None:
            assert len(leads) == 2
            lead1 = leads[0]
            lead2 = leads[1]
        else:
            lead1 = self.leads[0]
            lead2 = self.leads[1]
        gamma1 = np.zeros((self._size, self._size), dtype=np.complex128)
        pos = lead1.position
        size = lead1.size
        gamma1[pos: pos + size, pos:pos + size] += lead1.gamma
        gamma2 = np.zeros((self._size, self._size), dtype=np.complex128)
        pos = lead2.position
        size = lead2.size
        gamma2[pos: pos + size, pos:pos + size] += lead2.gamma
        green_ret = self.green_ret
        trans = (np.trace(np.dot(np.dot(np.dot(gamma1, green_ret),
                                        gamma2), green_ret.conj().T)))
        return trans

    def meirwingreen(self, lead=None):
        """
        Calculate the total current in a specified lead
        by applying the Meir-Wingreen formula

        Args:
            lead (Lead): reference lead where the current is evaluated
                equation
        Returns:
            float: value of current in given energy point
        """
        assert lead is not None
        glr = self.green_lr
        ggr = self.green_gr
        sgr = self.resize_leads_matrix([lead], 'sigma_gr')
        slr = self.resize_leads_matrix([lead], 'sigma_lr')
        const = 1.0  # pyta.consts.e / pyta.consts.h_eVs)
        current = const * np.trace(np.dot(slr, ggr) - np.dot(sgr, glr))
        return current

    def dattacurrent(self, lead=None):
        """Calculate the current using the Datta version of Meir Wingreen:
            I = Sigma_lesser * A - Gamma * G_lesser

        Args:
            lead (Lead): reference lead where the current is evaluated
                equation

        Returns:
            float: value of current in given energy point
        """
        assert lead is not None
        green_n = -1.0j * self.green_lr
        sigma_n = -1.0j * self.resize_leads_matrix([lead], 'sigma_lr')
        gamma = self.resize_leads_matrix([lead], 'gamma')
        spectral = self.spectral
        const = 1.0
        current = (const *
                   np.trace(np.dot(sigma_n, spectral) -
                            np.dot(gamma, green_n)))
        return current

    @property
    def occupation(self):
        """
        Output variable getter. See class doc.
        """
        diag1 = np.diag(self.green_lr)
        diag2 = np.diag(self.spectral)
        occupation = (np.imag(diag1 / diag2))
        return occupation

    def scba(self, lead, mode='equilibrium', niter=None, maxiter=None,
             tolerance=1e-6, alpha=1.0, gzero=None):
        """
        Perform Born Approximation mixing with respect to self energy contained
        in lead specified in input
        """

        if mode == 'equilibrium':
            green_varname = 'green_ret'
            sigma_varname = 'sigma_ret'
        elif mode == 'keldysh':
            green_varname = 'green_lr'
            sigma_varname = 'sigma_lr'
        else:
            raise AttributeError('Unknown mode: must be equilibrium or keldysh')

        #Initialize the virtual lead self-energy to zero or to given input value
        if gzero is None:
            setattr(lead, green_varname, np.zeros((self._size, self._size)))
        else:
            setattr(lead, green_varname, gzero)

        def func1():
            self.reset()
            return getattr(self, green_varname)

        def func2(var2):
            ## Here var2 in input must be the right green's function and the
            ## return value is the corresponding sigma
            setattr(lead, green_varname, var2)
            return getattr(lead, sigma_varname)

        ## G0 calculated here (self-energy initialized to zero)
        var1_guess = getattr(self, green_varname)
        mathutils.linear_mixer(func1, func2, var1_guess, niter=niter,
                               tolerance=tolerance,
                               alpha=alpha, maxiter=maxiter)


class ElGreen(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self,
                 #Parameters
                 ham, over=None, delta=0.0):
        """ElGreen is initialized by specifying an Hamiltonian as 2D numpy.ndarray.
        Optionally, overlap can be specified.
        If overlap is not specified, an orthogonal basis is assumed.

        See base class Green

        invar:
        1) energy

        2) leads
            set of Leads

        parameters:
        1)  ham
            system hamiltonian, 2D numpy.ndarray
        2)  overlap (optional)
            overlap matrix, real 2D numpy.ndarray
        3)  delta (optional)
            if no leads are present, it is used for shifting poles

        """

        #Param
        #========================================================
        if over is not None:
            self.over = over
        self.ham = ham

        #Consistency check (yes, it happened to wrongly set a non hermitian
        # hamiltonian...good luck spotting that without a check)
        if (np.abs(ham - ham.conj().T) > 1e-7).any():
            raise ValueError(
                'Error in Green parameter. The Hamiltonian is not hermitian')

        size = len(self.ham)
        if over is None:
            self.over = np.eye(size)

        self.delta = delta
        #Invar
        #=================================
        self._energy = None
        #=================================

        #Base constructor
        super(ElGreen, self).__init__(size)

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
            self.reset()
            #Update energy point in all leads where set_energy is defined
            self._spread_energy(value)
        return

    def _spread_energy(self, energy):
        """
        Distribute energy in leads
        """
        for lead in self.leads:
            try:
                getattr(lead, "energy")
                lead.energy = energy
            except AttributeError:
                pass
        return

    @property
    def local_currents(self):
        """
        Returns:
            the matrices of local currents for all orbitals in the system
        """
        green_lr = self.green_lr
        ham = self.ham
        over = self.over
        energy = self.energy
        currents = np.real(np.multiply(2. * (ham - energy * over), green_lr))
        return currents

    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        sigma = np.zeros((self._size, self._size), dtype=np.complex128)
        esh = self.energy * self.over - self.ham
        self._add_leads(sigma, 'sigma_ret')
        ## Note: I add an imaginary part, to avoid problem if I have no
        ## leads. Anyway it is small respect to self energy. If you don't want
        ## it, set delta to 0.0 in constructor
        esh = esh - sigma + 1j*self.delta * self.over
        self._green_ret = np.linalg.inv(esh)

        return self.green_ret


class PhGreen(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""

    def __init__(self,
                 #Param
                 spring, mass=None, delta=0.0):
        """PhGreen is initialized by specifying a coupling spring constant
        matrix as numpy.ndarray.
        Optionally, masses can be specified.
        Masses must be specified as a diagonal numpy.ndarray
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
            spring constants, numpy.ndarray
        2)  mass (optional)
            mass matrix, real numpy.ndarray

        """

        #Param
        #=======================================
        self.spring = spring
        self.mass = mass
        #assert (type(spring) == np.matrixlib.defmatrix.matrix)
        #if not (mass is None):
        #    assert (type(spring) == np.matrixlib.defmatrix.matrix)
        self.spring = spring
        size = len(self.spring)
        if mass is None:
            self.mass = np.eye(size)
        #======================================

        #Invar
        #=======================================
        self._frequency = None
        self.delta = delta
        #=======================================

        #Base constructor
        super(PhGreen, self).__init__(size)

    @property
    def frequency(self):
        """
        Returns:
            frequency (float): frequency point
        """
        return self._frequency

    @frequency.setter
    def frequency(self, value):
        """Set energy point"""
        if value != self._frequency:
            self._frequency = value
            self.reset()
            self._spread_frequency(value)

    def _spread_frequency(self, frequency):
        """Distribute energy in leads"""
        for lead in self.leads:
            try:
                getattr(lead, "frequency")
                lead.frequency = frequency
                lead.set_frequency(frequency)
            except AttributeError:
                pass
        return

    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        sigma = np.zeros((self._size, self._size), dtype=np.complex128)
        esh = self._frequency * self._frequency * self.mass - self.spring
        self._add_leads(sigma, "sigma_ret")
        ## Note: I add an imaginary part, to avoid problem if I have no
        ## leads. Anyway it is small respect to self energy. If you don't want
        ## it, set delta to 0.0 in constructor
        esh = esh + 1j*self.delta * self.mass
        self._green_ret = np.linalg.inv(esh)

        return self.green_ret
