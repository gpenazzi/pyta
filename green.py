import numpy as np
from pyta import defaults
from pyta import mathutils
from pyta import solver
import pyta.lead
import pyta.consts
import copy


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
        self.size = size

        #Outvar
        self.green_ret = None
        self.green_gr = None
        self.green_lr = None

        super(Green, self).__init__()

    def reset(self):
        """
        Set all output variables to undefined state
        """
        self.green_ret = None
        self.green_lr = None
        self.green_gr = None

    def get_green_ret(self):
        """
        Returns:
            np.matrix: Retarded Green's function
        """
        if self.green_ret is None:
            self._do_green_ret()
        return self.green_ret

    def get_green_lr(self):
        """
        Returns:
        np.matrix: Lesser Green's function
        """
        if self.green_lr is None:
            self._do_green_lr()
        return self.green_lr

    def get_green_gr(self):
        """
        Returns:
            np.matrix: Greater Green's function
        """
        if self.green_gr is None:
            self._do_green_gr()
        return self.green_gr

    def _do_green_ret(self):
        raise RuntimeError('Base Class Green has no _do_green_ret method')

    def _do_green_gr(self):
        """Calculate equilibrium Green's function.
        If the lesser is available avoid direct calculation of sigma_gr."""
        if self.green_lr is not None:
            spectral = self.get_spectral()
            green_lr = self.get_green_lr()
            self.green_gr = green_lr - 1j * spectral
        else:
            sigma_gr = np.asmatrix(
                np.zeros((self.size, self.size), dtype=np.complex128))
            self._add_leads(sigma_gr, "sigma_gr")
            gr = self.get_green_ret()
            self.green_gr = np.dot(np.dot(gr, sigma_gr),gr.conj().T)
        return

    def _do_green_lr(self):
        """Calculate equilibrium Green's function"""
        sigma_lr = np.asmatrix(
            np.zeros((self.size, self.size), dtype=np.complex128))
        self._add_leads(sigma_lr, "sigma_lr")
        green_ret = self.get_green_ret()
        self.green_lr = np.dot(np.dot(green_ret, sigma_lr), green_ret.conj().T)
        return

    def set_leads(self, leads):
        self.leads = leads

    def get_spectral(self):
        """Get spectral function A = j(G^{r} - G^{a})"""
        green_ret = self.get_green_ret()
        spectral = 1j * (green_ret - green_ret.conj().T)
        return spectral

    def _add_leads(self, mat, varname):
        """Subtract all the leads contributions from matrix mat.
         Result in place. """
        for lead in self.leads:
            sigma = lead.get(varname)
            assert (sigma.shape[0] == sigma.shape[1])
            size = sigma.shape[0]
            pos = lead.position
            mat[pos: pos + size, pos:pos + size] = \
                mat[pos: pos + size, pos:pos + size] + sigma
        return

    def _resize_lead_matrix(self, lead, varname):
        """Resize a lead matrix in a matrix with the shape of the Green.
        Elements out of the lead are set to zero.
        Variable to be converted must be given as string."""
        leadmat = lead.get(varname)
        vartype = np.result_type(leadmat)
        mat = np.zeros((self.size, self.size), dtype=vartype)
        leadsize = leadmat.shape[0]
        pos = lead.position
        mat[pos: pos + leadsize, pos:pos + leadsize] = leadmat
        return mat


    def get_transmission(self, leads=None):
        """Calculate the transmission in place for a given couple of leads,
        specified as an iterable. If leads are not specified, use lead 0 and 1
        if set.
        It implements the Landauer Caroli formula"""
        if leads is not None:
            assert (len(leads) == 2)
            lead1 = leads[0]
            lead2 = leads[1]
        else:
            lead1 = self.leads[0]
            lead2 = self.leads[1]
        gamma1 = np.zeros((self.size, self.size), dtype=np.complex128)
        pos = lead1.position
        size = lead1.size
        gamma1[pos: pos + size, pos:pos + size] += lead1.get('gamma')
        gamma2 = np.zeros((self.size, self.size), dtype=np.complex128)
        pos = lead2.position
        size = lead2.size
        gamma2[pos: pos + size, pos:pos + size] += lead2.get('gamma')
        green_ret = self.get_green_ret()
        trans = (np.trace(np.dot(np.dot(np.dot(gamma1, green_ret)
                                        , gamma2),green_ret.conj().T)))
        return trans

    def get_meirwingreen(self, lead=None):
        """
        Calculate the total current in a specified lead
        by applying the Meir-Wirgreen formula

        Args:
            lead (Lead): reference lead where the current is evaluated
                equation
        Returns:
            float: value of current in given energy point
        """
        assert (lead is not None)
        glr = self.get_green_lr()
        ggr = self.get_green_gr()
        sgr = self._resize_lead_matrix(lead, 'sigma_gr')
        slr = self._resize_lead_matrix(lead, 'sigma_lr')
        const = 1.0  #pyta.consts.e / pyta.consts.h_eVs)
        current = const * np.trace(np.dot(slr, ggr) - np.dot(sgr, glr))
        return current

    def get_dattacurrent(self, lead=None):
        """Calculate the current using the Datta version of Meir Wingreen:
            I = Sigma_lesser * A - Gamma * G_lesser

        Args:
            lead (Lead): reference lead where the current is evaluated
                equation
        Returns:
            float: value of current in given energy point
        """
        assert (lead is not None)
        green_n = -1.0j * self.get_green_lr()
        sigma_n = -1.0j * self._resize_lead_matrix(lead, 'sigma_lr')
        gamma = self._resize_lead_matrix(lead, 'gamma')
        spectral = self.get_spectral()
        current = ((pyta.consts.e / pyta.consts.h_eVs) *
                   np.trace(np.dot(sigma_n, spectral) - np.dot(gamma , green_n)))
        return current

    def get_occupation(self):
        """Calculate the occupation by comparing the lesser green function
        and the spectral function. """
        diag1 = self.get_green_lr()
        diag2 = self.get_spectral()
        occupation = (np.imag(diag1 / diag2))
        return occupation

    def scba(self, lead, mode='equilibrium', niter=None, maxiter=None,
             tolerance=None, alpha=1.0):
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

        #Initialize the virtual lead self-energy to zero
        lead.set(green_varname, np.zeros((self.size, self.size)))

        def func1(var1):
            ## Note: var1 here is dummy because it is automatically retrievable
            ## everytime a lead.get is invoked in func2
            self.reset()
            return self.get(green_varname)

        def func2(var2):
            ## Here var2 in input must be the right green's function and the
            ## return value is the corresponding sigma
            lead.set(green_varname, var2)
            return lead.get(sigma_varname)

        ## G0 calculated here (self-energy initialized to zero)
        var1_guess = self.get(green_varname)
        mathutils.linear_mixer(func1, func2, var1_guess, niter=niter,
                               tolerance=tolerance,
                               alpha=alpha, maxiter=maxiter)


class ElGreen(Green):
    """Build and manage Green's function for Fermions. Only the method which
    differentiate fermions from other particles are reimplemented here"""

    def __init__(self,
                 #Parameters
                 ham, over=None, delta=0.0):
        """ElGreen is initialized by specifying an Hamiltonian as numpy.matrix.
        Optionally, overlap can be specified.
        If overlap is not specified, an orthogonal basis is assumed.

        See base class Green

        invar:
        1) energy
            mode: overwrite

        2) leads
            set of Leads
            mode: overwrite, append

        parameters:
        1)  ham
            system hamiltonian, numpy.matrix
        2)  overlap (optional)
            overlap matrix, real numpy.matrix
        3)  delta (optional)
            if no leads are present, it is used for shifting poles

        """

        #Param
        #========================================================
        assert (type(ham) == np.matrixlib.defmatrix.matrix)
        if over is not None:
            assert (type(over) == np.matrixlib.defmatrix.matrix)
            self.over = over
        self.ham = ham

        #Consistency check (yes, it happened to wrongly set a non hermitian
        # hamiltonian...good luck spotting that without a check)
        if (np.abs(ham - ham.conj().T) > 1e-7 ).any():
            raise ValueError(
                'Error in Green parameter. The Hamiltonian is not hermitian')

        size = len(self.ham)
        if over is None:
            self.over = np.asmatrix(np.eye(size))

        self.delta = delta
        #Invar
        #=================================
        self.energy = None
        #=================================

        #Base constructor
        super(ElGreen, self).__init__(size)


    def set_energy(self, energy):
        """Set energy point"""
        if energy != self.energy:
            self.energy = energy
            self.reset()
            #Update energy point in all leads where set_energy is defined
            self._spread_energy(energy)
        return

    def _spread_energy(self, energy):
        "Distribute energy in leads"
        for lead in self.leads:
            try:
                lead.set('energy', energy)
            except AttributeError:
                pass
        return

    def get_local_currents(self):
        """
        Calculate the matrices of local currents for all orbitals in the system
        """
        green_lr = self.get_green_lr()
        ham = self.ham
        over = self.over
        en = self.energy
        lc = np.real(np.multiply(2. * (ham - en * over), green_lr))
        print('lc', lc[1, 2], 'energy ', en, 'gl', green_lr[1, :])
        return lc

    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        sigma = np.asmatrix(
            np.zeros((self.size, self.size), dtype=np.complex128))
        esh = self.energy * self.over - self.ham
        self._add_leads(sigma, 'sigma_ret')
        ## Note: I add an imaginary part, to avoid problem if I have no
        ## leads. Anyway it is small respect to self energy. If you don't want
        ## it, set delta to 0.0 in constructor
        esh = esh - sigma + 1j*self.delta * self.over
        self.green_ret = esh.I

        return self.green_ret


class PhGreen(Green):
    """Build and manage Green's function for phonons. Only the method which
    differentiate phonons from other particles are reimplemented here"""

    def __init__(self,
                 #Param
                 spring, mass=None, delta=0.0):
        """PhGreen is initialized by specifying a coupling spring constant
        matrix as numpy.matrix.
        Optionally, masses can be specified.
        Masses must be specified as a diagonal numpy.matrix
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
            spring constants, numpy.matrix
        2)  mass (optional)
            mass matrix, real numpy.matrix

        """

        #Param
        #=======================================
        self.spring = spring
        self.mass = mass
        assert (type(spring) == np.matrixlib.defmatrix.matrix)
        if not (mass is None):
            assert (type(spring) == np.matrixlib.defmatrix.matrix)
        self.spring = spring
        size = len(self.spring)
        if mass is None:
            self.mass = np.asmatrix(np.eye(size))
        #======================================

        #Invar
        #=======================================
        self.frequency = None
        self.delta = delta
        #=======================================

        #Base constructor
        super(PhGreen, self).__init__(size)

    def set_frequency(self, frequency):
        """Set energy point"""
        if frequency != self.frequency:
            self.frequency = frequency
            self.reset()
            self._spread_frequency(frequency)

    def _spread_frequency(self, frequency):
        "Distribute energy in leads"
        for lead in self.leads:
            try:
                lead.set_frequency(frequency)
            except AttributeError:
                pass
            else:
                lead.set_frequency(frequency)
        return

    def _do_green_ret(self):
        """Calculate equilibrium Green's function"""
        esh = self.frequency * self.frequency * self.mass - self.spring
        self._add_lead_sigma(esh)
        esh = esh + 1j*self.delta * self.over
        self.green_ret = esh.I

        return self.green_ret

