import numpy as np
import solver

""" A Module with some utilities to build up Hamiltonian and Overlap
    matrix of simple systems."""


def random_onsite(size, delta, seed=None):
    """
    Return a matrix containing a random disorder in the interval -delta, +delta

    Parameters
    ----------------
    int (int or matrix): matrix size (for square matrix)
    delta (float): disorder interval
    seed (hashable): random generator seed
    """
    if type(size) != int or type(size) != np.int:
        if size.shape[0] != size.shape[1]:
            raise ValueError('size must be an integer or a square matrix')
        size = size.shape[0]
    np.random.seed(seed)
    disorder = np.zeros((size, size))
    disorder_diag = np.random.rand(size) * 2.0 * delta - delta
    diag_ind = np.diag_indices(size)
    disorder[diag_ind] = disorder_diag

    return disorder


def gaussian_onsite(size, stddev, seed=None):
    """
    Return a matrix containing a gaussian with standard deviation stddev

    Parameters
    ----------------
    int (int or matrix): matrix size (for square matrix)
    delta (float): disorder interval
    seed (hashable): random generator seed
    """
    if type(size) != int or type(size) != np.int:
        if size.shape[0] != size.shape[1]:
            raise ValueError('size must be an integer or a square matrix')
        size = size.shape[0]
    np.random.seed(seed)
    disorder = np.zeros((size, size))
    disorder_diag = np.random.normal(scale=stddev, size=size)
    diag_ind = np.diag_indices(size)
    disorder[diag_ind] = disorder_diag

    return disorder


def random_hopping(mask, delta, seed=None, tol=0.0):
    """
    Return a matrix containing a random disorder in the interval -delta, +delta
    on the non-zero non-diagonal values of a mask matrix.
    Disorder is added symmetrically, such that the Hamiltonian
    is still hermitian.
    If diag='True' is specified, also the diagonal is filled

    Parameters
    ----------------
    int (int or matrix): matrix size (for square matrix)
    delta (float): disorder interval
    seed (hashable): random generator seed
    tol (float): specify the tolerance to consider a value of mask matrix zero
    """
    if mask.shape[0] != mask.shape[1]:
        raise ValueError('mask must be a square matrix')
    size = mask.shape[0]
    np.random.seed(seed)
    disorder = np.random.random(mask.shape) * 2.0 * delta - delta
    disorder[abs(mask)<=tol] = 0.0
    lowdiag_disorder = np.tril(disorder)
    disorder = lowdiag_disorder + lowdiag_disorder.T
    return disorder


def linear_ham(length, onsite, hopping):
    """Returns a matrxi with a linear chain hamiltonian"""
    n = length
    ham = np.asmatrix(np.zeros((n, n)))
    for i in range(n - 1):
        ham[i, i + 1] = np.conj(hopping)
        ham[i + 1, i] = hopping
        for i in range(n):
            ham[i, i] = onsite
    return ham


class GaussianProcess(solver.Solver):
    """
    A solver providing a model of time dependent Hamiltonian H(t) mimicking a
    Gaussian process with give time correlation.
    The Gaussian disorder is added on all the onsite energies

    Parameters
    ---------------------------------------------
    length (int): chain length (number of orbitals). Note: model is only tested
        for single orbital linear chains, not for block chains
    onsite (float): onsite energy.
    hopping (float): hopping matrix element.
    timesteps (int): length of time interval
    variance (float): variance of gaussian disorder
    correlation_length (int): correlation length of the process

    Output
    ----------------------------------------------
    hzero (ndarray length x length): matrix with unperturbed Hamiltonian
    ham   (ndarray time x length x length): time serie with perturbation
    """
    def __init__(self,
                 length=1,
                 onsite=0.0,
                 hopping=1.0,
                 variance=0.1,
                 correlation_length=1,
                 timesteps=1):
        self.hzero = linear_ham(length=length, onsite=onsite,
                                hopping=hopping)
        self.ham = np.zeros(shape=(timesteps, length, length))
        self.variance = variance
        self.hopping = hopping
        self.correlation_length = correlation_length
        self.timesteps = timesteps
        self.length = length

        ff = np.exp(-1.0/correlation_length)
        ff2 = ff*ff

        for ii in range(timesteps):
            self.ham[ii] = self.hzero + np.sqrt(1.0-ff2) * gaussian_onsite(
                length, np.sqrt(self.variance)) + ff * self.ham[ii-1]

    def plot_me(self):
        import matplotlib.pyplot as plt
        color = ['r', 'g', 'b', 'y']
        for nn in range(self.length):
            print(color[nn%len(color)])
            plt.plot(self.ham[:, nn, nn], color[nn%len(color)])
        plt.show()

    def show_spectrum(self):
        import matplotlib.pyplot as plt
        spectrum = np.fft.fft(self.ham, axis=0)
        color = ['r', 'g', 'b', 'y']
        for nn in range(self.length):
            print(color[nn%len(color)])
            plt.plot(np.real(spectrum[:, nn, nn]), color[nn%len(color)])
        plt.show()




class Stub(solver.Solver):
    """
    A solver providing a simple TB model of a stub, constructed as
    (P is the perturbation):

                        P
                        |
    ...**********************************...

    Parameters
    -----------------------------------
    length (int): chain length, as number of repetition of H
    onsite (float): onsite energy.
    hopping (float): hopping matrix element.
    onsite_perturbation (float): perturbation on-site energy
    hopping_perturbation (float): hopping element between P ad the chain
    interacting_sites (int or list): where to put P. Note that we can have several
        perturbation

    """
    def __init__(self,
                 length,
                 interacting_sites,
                 onsite_perturbation=0.0,
                 hopping_perturbation=1.0,
                 onsite=0.0,
                 hopping=1.0):
        n = length
        if type(interacting_sites) is int:
            interacting_sites = [interacting_sites]
        size = len(interacting_sites) + n
        self.ham = np.zeros((size, size))
        for i in range(n - 1):
            self.ham[i, i + 1] = np.conj(hopping)
            self.ham[i + 1, i] = hopping
        for i in range(n):
            self.ham[i, i] = onsite
        #Add the perturbations
        for ii in range(n,size):
            self.ham[ii, ii] = onsite_perturbation
        for ind, ii in enumerate(interacting_sites):
            self.ham[n+ind, ii] = np.conj(hopping_perturbation)
            self.ham[ii, n+ind] = hopping_perturbation



class LinearChainHam(solver.Solver):
    """
    A solver providing a simple linear chain nearest neighbor
    hamiltonian. Mainly for testing purpose
    
        Parameters 
        ----------------------
        length (int): chain length, as number of repetition of H
        onsite (float): onsite energy. A NxN block can be specified
        hopping (float): hopping matrix element. A NxN block can be
                                            specified


        Outvar
        -----------------------
        ham (matrix): Hamiltonian of the syste
        over (matrix): Overlap matrix of the system

    """

    def __init__(self,
                 #parameters
                 length, onsite=0.0, hopping=1.0, overlap=None):
        """
            Parameters
            ------------------

        length (int): chain length, as number of repetition of H
        onsite (float): onsite energy. 
        hopping (float): hopping matrix element.
        overlap (float): Overlap in sub-diagonal and
                                        super-diagonal elements
        
        """

        #Param
        self.length = length
        self.onsite = onsite
        self.hopping = hopping
        self.overlap = overlap

        #Outvar
        self.ham = None
        self.over = None

        #First hamiltonian and overlap calculation
        self._init_ham_over()


    def _init_ham_over(self):
        """Private: create the first hamiltonian and overlap"""
        n = self.length
        self.ham = np.zeros((n, n))
        self.over = np.identity(n)
        for i in range(n - 1):
            self.ham[i, i + 1] = np.conj(self.hopping)
            self.ham[i + 1, i] = self.hopping
            self.over[i, i + 1] = self.overlap
            self.over[i + 1, i] = self.overlap
        for i in range(n):
            self.ham[i, i] = self.onsite
        return


    def _do_ham(self):
        """
        Update the Hamiltonian when a shift is set
        """
        self.ham += self.delta_h



