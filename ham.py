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


def random_hopping(mask, delta, seed=None, diag=False, tol=0.0):
    """
    Return a matrix containing a random disorder in the interval -delta, +delta
    on the non-zero non-diagonal values of a mask matrix.
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
    if not diag:
        np.fill_diagonal(disorder, 0.0)
    return disorder


def linear_ham(length, onsite, hopping):
    """Returns a matrxi with a linear chain hamiltonian"""
    n = length
    ham = np.matrix(np.zeros((n, n)))
    for i in range(n - 1):
        ham[i, i + 1] = np.conj(hopping)
        ham[i + 1, i] = hopping
        for i in range(n):
            ham[i, i] = onsite
    return ham


class LinearChainHam(solver.Solver):
    """A solver class providing a simple linear chain nearest neighbor hamiltonian.
    
        Parameters 
        ----------------------
        length (int): chain length, as number of repetition of H
        onsite (float): onsite energy. A NxN block can be specified
        hopping (float): hopping matrix element. A NxN block can be
                                            specified

        Invar
        -----------------------
        delta_h (ndarray_like): add a shift to the Hamiltonian. The input
                                quantity must be of same size as the
                                hamiltonian.

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

        #Invar
        self.delta_h = None

        #First hamiltonian and overlap calculation
        self._init_ham_over()


    def _init_ham_over(self):
        """Private: create the first hamiltonian and overlap"""
        n = self.length
        self.ham = np.matrix(np.zeros((n, n)))
        self.over = np.matrix(np.identity(n))
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



