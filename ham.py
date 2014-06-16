import numpy as np
import solver

""" A Module with some utilities to build up Hamiltonian and Overlap
    matrix of simple systems."""


class LinearChainHam(solver.Solver):
    """A class providing a simple linear chain nearest neighbor hamiltonian."""

    def __init__(self,
                 #parameters
                 length, onsite=0.0, hopping=1.0, overlap=0.0):
        """Constructor arguments:
        length (int): chain length
        onsite (float or complex): onsite energy
        hopping (float or complex): hopping matrix element
        overlap (float): nearest neighbor overlap (if any).
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




