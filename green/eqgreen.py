import numpy as np
import pyta.core.defaults as defaults

class Green:
    """A class for equilibrium Green's function"""
    def __init__(self, energy, h, s = None, delta = defaults.delta):
        """EqGreen is initialized by specifying an Hamiltonian as numpy.matrix.
        Optionally, overlap and imaginary delta can be specified."""

        assert(type(h) == numpy.matrixlib.defmatrix.matrix)
        if s:
            assert(type(s) == numpy.matrixlib.defmatrix.matrix)
        self._h = h
        self._n = len(self._h)
        if s == None:
            self._s = np.matrix(np.eye(self._n))
        self.lead_names = list()
        self.selfenergies = dict()
        self.selfener = np.matrix(np.zeros((self._n, self_n)))
        self._energy = energy


    def get_ham(self):
        return self._h

    def get_over(self):
        return self._s

    def add_selfenergy(self, selfener, name = None):
        """Explicitely add a Self Energy.
        A single matrix with the same size of the system can be passed. 
        You can associate a name to the self energy which can be used as lead
        identifier (also for virtual leads). 

        Self energies are stored by reference, and summed up when set.
        If you need to refresh the sum (i.e. a self energy changed value), use
        refresh_selfenergy(),"""
        if not name:
            ind = 0
            is_set = False
            while is_set == False:
                name = str(ind)
                if not(name in sef.lead_names):
                    is_set = True
                    self.lead_name.append(name)
                else:
                    ind += 1
        else:
            self._lead_names.append
        self.selfenergies += selfener


        
