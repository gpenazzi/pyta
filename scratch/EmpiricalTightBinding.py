import ase
import ase.calculators.neighborlist
import numpy as np
import scipy.linalg

class EmpiricalTightBinding():
    """Build ETB Hamiltonian of structures imported by ASE, according to e
    specific dictionary of parameter"""

    def __init__(self, 
            #Constant
            structure, param, onsite, overparam=None):
        """Structure is the atomistic structure. Param is a dictionary of
        parameter built as follows:

        key: (sp1, sp2) where sp1 and sp2 are identifiers (atomic species in
        this model)
        item: (dmin, dmax, val): dmin and dmax define minimum and maximum
        distances (to distinguish first, second, third neighbors etc.). Val is
        the Hamiltonian element.
        
        Onsite contains a dictionary of onsites {sp: onsite}
        All energies in eV"""

        self._structure = structure
        self._param = param
        self._onsite = onsite
        self._nnlist = None
        self._size = len(structure)
        self._ham = None
        self._overparam = None
        self._over = None

        self._define_nnlist()

    def get_ham(self):
        if self._ham == None:
            self.do_ham()
        return self._ham

    def get_over(self):
        if self._overparam is None:
            return None
        else:
            if self._over is None:
                self.do_over()
            else:
                return self._over

    def _define_nnlist(self):
        """Set up a neighbor list according to distance cutoff in
        parameterization"""

        cutoffs = dict()
        for atom in self._structure:
            cutoffs[atom.symbol] = 0.0
        for atom in self._structure:
            for key in self._param:
                if ((atom.symbol == key[0]) or (atom.symbol ==
                    key[1])):
                    for distcoupl in self._param[key]:
                        if 0.5 * distcoupl[1] > cutoffs[atom.symbol]:
                            cutoffs[atom.symbol] = 0.5 * distcoupl[1]

        cutoff_per_atom = list()
        for atom in self._structure:
            cutoff_per_atom.append(cutoffs[atom.symbol])

        self._nnlist = \
        ase.calculators.neighborlist.NeighborList(cutoff_per_atom,
                skin=0.0, bothways=False)
        self._nnlist.build(self._structure)
    

    def get_ham(self, k_point = 0.0):
        """Pretty intuitive. This assembles the Hamiltonian"""

        size = len(self._structure)
        self._ham = 1j*np.zeros((size, size))

        for ind, atm1 in enumerate(self._structure):
            nnlist = self._nnlist.get_neighbors(ind)
            #On every neighbor
            for neighind, neigh in enumerate(nnlist[0]):
                atm2 = self._structure[neigh]
                #Onsites
                if ind == neigh and (nnlist[1] == [0, 0, 0]).all():
                    self._ham[ind, ind] = self._onsite[(atm1.symbol)]
                else:
                    distance = (atm1.position - atm2.position - 
                    np.dot(nnlist[1][neighind], self._structure.get_cell()))
                    values = self._param[(atm1.symbol, atm2.symbol)]
                    phase = (-1j * np.pi * k_point * distance[2] /
                        self._structure.get_cell()[2,2])
                    for value in values:
                        if (np.linalg.norm(distance) > value[0]) and (np.linalg.norm(distance) < value[1]):
                            self._ham[ind, neigh] = self._ham[ind, neigh] + value[2] * np.exp(phase)
                            self._ham[neigh, ind] =  self._ham[neigh, ind] + value[2] * np.exp(-phase)
        if k_point == 0:
            np.savetxt('ham.txt', self._ham)
        self._ham[5,5]=self._ham[5,5]-0.2 
        return self._ham
       
    def do_over(self):
        """Pretty intuitive. This assembles the Hamiltonian"""

        size = len(self._structure)
        self._over = np.zeros((size, size))

        for ind, atm1 in enumerate(self._structure):
            nnlist = self._nnlist.get_neighbors(ind)
            #On every neighbor
            for neighind, neigh in enumerate(nnlist[0]):
                atm2 = self._structure[neigh]
                #Onsites
                if ind == neigh and (nnlist[1] == [0, 0, 0]).all():
                    self._over[ind, ind] = 1.0
                else:
                    distance = np.linalg.norm(atm1.position - atm2.position - 
                    np.dot(nnlist[1][neighind], self._structure.get_cell()))
                    values = self._overparam[(atm1.symbol, atm2.symbol)]
                    for value in values:
                        print('values', values)
                        if (distance > value[0]) and (distance < value[1]):
                            self._over[ind, neigh] = value[2]
                            self._over[neigh, ind] = value[2]

        print('over is',self._over)


    def get_k_ham(self, k_point):
        """Get the Hamiltonian including Bloch terms.
        k is referred to Brillouin zone edge (a_max=1.0).
        Atom B is always assumed in the middle of the unit cell"""

        #DO IT BETTER WITH NEIGHBOR LIST. NOW ONLY FOR 1D in Z DIR
        ham = self.get_ham().copy()
        for ind, atm1 in enumerate(self._structure):
            nnlist = self._nnlist.get_neighbors(ind)
            #On every neighbor
            for neighind, neigh in enumerate(nnlist[0]):
                atm2 = self._structure[neigh]
                #Onsites
                if ind == neigh and (nnlist[1] == [0, 0, 0]).all():
                    pass
                else:
                    distance = (atm1.position - atm2.position - 
                    np.dot(nnlist[1][neighind], self._structure.get_cell()))
                    ham[ind, neigh] = ham[ind, neigh] * (1.0 + np.exp(-1j *
                        np.pi * k_point * distance[2] /
                        self._structure.get_cell()[2,2]))

        return ham


    def get_band(self, k_points):
        """Get band structure over an array of k points"""
        eigenenergies = list() 
        for ind, k_point in enumerate(k_points):
            eigenenergies.append(scipy.linalg.eigvals(self.get_ham(k_point),
                b=self.get_over()))

        return np.array(eigenenergies)

