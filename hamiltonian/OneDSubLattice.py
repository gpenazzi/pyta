import numpy as np

class OneDSubLattice():
    """Build the TB Hamiltonian for a 1d system with two sublattices.
    The Hamiltonian is:

    e_a     t_ab    t_aa    0
    t_ab+   e_b     t_ab    t_bb
    t_aa+   t_ab+   e_a     t_ab
    0       t_bb+   t_ab+   e_b
    """
    def __init__(self,
            #Constant
            t_ab, t_aa, t_bb, e_a=0.0, e_b=0.0):

        self._ham = 1j*np.zeros((2,2))
        self._over = np.eye(2)

        self._ham[0,0] = e_a
        self._ham[1,1] = e_b
        #self._ham[2,2] = e_a
        #self._ham[3,3] = e_b
        self._ham[0,1] = t_ab
        #self._ham[0,2] = t_aa
        self._ham[1,0] = np.conj(t_ab)
        #self._ham[1,2] = t_ab
        #self._ham[2,0] = np.conj(t_aa)
        #self._ham[2,1] = np.conj(t_ab)
        #self._ham[2,3] = t_ab
        #self._ham[3,1] = np.conj(t_bb)
        #self._ham[3,2] = np.conj(t_ab)
   
    def get_ham(self):
        """Get the hamiltonian"""
        return self._ham

    def get_k_ham(self, k_point):
        """Get the Hamiltonian including Bloch terms.
        k is referred to Brillouin zone edge (a_max=1.0).
        Atom B is always assumed in the middle of the unit cell"""

        ham = self._ham.copy()
        ham[1,0] = ham[1,0] * (1.0 + np.exp(-1j * np.pi * k_point))
        ham[0,1] = ham[0,1] * (1.0 + np.exp(1j * np.pi * k_point))

        return ham

    def get_band(self, k_points):
        """Get band structure over an array of k points"""
        eigenenergies = list() 
        for ind, k_point in enumerate(k_points):
            eigenenergies.append(np.linalg.eig(self.get_k_ham(k_point))[0])

        return np.array(eigenenergies)
            
