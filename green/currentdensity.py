import numpy as np
import pyta.grid.cubicgrid 

"""Methods and class to build real space current density"""

class CurrentDensity:
    """ A class to calculate and plot real space resolved current density"""
    def __init__(self, 
                #Constant
                orbind, #list or array of orbital indexes
                orb,    #orbitals (orbital indexes point to these) 
                orb_r,  #orbital positions, 2D array (row orbital, col
                #coordinates)
                weight,  #weight for current operator (density matrix)
                res = 0.05, #grid resolution, the other parameter are calculated
                           #internally
                tol = 3.0 # Lateral Grid tolerance in Angstrom
                ):

        #Constants
        self._orbind = orbind
        self._orb = orb
        self._orb_r = orb_r
        self._weight = weight  # Must be a coo sparse
        self._res = res

        #Dependent variables
        self._j_vec = None
        self._j_mag = None

        #Grid data, this will be moved to a separate class
        self._res = res
        self._tol = tol
        self._x_n = 0
        self._y_n = 0
        self._z_n = 0
        self._min_xyz = np.zeros(3)
        self._x_grid = None
        self._y_grid = None
        self._z_grid = None

        # Local
        self._norb = len(self._orbind)
        self._nonzerocouples = np.zeros((self._norb,2),dtype=int)
        self._grid = pyta.grid.cubicgrid.CubicGrid((self._orb_r, self._tol),
                self._res)
        
        self._j_vec = np.zeros((self._grid.get_npoints()[0],
            self._grid.get_npoints()[1], self._grid.get_npoints()[2], 3))
        self._j_mag = np.zeros((self._grid.get_npoints()[0],
            self._grid.get_npoints()[1], self._grid.get_npoints()[2]))
    

    def _do_currdensop(self, ii, jj, coord):
        """Calculate the current density operator between two orbitals ii and jj at a
        given coordinate"""
         
        i_orb = self._orb[self._orbind[ii]]
        j_orb = self._orb[self._orbind[jj]]
        i_coord = coord - self._orb_r[ii,:]
        j_coord = coord - self._orb_r[jj,:]
        #NOTE: implemented for real wavefunctions
        if ((i_coord > i_orb.get_cutoff()).any() or  
            (j_coord > j_orb.get_cutoff()).any() or   
            (i_coord < -i_orb.get_cutoff()).any() or 
            (j_coord < -j_orb.get_cutoff()).any()):
            return np.zeros(3)
        else:
            val = i_orb.get_value(i_coord) * j_orb.get_grad(j_coord) - \
                    i_orb.get_grad(i_coord) * j_orb.get_value(j_coord)
            print('returning',val, i_coord, j_coord)
            return val


    def do_j(self):
        """Calculate current density"""

        #UGLY UNEFFICIENT: ONLY FOR TESTING
        
        #Cycle on all coordinates
        for ii, xx in enumerate(self._grid.get_grid()[0]):
            print("grid line", ii, " of ", len(self._grid.get_grid()[0]), xx)
            for jj, yy in enumerate(self._grid.get_grid()[1]):
                for kk, zz in enumerate(self._grid.get_grid()[2]):

                    coord = np.array([xx, yy, zz])

                    #Cycle on weight matrixes (orbitals)
                    for ii_nnz in range(self._weight.nnz):
                        i_orb = self._weight.row[ii_nnz]
                        j_orb = self._weight.col[ii_nnz]
                        #Diagonal elements give no contribution
                        if i_orb == j_orb:
                            continue
                        weight = self._weight.data[ii_nnz]
                        coord = np.array([xx,yy,zz])
                        tmp_j_vec = weight * self._do_currdensop(i_orb, j_orb, coord)
                        self._j_vec[ii, jj, kk, :] += tmp_j_vec
                        self._j_mag[ii, jj, kk] += np.linalg.norm(tmp_j_vec)
        print('Done')
        return

    def dump_to_cube(self, filename = 'jmag.cube'):
        """Write orbital to cube file """

        self._grid.dump_to_cube(self._j_mag, filename=filename, 
                header='Current Density Magnitude')
        return

