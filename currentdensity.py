import numpy as np
import pyta.grid 

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
        self._grid = pyta.grid.CubicGrid((self._orb_r, self._tol),
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


        for ii, xx in enumerate(self._grid.get_grid()[0]):
            for jj, yy in enumerate(self._grid.get_grid()[1]):
                for kk, zz in enumerate(self._grid.get_grid()[2]):
                    self._j_mag[ii, jj, kk] = np.linalg.norm(self._j_vec[ii,
                        jj, kk, :])
        print('Done')
        return
    
    def do_j_improved(self):
        """Calculate current density"""

        #FIND SOMETHING SMART HERE
        #For any nonzero density matrix couple define maximum and minimum
        #ii,jj,kk and then make a dictionary, cycle on the items and on
        #subblocks
        local_cutoff = 8.0
        for ii_nnz in range(self._weight.nnz):
            if np.mod(ii_nnz, 50) == 0:
                print('nzval', ii_nnz, ' of ', self._weight.nnz)
            weight = self._weight.data[ii_nnz]
            i_orb = self._weight.row[ii_nnz]
            j_orb = self._weight.col[ii_nnz]
            if i_orb == j_orb:
                continue
            #i_storb = self._orb[self._orbind[i_orb]]
            #j_storb = self._orb[self._orbind[j_orb]]
            #local_cutoff = i_storb.get_cutoff() + j_storb.get_cutoff()
            i_coord = self._orb_r[i_orb,:]
            j_coord = self._orb_r[j_orb,:]
            if ((abs(i_coord - j_coord) > local_cutoff).any()
                    or (abs(i_coord - j_coord) < 1e-2).all() ):
                continue
            i_min_coord = i_coord - local_cutoff#i_storb.get_cutoff()
            j_min_coord = j_coord - local_cutoff#j_storb.get_cutoff()
            i_max_coord = i_coord + local_cutoff#i_storb.get_cutoff()
            j_max_coord = j_coord + local_cutoff#j_storb.get_cutoff() 
            cpl_min_coord = np.array([max(i_min_coord[0], j_min_coord[0]),
                max(i_min_coord[1], j_min_coord[1]),  
                max(i_min_coord[2], j_min_coord[2])])
            cpl_max_coord = np.array([min(i_max_coord[0], j_max_coord[0]),
                min(i_max_coord[1], j_max_coord[1]),  
                min(i_max_coord[2], j_max_coord[2])])
            ind_min, foo = self._grid.get_grid_coord(cpl_min_coord)
            if ind_min is None:
                ind_min = np.zeros(3, dtype=int)
            ind_max, foo = self._grid.get_grid_coord(cpl_max_coord)
            if ind_max is None:
                ind_max = self._grid.get_npoints() - 1
            for ii, xx in enumerate(
                    self._grid.get_grid()[0][ind_min[0]:ind_max[0]+1]):
                for jj, yy in enumerate(
                        self._grid.get_grid()[1][ind_min[1]:ind_max[1]+1]):
                    for kk, zz in enumerate(
                            self._grid.get_grid()[2][ind_min[2]:ind_max[2]+1]):
                        
                        coord = np.array([xx, yy, zz])
                        tmp_j_vec = weight * self._do_currdensop(i_orb, j_orb, coord)
                        self._j_vec[ii + ind_min[0], jj + ind_min[1], kk +
                                ind_min[2], :] += tmp_j_vec

        for ii, xx in enumerate(self._grid.get_grid()[0]):
            for jj, yy in enumerate(self._grid.get_grid()[1]):
                for kk, zz in enumerate(self._grid.get_grid()[2]):
                    self._j_mag[ii, jj, kk] = np.linalg.norm(self._j_vec[ii,
                        jj, kk, :])
        print('Done')
        return

    def surface_flux(self, axis=2, filename = 'flux.dat'):
        j_axis = np.zeros(self._grid.get_npoints()[axis])
        if axis==2:
            for kk, zz in enumerate(self._grid.get_grid()[2]):
                for ii, xx in enumerate(self._grid.get_grid()[0]):
                    for jj, yy in enumerate(self._grid.get_grid()[1]):
                        j_axis[kk] = j_axis[kk] + self._j_vec[ii, jj, kk, 2]
         
        print('j_axis', j_axis)
        with open(filename, 'w') as outfile:
            for val in j_axis:
                outfile.write('{} \n'.format(val))
        return j_axis


    def dump_to_cube(self, var = None, filename = 'jmag.cube'):
        """Write orbital to cube file """
        if var is None:
            var = self._j_mag

        self._grid.dump_to_cube(var, filename=filename, 
                header='Current Density Magnitude')
        return

    def dump_to_vtk(self, var = None, filename = 'jmag.vtk'):
        """Write orbital to cube file """
        if var is None:
            var = self._j_mag

        self._grid.dump_to_vtk(var, filename=filename,
                                header='Current Density Magnitude')
        return

    def dump_vec_to_vtk(self, var = None, filename = 'jmag.vtk'):
        """Write orbital to cube file """
        if var is None:
            var = self._j

        self._grid.dump_vec_to_vtk(var, filename=filename,
                               header='Current Density Magnitude')
        return


    def get_j_mag(self):
        return self._j_mag

    def get_j_vec(self):
        return self._j_vec
