import numpy as np
import pyta.grid 

"""Methods and class to build real space current density"""

au_to_nm = 5.2918e-2
au_to_ang = 5.2918e-1

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
            val = 0.5 * (i_orb.get_value(i_coord)[:,None] * j_orb.get_grad(j_coord) -
                    i_orb.get_grad(i_coord) * j_orb.get_value(j_coord)[:,None])
            return val

    def _do_overlap(self, ii, jj, coord):
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
            val = (np.conj(i_orb.get_value(i_coord)[:,None]) * j_orb.get_value(j_coord)[:,None])
            return val



    def do_j(self):
        """Calculate current density"""


        #UGLY INEFFICIENT: ONLY FOR TESTING
        
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
        #local_cutoff = 12.0

        meshgrid = self._grid.get_meshgrid()
        for ii_nnz in range(self._weight.nnz):
            if np.mod(ii_nnz, 50) == 0:
                print('nzval', ii_nnz, ' of ', self._weight.nnz)
            weight = self._weight.data[ii_nnz]
            i_orb = self._weight.row[ii_nnz]
            j_orb = self._weight.col[ii_nnz]
            if i_orb == j_orb:
                continue
            i_storb = self._orb[self._orbind[i_orb]]
            j_storb = self._orb[self._orbind[j_orb]]
            local_cutoff = (i_storb.get_cutoff() + j_storb.get_cutoff()) / 2.0
            i_coord = self._orb_r[i_orb,:]
            j_coord = self._orb_r[j_orb,:]

            #TEMPORARY: I WANT TO PLAY WITH SOME CONDITION ON DISTANCE
            # REMEMBER THAT DISTANCES ARE IN AU HERE
            #distance = np.linalg.norm(i_coord - j_coord)
            #if distance > 1.0:
            #    continue

            if ((abs(i_coord - j_coord) > local_cutoff).any()):
#                    or (abs(i_coord - j_coord) < 1e-2).all() ):
                continue
            i_min_coord = i_coord - i_storb.get_cutoff()
            j_min_coord = j_coord - j_storb.get_cutoff()
            i_max_coord = i_coord + i_storb.get_cutoff()
            j_max_coord = j_coord + j_storb.get_cutoff()
            cpl_min_coord = np.maximum(i_min_coord, j_min_coord)
            cpl_max_coord = np.minimum(i_max_coord, j_max_coord)
            ind_min = self._grid.get_grid_coord(cpl_min_coord)[0] + 1
            #if ind_min is None:
            #    ind_min = np.zeros(3, dtype=int)
            ind_max = self._grid.get_grid_coord(cpl_max_coord)[0] - 1
            #if ind_max is None:
            #    ind_max = self._grid.get_npoints() - 1

            n_points = (ind_max[0]-ind_min[0])*(ind_max[1]-ind_min[1])*(ind_max[2]-ind_min[2])
            coords = np.zeros(shape=(n_points,3))
            count = 0
            for ii, xx in enumerate(
                    self._grid.get_grid()[0][ind_min[0]:ind_max[0]]):
                for jj, yy in enumerate(
                        self._grid.get_grid()[1][ind_min[1]:ind_max[1]]):
                    for kk, zz in enumerate(
                            self._grid.get_grid()[2][ind_min[2]:ind_max[2]]):
                        coords[count, :] = [xx, yy, zz]
                        count += 1

            #NOTE: units conversion in mA/nm^2
            #Assuming to get only the current operator from the density matrix
            conversion = 0.23653e4
            tmp_j_vec = weight * self._do_currdensop(i_orb, j_orb, coords) * conversion
            self._j_vec[ind_min[0]:ind_max[0],
                        ind_min[1]:ind_max[1],
                        ind_min[2]:ind_max[2],:] += np.reshape(tmp_j_vec,
                                        ( ind_max[0]-ind_min[0],
                                          ind_max[1]-ind_min[1],
                                          ind_max[2]-ind_min[2], 3))

        for ii, xx in enumerate(self._grid.get_grid()[0]):
            for jj, yy in enumerate(self._grid.get_grid()[1]):
                for kk, zz in enumerate(self._grid.get_grid()[2]):
                    self._j_mag[ii, jj, kk] = np.linalg.norm(self._j_vec[ii,
                        jj, kk, :])
        print('Done')
        np.savez('currents', self._j_vec, self._j_mag)

        return

    def do_j_general(self):
        """
        Calculate current density with '3 center' contributions:
        J(r) = Sum_ijk G_ij <j|J|k> S_ik

        Note: this form is probably wrong. There should not be a contribution
        from a third orbital (see Lake)

        """
        pass
        #FIND SOMETHING SMART HERE
        #For any nonzero density matrix couple define maximum and minimum
        #ii,jj,kk and then make a dictionary, cycle on the items and on
        #subblocks
        #local_cutoff = 12.0

        #meshgrid = self._grid.get_meshgrid()
        #for ii_nnz in range(self._weight.nnz):
        #    if np.mod(ii_nnz, 50) == 0:
        #        print('nzval', ii_nnz, ' of ', self._weight.nnz)
        #    weight = self._weight.data[ii_nnz]
        #    i_orb = self._weight.row[ii_nnz]
        #    j_orb = self._weight.col[ii_nnz]
        #    #if i_orb == j_orb:
        #    #    continue
        #    i_storb = self._orb[self._orbind[i_orb]]
        #    j_storb = self._orb[self._orbind[j_orb]]

        #    # Look for k orbital for 3 center term
        #    for k_orb, foo in enumerate(self._orb_r):
        #        k_storb = self._orb[self._orbind[k_orb]]
        #        cutoff_jk = (k_storb.get_cutoff() + j_storb.get_cutoff()) / 2.0
        #        cutoff_ik = (i_storb.get_cutoff() + k_storb.get_cutoff()) / 2.0
        #        i_coord = self._orb_r[i_orb,:]
        #        j_coord = self._orb_r[j_orb,:]
        #        k_coord = self._orb_r[k_orb,:]
        #        if ((abs(i_coord - k_coord) > cutoff_ik).any()):
        #            continue
        #        if ((abs(j_coord - k_coord) > cutoff_jk).any()):
        #            continue

        #    #TEMPORARY: I WANT TO PLAY WITH SOME CONDITION ON DISTANCE
        #    # REMEMBER THAT DISTANCES ARE IN AU HERE
        #    #distance = np.linalg.norm(i_coord - j_coord)
        #    #if distance > 1.0:
        #    #    continue

        #        #I only calculate non-zero contributions where the
        #        #wavefunctions involved in current operator are overlapping
        #        #It can be improved by adding the condition on S_ik
        #        i_min_coord = i_coord - i_storb.get_cutoff()
        #        j_min_coord = j_coord - j_storb.get_cutoff()
        #        k_min_coord = k_coord - k_storb.get_cutoff()
        #        i_max_coord = i_coord + i_storb.get_cutoff()
        #        j_max_coord = j_coord + j_storb.get_cutoff()
        #        k_max_coord = k_coord + k_storb.get_cutoff()
        #        cpl_min_coord = np.maximum(np.maximum(k_min_coord, j_min_coord), i_min_coord)
        #        cpl_max_coord = np.minimum(np.minimum(k_max_coord, j_max_coord), i_max_coord)
        #        ind_min = self._grid.get_grid_coord(cpl_min_coord)[0] + 1
        #        #if ind_min is None:
        #        #    ind_min = np.zeros(3, dtype=int)
        #        ind_max = self._grid.get_grid_coord(cpl_max_coord)[0] - 1
        #        #if ind_max is None:
        #        #    ind_max = self._grid.get_npoints() - 1

        #        n_points = (ind_max[0]-ind_min[0])*(ind_max[1]-ind_min[1])*(ind_max[2]-ind_min[2])
        #        coords = np.zeros(shape=(n_points,3))
        #        count = 0
        #        for ii, xx in enumerate(
        #            self._grid.get_grid()[0][ind_min[0]:ind_max[0]]):
        #            for jj, yy in enumerate(
        #                self._grid.get_grid()[1][ind_min[1]:ind_max[1]]):
        #                for kk, zz in enumerate(
        #                    self._grid.get_grid()[2][ind_min[2]:ind_max[2]]):
        #                    coords[count, :] = [xx, yy, zz]
        #                    count += 1

        #        #NOTE: units conversion in mA/nm^2
        #        #Assuming to get only the current operator from the density matrix
        #        conversion = 0.23653e4
        #        tmp_j_vec = weight * self._do_currdensop(j_orb, k_orb, coords) * conversion
        #        if k_orb == i_orb:
        #            overlap = 1.0
        #        else:
        #            overlap = self._do_overlap(i_orb, k_orb, coords)
        #        tmp_j_vec = np.multiply(tmp_j_vec, overlap)
        #        self._j_vec[ind_min[0]:ind_max[0],
        #        ind_min[1]:ind_max[1],
        #        ind_min[2]:ind_max[2],:] += np.reshape(tmp_j_vec,
        #                                           ( ind_max[0]-ind_min[0],
        #                                             ind_max[1]-ind_min[1],
        #                                             ind_max[2]-ind_min[2], 3))

        #for ii, xx in enumerate(self._grid.get_grid()[0]):
        #    for jj, yy in enumerate(self._grid.get_grid()[1]):
        #        for kk, zz in enumerate(self._grid.get_grid()[2]):
        #            self._j_mag[ii, jj, kk] = np.linalg.norm(self._j_vec[ii,
        #                                                     jj, kk, :])
        #print('Done')
        #np.savez('currents', self._j_vec, self._j_mag)

        #return

    def load(self, filename):
        npzfile = np.load(filename)
        self._j_vec = npzfile['arr_0']
        self._j_mag = npzfile['arr_1']
        return

    def surface_flux(self, axis=2, filename = 'flux.dat'):
        """
        Calculate the flux integral along x axis. We assume that
        the current is evaluated in mA/nm^2 and return a value in mA
        """
        conversion = au_to_nm * au_to_nm * self._res * self._res
        j_axis = np.zeros(self._grid.get_npoints()[axis])
        if axis==2:
            for kk, zz in enumerate(self._grid.get_grid()[2]):
                for ii, xx in enumerate(self._grid.get_grid()[0]):
                    for jj, yy in enumerate(self._grid.get_grid()[1]):
                        j_axis[kk] = j_axis[kk] + self._j_vec[ii, jj, kk, 2] * conversion
         
        with open(filename, 'w') as outfile:
            for val in j_axis:
                outfile.write('{} \n'.format(val))
        return j_axis


    def dump_to_cube(self, var = None, filename = 'jmag.cube'):
        """Write orbital to cube file """
        if var is None:
            var = self._j_mag

        self._grid.dump_to_cube(var, filename=filename, 
                header='Current Density Magnitude',
                conversion = au_to_ang)
        return

    def dump_to_vtk(self, var = None, filename = 'jmag.vtk'):
        """Write orbital to cube file """
        if var is None:
            var = self._j_mag

        self._grid.dump_to_vtk(var, filename=filename,
                                header='Current Density Magnitude',
                                conversion = au_to_ang)
        return

    def dump_vec_to_vtk(self, var = None, filename = 'jmag.vtk'):
        """Write orbital to cube file """
        if var is None:
            var = self._j

        self._grid.dump_vec_to_vtk(var, filename=filename,
                               header='Current Density Magnitude',
                               conversion = au_to_ang)
        return


    def get_j_mag(self):
        return self._j_mag

    def get_j_vec(self):
        return self._j_vec
