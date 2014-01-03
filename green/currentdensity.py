import numpy as np

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
                tol = 5.0 # Lateral Grid tolerance in Angstrom
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

    
    def _makegrid(self):
        """Define the rectangular grid"""
        
        x_min = self._orb_r[0,0]
        x_max = self._orb_r[0,0]
        for xx in self._orb_r[:,0]:
            if xx < x_min:
                x_min = xx
            if xx > x_max:
                x_max = xx
        y_min = self._orb_r[0,1]
        y_max = self._orb_r[0,1]
        for yy in self._orb_r[:,1]:
            if yy < y_min:
                y_min = yy
            if yy > y_max:
                y_max = yy
        z_min = self._orb_r[0,2]
        z_max = self._orb_r[0,2]
        for zz in self._orb_r[:,2]:
            if zz < z_min:
                z_min = zz
            if zz > z_max:
                z_max = zz
        x_min = x_min - self._tol
        x_max = x_max + self._tol
        y_min = y_min - self._tol
        y_max = y_max + self._tol
        z_min = z_min - self._tol
        z_max = z_max + self._tol
        print("aa", x_min, x_max, y_min, y_max, z_min, z_max)
        self._x_n = int((x_max - x_min) / self._res)
        self._x_grid = np.linspace(x_min, x_max, self._x_n)
        self._y_n = int((y_max - y_min) / self._res)
        self._y_grid = np.linspace(y_min, y_max, self._y_n)
        self._z_n = int((z_max - z_min) / self._res)
        self._z_grid = np.linspace(z_min, z_max, self._z_n)
        self._min_xyz = np.array([x_min, y_min, z_min])

        print("Number of grid points:", self._x_n * self._y_n * self._z_n)

        #Reserve dependent variables
        self._j_vec = np.zeros((self._x_n, self._y_n, self._z_n, 3))
        self._j_mag = np.zeros((self._x_n, self._y_n, self._z_n))


    def _do_currdensop(self, ii, jj, coord):
        """Calculate the current density operator between two orbitals ii and jj at a
        given coordinate"""
         
        i_orb = self._orb[self._orbind[ii]]
        j_orb = self._orb[self._orbind[jj]]
        i_coord = coord - self._orb_r[ii,:]
        j_coord = coord - self._orb_r[jj,:]

        #NOTE: imlemented for real wavefunctions
        if np.linalg.norm(i_coord) > i_orb.get_cutoff or np.linalg.norm(j_coord) > j_orb.get_cutoff():
            tmp_j_mag = 0.0
            tmp_j_vec = np.zeros(3)
            return (tmp_j_mag, tmp_j_vec)
        else:
            val = i_orb.get_value(i_coord) * j_orb.get_grad(j_coord) - \
                    i_orb.get_grad(i_coord) * j_orb.get_value(j_coord)
            return (np.linalg.norm(val), val)


    def do_j(self):
        """Calculate current density"""

        #UGLY UNEFFICIENT: ONLY FOR DEBUG
        self._makegrid()

        #Cycle on all coordinates
        for ii, xx in enumerate(self._x_grid):
            print("grid line", ii, " of ", len(self._x_grid))
            for jj, yy in enumerate(self._y_grid):
                for kk, zz in enumerate(self._z_grid):

                    coord = np.array([xx, yy, zz])

                    #Cycle on weight matrixes (orbitals)
                    for ii_nnz in range(self._weight.nnz):
                        i_orb = self._weight.row[ii_nnz]
                        j_orb = self._weight.col[ii_nnz]
                        weight = self._weight.data[ii_nnz]

                        coord = np.array([xx,yy,zz])
                        tmp_j_mag, tmp_j_vec = weight * \
                            self._do_currdensop(i_orb, j_orb, coord)

                        self._j_vec[ii, jj, kk, :] = \
                                self._j_vec[ii, jj, kk, :] + tmp_j_vec
                        self._j_mag[ii, jj, kk] = \
                                self._j_mag[ii, jj, kk] + tmp_j_mag

        print("j_mag", self._j_mag)


    def dump_to_cube(self, filename = 'jmag.cube'):
        """Write orbital to cube file """
        with open(filename, 'w') as outfile:
            #REMINDER: in cube file negative voxel length means Angstrom Units
                outfile.write('SLATER ORBITAL CUBE FILE \n')
                outfile.write('Created with PYTA \n')
                outfile.write('{} {} {} {} \n'.format(1, self._min_xyz[0],
                    self._min_xyz[1], self._min_xyz[2]))
                outfile.write('{} {} {} {}\n'.format(self._x_n, 
                    -self._res, 0.0, 0.0))
                outfile.write('{} {} {} {}\n'.format(self._y_n, 
                    0.0, -self._res, 0.0))
                outfile.write('{} {} {} {}\n'.format(self._z_n, 
                    0.0, 0.0, -self._res))
                outfile.write('{} {} {} {} {} \n'.format(1, 0.0, 0.0, 0.0, 0.0))
                for ii in range(self._x_n):
                    for jj in range(self._y_n):
                        for kk in range(self._z_n):
                            outfile.write('{}  '.format(
                                self._j_mag[ii, jj, kk]))

                             
        


