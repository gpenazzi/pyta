import numpy as np
import ase

class CubicGrid:
    """A class to build and manage cubic grids and variables mapped on
    structured rectilinear grids with regularly spaced points (spacing can be
    different along x,y,z but must be constant). Corresponds to cub format and
    VTK StructuredPoints"""
    def __init__(self,
                #Contants 
                boundary, #(Rmin, Rmax) or (Atoms, tolerance) or (ndarray,
                          #tolerance)
                res #float or 3-ple
                ):
        
        #By default grids are in real space but in theory more dimensions would
        #be possible
        self._dim = 3

        #Constants
        assert type(boundary) == tuple
        assert type(res) == float or type(res) == tuple or \
                type(res) == list or type(res) == np.array

        self._rmin = None
        self._rmax = None

        #Manage override of boundaries
        if type(boundary[0]) == tuple or type(boundary[0]) == list \
                or (type(boundary[0]) == np.ndarray and boundary[0].size == 3):
                self._rmin = boundary[0]
                self._rmax = boundary[1]
        elif type(boundary[0]) == ase.atoms.Atoms or \
                type(boundary[0]) == np.ndarray:
            self._rmin, self._rmax = self._positions_to_boundary(boundary[0],
                boundary[1])

        if type(res) == float:
            self._res = np.array([res, res, res])
        else:
            self._res = res

        #Number of nodes in x,y,z direction
        self._npoints = np.zeros(self._dim, dtype = int)
        #Note: grid is a list of arrays, not a ndarray because it's not
        #ensured that the grids have the same number of nodes 
        self._grid = list()
        #Real step after discretization
        self._step = list()
        #Total number of nodes
        self._nnodes = None
        #A meshgrid 3-ple can also be built, if asked for
        self._meshgrid = None

        #Prepare the grid
        self._makegrid()

    def _positions_to_boundary(self, positions, tol):
        """Define the grid boundaries starting from positions array.
        Position array can be a Nx3 numpy.ndarray or a ase.atoms.Atoms object.
        tol describe the additional lateral spacing. If a float, same tolerance
        along 3 directions is taken. If a 3-ple, different distances can be
        specified"""
        
        if type(positions) == ase.atoms.Atoms:
            loc_pos = positions.get_positions()
        else:
            loc_pos = positions
        rmin = np.array(positions[0,:])
        rmax = np.array(positions[0,:])
        for pos in loc_pos:
            for ind, val in enumerate(pos):
                if val < rmin[ind]:
                    rmin[ind] = val
                if val > rmax[ind]:
                    rmax[ind] = val

        if type(tol) == float:
            rmin = rmin - np.array([tol, tol, tol])
            rmax = rmax + np.array([tol, tol, tol])
        else:
            rmin = rmin - tol
            rmax = rmax + tol

        return (rmin, rmax)

    def _makegrid(self):
        """Define the grid mesh"""
        
        for ind in range(self._dim):
            self._npoints[ind] = int(round((self._rmax[ind] - 
                self._rmin[ind]) / self._res[ind])) + 1
            tmp_grid, tmp_step = np.linspace(self._rmin[ind], self._rmax[ind],
                    num=self._npoints[ind], endpoint=True, retstep=True)

            self._grid.append(tmp_grid)
            self._step.append(tmp_step)
        self._nnodes = 1
        for npoints in self._npoints:
            self._nnodes *= npoints
        return

    def dump_to_cube(self, var, filename = 'tmp.cube', header = "Created with PYTA",
                     conversion = 1.0):
        """Write orbital to cube file """
        with open(filename, 'w') as outfile:
            #REMINDER: in cube file negative voxel length means Angstrom Units
                outfile.write('Pyta cube file \n')
                outfile.write(header + '\n')
                outfile.write('{} {} {} {} \n'.format(1, self._rmin[0] * conversion,
                    self._rmin[1] * conversion, self._rmin[2] * conversion))
                outfile.write('{} {} {} {}\n'.format(self._npoints[0], 
                    self._step[0] * conversion, 0.0, 0.0))
                outfile.write('{} {} {} {}\n'.format(self._npoints[1], 
                    0.0, self._step[1] * conversion, 0.0))
                outfile.write('{} {} {} {}\n'.format(self._npoints[2], 
                    0.0, 0.0, self._step[2] *  conversion))
                outfile.write('{} {} {} {} {} \n'.format(1, 0.0, 0.0, 0.0, 0.0))
                for ii in range(self._npoints[0]):
                    for jj in range(self._npoints[1]):
                        for kk in range(self._npoints[2]):
                            outfile.write('{}  '.format(
                                var[ii, jj, kk]))

    def dump_to_vtk(self, var, filename = 'tmp.cube', header = "var", conversion = 1.0):
        """Write orbital to cube file """
        with open(filename, 'w') as outfile:
            #REMINDER: in cube file negative voxel length means Angstrom Units
            outfile.write('# vtk DataFile Version 3.0 \n')
            outfile.write('vtk output \n')
            outfile.write('ASCII\n')
            outfile.write('DATASET STRUCTURED_POINTS\n')
            outfile.write('DIMENSIONS ')
            outfile.write('{} {} {}\n'.format(self._npoints[0],
                                              self._npoints[1], self._npoints[2]))
            outfile.write('SPACING ')
            outfile.write('{} {} {}\n'.format(self._step[0] * conversion,
                                              self._step[1] * conversion,
                                              self._step[2] * conversion))
            outfile.write('ORIGIN ')
            outfile.write('{} {} {}\n'.format(self._rmin[0] * conversion,
                                              self._rmin[1] * conversion,
                                              self._rmin[2] * conversion))
            outfile.write('POINT_DATA ')
            outfile.write('{}\n'.format(self._npoints[0] * self._npoints[1]
                                        * self._npoints[2]))
            outfile.write('SCALARS var%0A float\n')
            outfile.write('LOOKUP_TABLE default\n')
            for ii in range(self._npoints[2]):
                for jj in range(self._npoints[1]):
                    for kk in range(self._npoints[0]):
                        outfile.write('{}\n'.format(
                            var[kk, jj, ii]))

    def dump_vec_to_vtk(self, var, filename = 'tmp.cube', header = "var", conversion = 1.0):
        """Write orbital to cube file """
        with open(filename, 'w') as outfile:
            #REMINDER: in cube file negative voxel length means Angstrom Units
            outfile.write('# vtk DataFile Version 3.0 \n')
            outfile.write('vtk output \n')
            outfile.write('ASCII\n')
            outfile.write('DATASET UNSTRUCTURED_GRID\n')
            outfile.write('POINTS {} float\n'.format(self._nnodes))
            for ii in range(self._npoints[2]):
                for jj in range(self._npoints[1]):
                    for kk in range(self._npoints[0]):
                        coord = self.get_space_coord(np.array([kk,jj,ii])) * conversion
                        outfile.write('{} {} {}\n'.format(
                            coord[0], coord[1], coord[2]))
            outfile.write('\nCELLS {} {}\n'.format(self._nnodes, 2*self._nnodes))
            for ind in range(self._nnodes):
                outfile.write('1 {}\n'.format(ind))
            outfile.write('\nCELL_TYPES {}\n'.format(self._nnodes))
            for ind in range(self._nnodes):
                outfile.write('1\n'.format())
            outfile.write('\nPOINT_DATA {}\n'.format(self._nnodes))
            outfile.write('VECTORS local_current float\n'.format())
            for ii in range(self._npoints[2]):
                for jj in range(self._npoints[1]):
                    for kk in range(self._npoints[0]):
                        outfile.write('{} {} {}\n'.format(
                            var[kk, jj, ii, 0], var[kk, jj, ii, 1],
                            var[kk, jj, ii, 2]))



    def get_grid_coord(self, coord):
        """Gives the i,j,k coordinates on the grid for a given real space
        coordinate (round down). None means that we are out of 
        boundaries.
        
        Return the grid coordinates and the distance between the exact point and
    coord."""

        for ii in range(self._dim):
            if coord[ii] < self._rmin[ii] or coord[ii] > self._rmax[ii]:
                return (None, 0.0)

        shift_coord = coord - self._rmin
        grid_coord = np.array(np.trunc(shift_coord / self._step), dtype=int)
        dist = coord - grid_coord * self._step

        return (grid_coord, dist)

    def get_space_coord(self, coord):
        """
        Gives the real space coordinate for a given i,j,k integer array coordinate
        """
        return self._rmin + np.multiply(coord, self._res)
        

    def get_value(self, coord, var):
        """Get the value of a grid variable at a given point coord"""

        if not self.is_inside(coord):
            return None

        grid_coord, dist = self.get_grid_coord(coord)
        #JUST TO BE EASY I NOW RETURN THE LEFT VALUE BUT I'LL NEED TO
        #INTERPOLATE
        val = var[tuple(grid_coord)]

        return val
    
    def get_grid(self):
        """Get grid linear space"""

        return self._grid
                             
    def get_npoints(self):
        """Get number of points along x,y,z direction"""

        return self._npoints


    def get_meshgrid(self):
        """Get a meshgrid object"""
        if self._meshgrid == None:
            self._meshgrid = np.meshgrid(self._grid[0], self._grid[1],
                    self._grid[2], indexing='xy')
        return self._meshgrid

    def get_minmax(self):
        """Return minimum and maximum coordinates"""
        return (self._rmin, self._rmax)

    def get_nnodes(self):
        """Return the number of nodes"""
        return self._nnodes

    def is_inside(self, coord):
        """True if coord is inside the grid, False otherwise"""
        for ii in range(self._dim):
            if coord[ii] < self._rmin[0] or coord[ii] > self._rmax[ii]:
                return False
        return True
                             
