import numpy as np

"""A module to build slater type orbitals and perform related operations"""

#Dictionary of coefficients for DFTB sets
#key: (Type, ll, <coefficient set>)
# where coefficient set is either "pow" or "exp"
mio_0_1_so = dict()

mio_0_1_so[('H', 0, "exp")] = np.array([5.0e-01, 1.0e+00, 2.0e+00])
mio_0_1_so[('H', 0, "pow")] = np.ndarray(shape=(3,3))
mio_0_1_so[('H', 0, "pow")][0,:] = np.array(
        [-2.276520915935000e+00, 2.664106182380000e-01, -7.942749361803000e-03])
mio_0_1_so[('H', 0, "pow")][1,:] = np.array(
    [1.745369301500000e+01, -5.422967929262000e+00, 9.637082466960000e-01])
mio_0_1_so[('H', 0, "pow")][2,:] = np.array(
    [-1.270143472317000e+01, -6.556866359468000e+00, -8.530648663672999e-01])

mio_0_1_so[('C', 0, "exp")] = np.array([5.0e-01, 1.14e+00, 2.62e+00, 6.0e+00])
mio_0_1_so[('C', 0, "pow")] = np.ndarray(shape=(4,3))
mio_0_1_so[('C', 0, "pow")][0,:] = np.array(
    [-5.171232639696000e-01, 6.773263954720000e-02, -2.225281827092000e-03])
mio_0_1_so[('C', 0, "pow")][1,:] = np.array(
    [1.308444510734000e+01, -5.212739736338000e+00, 7.538242674175000e-01])
mio_0_1_so[('C', 0, "pow")][2,:] = np.array(
    [-1.215154761544000e+01, -9.329029568076001e+00, -2.006616061528000e-02])
mio_0_1_so[('C', 0, "pow")][3,:] = np.array(
    [-7.500610238649000e+00, -4.778512145112000e+00, -6.236333225369000e+00])

mio_0_1_so[('C', 1, "exp")] = np.array([5.0e-01, 1.14e+00, 2.62e+00, 6.0e+00])
mio_0_1_so[('C', 1, "pow")] = np.ndarray(shape=(4,3))
mio_0_1_so[('C', 1, "pow")][0,:] = np.array(
    [-2.302004373076000e-02, 2.865521221155000e-03, -8.868108742828000e-05])
mio_0_1_so[('C', 1, "pow")][1,:] = np.array(
    [3.228406687797000e-01, -1.994592260910000e-01, 3.517324557778000e-02])
mio_0_1_so[('C', 1, "pow")][2,:] = np.array(
    [1.328563289838000e+01, -7.908233500176000e+00, 6.945422441225000e+00])
mio_0_1_so[('C', 1, "pow")][3,:] = np.array(
    [-5.876689745586000e+00, -1.246833563825000e+01, -2.019487289358000e+01])



def realtessy(ll, mm, coord, origin, near_cutoff = 1e-2):
    """Calculate the value of a Real Tesseral harmonic in a give point coord"""

    value = 0.0
    rr = np.linalg.norm(coord - origin)
    xx = coord[0] - origin[0]
    yy = coord[1] - origin[1]
    zz = coord[2] - origin[2]

    
    if ll == 0:
        value = 0.2820947917738782
    elif ll == 1:
        if rr < near_cutoff:
            rr = near_cutoff
        if mm == -1:
            value = 0.4886025119029198 * yy / rr
        if mm == 0:
            value = 0.4886025119029198 * zz / rr
        if mm == 1:
            value = 0.4886025119029198 * xx / rr

    return value


def grad_realtessy(ll, mm, coord, origin, near_cutoff = 1e-2):
    """Calculate the value of the gradient of a Real 
    Tesseral harmonic in a give point coord"""

    value = np.zeros(3)
    rr = np.linalg.norm(coord - origin)
    xx = coord[0] - origin[0]
    yy = coord[1] - origin[1]
    zz = coord[2] - origin[2]

    if ll == 0:
        value = np.array([0.0, 0.0, 0.0])
    elif ll == 1:
        if rr < near_cutoff:
            rr = near_cutoff
        if mm == -1:
            value[0] = -0.4886025119029198 * (xx * yy / (rr**3.0))
            value[1] = 0.4886025119029198 * (xx**2.0 + zz**2.0) / (rr**3.0)
            value[2] = -0.4886025119029198 * (zz * yy / (rr**3.0))
        if mm == 0:
            value[0] = -0.4886025119029198 * (xx * zz / (rr**3.0))
            value[1] = -0.4886025119029198 * (yy * zz / (rr**3.0))
            value[2] = 0.4886025119029198 * (xx**2.0 + yy**2.0) / (rr**3.0)
        if mm == 1:
            value[0] = 0.4886025119029198 * (yy**2.0 + zz**2.0) / (rr**3.0)
            value[1] = -0.4886025119029198 * (xx * yy / (rr**3.0))
            value[2] = -0.4886025119029198 * (xx * zz / (rr**3.0))
    
    return value


class RadialFunction:
    """A class storing data far the radial function of the slater orbital"""
    def __init__(self, ll, origin, exp_coeff, pow_coeff, cutoff = 5.0):
        """ ll: angular quantum number
            coord: where the radial function is centered
            exp_coeff: array with exponential coefficients
            pow_coeff: 2D array with power coefficients [exp_ind, i]
            cutoff: after this radial value the radial function is considered to
            be 0.0
            """
        self._ll = ll
        self._origin = origin
        self._exp_coeff = exp_coeff
        self._pow_coeff = pow_coeff
        self._cutoff = cutoff

        assert(self._pow_coeff.shape[0] == self._exp_coeff.size)

    
    def get_value(self, coord):
        """ Calculate the value of the radial function on a given point.
        
        coord: if a single double, meant as rr. If an array, meant as
        cartesian coordinates """

        if type(coord) == float:
            rr = coord
        elif type(coord) == np.array or type(coord) == np.ndarray:
            rr = np.linalg.norm(coord - self._origin)
        else:
            raise TypeError(str(type(coord)))

        if rr > self._cutoff:
            return 0.0

        ll = self._ll
        val = 0.0
        for exp_ind, exp_val in enumerate(self._exp_coeff):
            for pow_ind, pow_val in enumerate(self._pow_coeff[exp_ind,:]):
                val += pow_val * rr**(ll + pow_ind) * np.exp(-1.0 *
                        exp_val * rr)
        return val

    def get_grad(self, coord):
        """ Calculate the value of the gradient of the radial function on a 
        given point.
         """

        assert type(coord) == np.array or type(coord) == np.ndarray
        rr = np.linalg.norm(coord - self._origin)
        if rr > self._cutoff:
            return 0.0


        ll = self._ll
        val = np.zeros(3)

        #Radial derivative (angular derivatives are zero)
        rad_d = 0.0
        for exp_ind, exp_val in enumerate(self._exp_coeff):
            for pow_ind, pow_val in enumerate(self._pow_coeff[exp_ind,:]):
                rad_d += (pow_val * rr**(ll + pow_ind) * np.exp(-1.0 *
                        exp_val * rr)) * ((ll + pow_ind) / rr - exp_val)


        #Radial gradient to cartesian gradient
        teta = np.arccos((coord[2] - self._origin[2]) / rr)
        if np.isnan(teta): 
            teta = 0.0
        psi = np.arctan((coord[1] - self._origin[1]) / (coord[0] -
            self._origin[0]))
        if np.isnan(psi):
            psi = 0.0

        val[0] = rad_d * np.sin(teta) * np.cos(psi)
        val[1] = rad_d * np.sin(teta) * np.sin(psi)
        val[2] = rad_d * np.cos(teta)

        return val



class SlaterType:
    """A class to build slater type orbitals and perform related operations"""
    def __init__(self, ll, mm, exp_coeff, pow_coeff, res = 0.1, cutoff = 3.0):
        """ ll: angular quantum number
            coord: where the radial function is centered
            exp_coeff: array with exponential coefficients
            pow_coeff: 2D array with power coefficients [exp_ind, i]
            cutoff: after this radial value the radial function is considered to
            be 0.0
            """
        self._ll = ll
        self._mm = mm
        self._cutoff = cutoff

        # Orbital and gradient are always built centered on zero. The methods
        # to retrieve values take care of translation

        self._origin = np.zeros(3)

        assert pow_coeff.shape[0] == exp_coeff.size

        self._radfunc = RadialFunction(ll, self._origin, exp_coeff, pow_coeff, cutoff
                = 3.0)

        self._npoints = int(cutoff / res * 2)
        #I specify only a coordinate because the grid is cubic, y_grid and
        #z_grid are identical
        self._x_grid = np.linspace(-cutoff, cutoff, self._npoints)
        self._res = res

        self._so_grid = np.zeros((self._npoints, self._npoints, self._npoints))
        self._sograd_grid = np.zeros((self._npoints, self._npoints,
            self._npoints, 3))


    def do_cache_grid(self, save=None, load=None):
        """Build a cached version of the real space orbital and its gradient.
        Save and load must be a 2-pla with two filenames, one for orbital and
        one for gradient.
        Cached values are always calculated with the origin in zero, to reuse
        grid from same orbitals centered on different origins. 
        The correct translation are taken into account in get methods."""

        print("Caching orbital", self._ll, self._mm)

        if load is None:
            print("Loading cached orbitals")
        else:
            self._so_grid = np.load(load[0])
            self._sograd_grid = np.load(load[1])
            return 0

        zero_origin = np.zeros(3)

        for i in range(self._npoints):
            for j in range(self._npoints):
                for k in range(self._npoints):
                    coord = np.array([self._x_grid[i], self._x_grid[j],
                        self._x_grid[k]]) 
                    self._so_grid[i,j,k] = self._radfunc.get_value(coord) * \
                        realtessy(self._ll, self._mm, coord, zero_origin)

                    self._sograd_grid[i,j,k,:] = \
                        self._radfunc.get_grad(coord) * \
                        realtessy(self._ll, 
                                self._mm, coord, zero_origin) + \
                        self._radfunc.get_value(coord) * \
                        grad_realtessy(self._ll, self._mm, 
                                coord, zero_origin) 

        if save is None:
            return 0
        else:
            self._so_grid.dump(save[0])
            self._sograd_grid.dump(save[1]) 
            return 0

        print("Done")
    
    def _get_grid_coord(self, coord):
        """Gives the i,j,k coordinates on the grid for a given real space
        coordinate (round down). [-1, -1, -1] means that we are out of 
        cutoff.
        
        Return the grid coordinates and the distance between the exact point and
    coord."""

        cutoff = self._cutoff
        if (coord[0] < -cutoff or coord[1] < -cutoff or coord[2] < -cutoff
                or coord[0] > cutoff or coord[1] > cutoff or
                coord[2] > cutoff):

            return (np.array([-1, -1, -1], dtype=int), 0.0)

        else:

            shift_coord = coord + np.array([cutoff, cutoff, cutoff])
            grid_coord = np.array(np.trunc(shift_coord / self._res), dtype=int)
            dist = coord - grid_coord * self._res

            return (grid_coord, dist)


    def get_value(self, coord):
        """Get the orbital value at a given point coord"""

        cutoff = self._cutoff
        if (coord[0] < -cutoff or coord[1] < -cutoff or coord[2] < -cutoff
                or coord[0] > cutoff or coord[1] > cutoff or
                coord[2] > cutoff):

            return 0.0

        grid_coord, dist = self._get_grid_coord(coord - self._origin)
        #JUST TO BE EASY I NOW RETURN THE LEFT VALUE BUT I'LL NEED TO
        #INTERPOLATE
        val = self._so_grid[tuple(grid_coord)]

        return val


    def get_grad(self, coord):
        """Get the orbital value at a given point coord"""

        cutoff = self._cutoff
        if (coord[0] < -cutoff or coord[1] < -cutoff or coord[2] < -cutoff
                or coord[0] > cutoff or coord[1] > cutoff or
                coord[2] > cutoff):

            return 0.0

        grid_coord, dist = self._get_grid_coord(coord - self._origin)
        #JUST TO BE EASY I NOW RETURN THE LEFT VALUE BUT I'LL NEED TO
        #INTERPOLATE
        val = self._sograd_grid[tuple(grid_coord)]

        return val


    def dump_to_cube(self, filename = 'orb.cube'):
        """Write orbital to cube file """
        with open(filename, 'w') as outfile:
                outfile.write('SLATER ORBITAL CUBE FILE \n')
                outfile.write('Created with PYTA \n')
                outfile.write('{} {} {} {} \n'.format(1, self._origin[0],
                    self._origin[1], self._origin[2]))
                outfile.write('{} {} {} {}\n'.format(self._npoints, 
                    self._res, 0.0, 0.0))
                outfile.write('{} {} {} {}\n'.format(self._npoints, 
                    0.0, self._res, 0.0))
                outfile.write('{} {} {} {}\n'.format(self._npoints, 
                    0.0, 0.0, self._res))
                outfile.write('{} {} {} {} {} \n'.format(1, 0.0, 0.0, 0.0, 0.0))
                for ii in range(self._npoints):
                    for jj in range(self._npoints):
                        for kk in range(self._npoints):
                            outfile.write('{}  '.format(
                                self._so_grid[ii, jj, kk]))
                

                
        
