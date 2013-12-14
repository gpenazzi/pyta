import numpy as np

"""A module to build slater type orbitals and perform related operations"""

def realtessy(ll, mm, coord, origin, near_cutoff = 1e-2):
    """Calculate the value of a Real Tesseral harmonic in a give point coord"""

    value = 0.0
    rr = np.linalg.norm(coord - origin)
    xx = coord[0] - origin
    yy = coord[1] - origin
    zz = coord[2] - origin

    
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
    xx = coord[0] - origin
    yy = coord[1] - origin
    zz = coord[2] - origin

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
    def __init__(self, ll, origin, exp_coeff, pow_coeff, cutoff = 3.0):
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



class Storbital:
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


    def do_cache_grid(self):
        """Build a cached version of the real space orbital and its gradient"""

        for i in range(self._npoints):
            for j in range(self._npoints):
                for k in range(self._npoints):
                    coord = np.array([self._x_grid[i], self._x_grid[j],
                        self._x_grid[k]]) 
                    self._so_grid[i,j,k] = self._radfunc.get_value(coord) * \
                        realtessy(self._ll, self._mm, coord, self._origin)

                    self._sograd_grid[i,j,k,:] = \
                        self._radfunc.get_grad(coord) * \
                        realtessy(self._ll, 
                                self._mm, coord, self._origin) + \
                        self._radfunc.get_value(coord) * \
                        grad_realtessy(self._ll, self._mm, 
                                coord, self._origin) 

    
    def _get_grid_coord(self, coord):
        """Gives the i,j,k coordinates on the grid for a given real space
        coordinate (round down). [-1, -1, -1] means that we are out of 
        cutoff.
        
        Return the grid coordinates and the distance between the exact point and
    coord."""

        cutoff = self._cutoff
        if (coord[0] < cutoff or coord[1] < cutoff or coord[2] < cutoff
                or coord[0] > cutoff or coord[1] > cutoff or
                coord[2] > cutoff):

            return np.aray([-1, -1, -1])

        else:

            shift_coord = coord + np.array([cutoff, cutoff, cutoff])
            grid_coord = np.ceil(shift_coord / self._res)
            dist = coord - grid_coord * self._res

            return (grid_coord, dist)



    def get_value(self, coord):
        """Get the orbital value at a given point coord"""

        cutoff = self._cutoff
        if (coord[0] < -cutoff or coord[1] < -cutoff or coord[2] < -cutoff
                or coord[0] > cutoff or coord[1] > cutoff or
                coord[2] > cutoff):

            return 0.0

        (grid_coord, dist) = self._get_grid_coord(coord)
        #JUST TO BE EASY I NOW RETURN THE LEFT VALUE BUT I'LL NEED TO
        #INTERPOLATE
        val = self._so_grid[grid_coord]

        return val


    def get_grad(self, coord):
        """Get the orbital value at a given point coord"""

        cutoff = self._cutoff
        if (coord[0] < -cutoff or coord[1] < -cutoff or coord[2] < -cutoff
                or coord[0] > cutoff or coord[1] > cutoff or
                coord[2] > cutoff):

            return 0.0

        (grid_coord, dist) = self._get_grid_coord(coord)
        #JUST TO BE EASY I NOW RETURN THE LEFT VALUE BUT I'LL NEED TO
        #INTERPOLATE
        val = self._sograd_grid[grid_coord]

        return val


        
