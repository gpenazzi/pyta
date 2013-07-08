import numpy as np
import pyta.core.defaults as defaults

class Lead:
    """A base class for managing and building up real and virtual leads
    (inscattering and outscattering sources)."""

    def __init__(self, name, delta = defaults.delta):
        """A name to associate to the lead is always needed (e.g. 'left',
        'phonon')"""

        self._type = None


class PhysicalLead(Lead):
    """A class derived from Lead for the description of physical contacts"""

    def __init__(self, name, h, s, delta = defaults.delta):
         """A name to associate to the lead is always needed""" 
        Lead.__init__(self, name, delta)

        assert(type(h) == numpy.matrixlib.defmatrix.matrix)
        if s:
            assert(type(s) == numpy.matrixlib.defmatrix.matrix)
        self._h = h
        self._n = len(self._h)
        if s == None:
            self._s = np.matrix(np.eye(self._n))
