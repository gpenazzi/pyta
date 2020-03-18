import collections

class Solver(object):
    """
    Base class for solvers in pyta.
    Since the modifications in the API this is a sort of vitual class
    to describe methods which need to be implemented in derived classes

    Every Solver is characterized by:

    parameters
    immutable during the instance lifetime, initialized during construction.
    They can be retrieved but should not be set

    input variables:
    mutable during the instance lifetime. The are set with set() method.
    Input variables may be class instances. A set operation may propagate
    to other input variables (for example, if you set a temperature and
    an input variable is a solver depending on temperature itself)

    output variable:
    mutable during the instance lifetime. They are retrieved using get()
    methods

    Parameters, input variables and output variables are specified
    in the docstring of the class


    """

    def __init__(self):
        pass

    def reset(self):
        raise NotImplementedError()
