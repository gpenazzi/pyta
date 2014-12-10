import collections

class Solver(object):
    """
    Base class for solvers in pyta. Define the general infrastructure
    and external API of a solver type class

    Every Solver is characterized by:

    parameters
    immutable during the instance lifetime, initialized during construction.
    They are retrieved using get() methods.

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


    def set(self, invarname, value):
        """
        Set an input variable invar specified by a string.
        If a _set_<name> function exists in the derived class, this
        function is invoked and arguments passed, otherwise a minimal
        set function is invoked.
        """

        # If a set function exists, it is called
        function_name = 'set_' + invarname
        exist = getattr(self, function_name, None)
        if callable(exist):
            exist(value)
            return

        # If not, we try to resolve it automagically
        # If the member is not an iterable, look for
        # a member with the given name
        exist = getattr(self, invarname)
        # Default implementation for non iterable
        setattr(self, invarname,  value)
        # Call cleandep, if any
        self.cleandep(invarname)
        return


    def get(self, outvarname, **kwargs):
        """
        Get an output variable invar specified by a string.
        If a _get_<name> function exists in the derived class, this
        function is invoked and arguments passed, otherwise a minimal
        set function is invoked.
        """
        function_name = 'get_' + outvarname
        exist = getattr(self, function_name, None)
        if callable(exist):
            return exist(**kwargs)

        member_name = outvarname
        exist = getattr(self, member_name)
        if exist is None:
            #Trying to invoke a do function
            do_function = '_do_' + outvarname
            do_exist = getattr(self, do_function, None)
            if callable(do_exist):
                do_exist(**kwargs)
                return getattr(self, member_name)
            else:
                raise ValueError('invar does not correspond to any member')
        else:
            return exist


    def cleandep(self, invarname):
        """
        Clean up dependencies relying on a given input variable.
        If no cleandep action is specified in the derived class,
        it doesn't do anything
        """
        function_name = 'cleandep_' + invarname
        try:
            exist = getattr(self, function_name)
            exist()
        except AttributeError:
            pass
        #if callable(exist):
        #    exist()
        #    return
        #else:
        #    errorstring = 'Could not find cleandep method for variable' + str(invarname)
        #    raise ValueError(errorstring)

