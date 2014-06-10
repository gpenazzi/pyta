class Solver:
    """
    Base class for solvers in pyta. Define the general infrastructure
    and external API of a solver type class
    """
    def __init__(self):
        pass


    def set(self, invarname, value, mode='replace'):
        """
        Set an input variable invar specified by a string.
        If a _set_<name> function exists in the derived class, this
        function is invoked and arguments passed, otherwise a minimal
        set function is invoked.
        """

        function_name = 'set_' + invarname
        exist = getattr(self, function_name)
        if callable(exist):
            exist(value, mode)
            return

        member_name = invarname
        exist = getattr(self, member_name)
        if mode == 'replace':
            exist = value
            return
        if mode == 'sum':
            exist += value
            return
        if mode == 'append':
            exist.append(value)
            return

        return


    def get(self, outvarname, **kwargs):
        """
        Get an output variable invar specified by a string.
        If a _get_<name> function exists in the derived class, this
        function is invoked and arguments passed, otherwise a minimal
        set function is invoked.
        """
        function_name = 'get_' + outvarname
        exist = getattr(self, function_name, **kwargs)
        if callable(exist):
            return exist(**kwargs)

        member_name = outvarname
        exist = getattr(self, member_name)
        if exist is None:
            #Trying to invoke a do function
            do_function = '_do_' + outvarname
            do_exist = getattr(self, do_function, **kwargs)
            if callable(do_exist):
                do_exist(**kwargs)
                return exist
            else:
                raise ValueError('invar does not correspond to any member')
        else:
            return exist


    def cleandep(selfself, invarname):
        """
        Clean up dependencies relying on a given input variable.
        If no cleandep action is specified in the derived class,
        it doesn't do anything
        """
        function_name = 'cleandep_' + outvarname
        exist = getattr(self, function_name, None)
        if callable(exist):
            exist()
            return

