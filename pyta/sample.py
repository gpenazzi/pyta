"""
Implementation of a minimal example of interoperating solvers,
for illustration of the logic
"""

import solver


class OhmLawSolver(solver.Solver):
    """

    This class evaluates the potential given an input current
    and serie resistances provided by external solvers

    invar:
    1) resistance_solvers
        list of solver providing serie resistance
    2) current
        input current
    """

    def __init__(self):
        super().__init__()
        self.resistance_solvers = list()
        self.current = None
        self.potential = None

    def _do_potential(self):

        res = 0.0
        for resistance_solver in self.resistance_solvers:
            res += resistance_solver.get('resistance')
        if self.current is None:
            raise ValueError('Current has not been set')
        self.potential = self.current * res


class ResistanceSolver(solver.Solver):
    """
    Dummy class which assigns a fixed resistance to the output variable,
    given as input parameter
    """

    def __init__(self, in_resistance=0.0):
        super().__init__()
        self.in_resistance = in_resistance
        self.resistance = None

    def _do_resistance(self):

        self.resistance = self.in_resistance


def main():
    """
    Test sample implementation
    """
    res_solv1 = ResistanceSolver(in_resistance=1.0)
    res_solv2 = ResistanceSolver(in_resistance=2.0)
    resistance_solvers = [res_solv1, res_solv2]
    ohm = OhmLawSolver()
    ohm.set('resistance_solvers', resistance_solvers)
    ohm.set('current', 1.0)
    potential = ohm.get('potential')
    print('potential is {}'.format(potential))

if __name__ == "__main__":
    main()