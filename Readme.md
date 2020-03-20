Pyta
=========

Pyta is a small collection of Python tools for computational nanodevice modelling and materials science.
It serves a double purpose: provide a swiss knife of modules for fast scientific prototyping and analytics and a playground to experiment with different object oriented programming paradigma (which is probably the most important point to me).

Up to now the only usable part is aimed to Green's function calculations and offers the following features:

  - Equilibrium and non-equilibrium Green's function
  - Open boundary conditions
  - SCBA local self energy correction
  - Fermion and phonon implementation

You can find simple examples of usage in examples/

Coding Philosophy
=========
Pyta relies on object oriented programming, with a philosophy close to top-down dynamic programming.

Pyta does not aim to be a tool used for production run (not yet), no focus on performance is given. It is rather a tool useful to implement prototypes of quantum transport algorithms and to implement them quickly. To this end, it
tries to avoid the pain of refresh/rerun specific parts of calculation when some input variables is changed. In the most recent version, this is achieved by a logical separation of input variables (mutable during the lifetime of the instance), input parameters (immutable during the lifetime of the instance) and output variables. Every solver apply a JIC (Just In Case) execution philosophy, i.e. a calculation is run when is needed to provide a specific output quantity requested to the class, be it from another solver, from a method in the same class or from the user directly in a driving script.

To this end some kind of encapsulation is needed. Most class members are properties, i.e. assigning a value to a member might hide more complex operations or invalidate other members. Similarly retrieving a value might trigger calculations.
In this way a class should always be in a consistent internal state, e.g. if you change electrode temperature, then the occupation will be recalculated the first time it is requested.

In general, the idea you will not need to explicitly run the calculation (you
can still do it via private methods, if you like). It is left to the internal machinery whether the actual calculation is invoked in initialization, or just
in time.


Is the coding philosophy respected all over?
=============================
A JIC implementation is not trivial in the case of SCC loops, as it may lead to recursion and I consider it undesirable. However, there may be some temporary solution based on explicit calls rather than JIC. One example is now the Self Consitent Born Approximation, which has to be explicitely invoked calling an `scba` method.


Pyta is terrible, what else can I use?
=============================

Pyta is still a very personal toy, with no test coverage nor documentation and it's not a production code. Also, some choice were taken to experiment with design. If you are interested in quantum transport and you are looking for some more mature tool already used to produce actual science, and still like to work with Python, you can give a look to these other packages (I am not affiliated in any way with the authors):

http://kwant-project.org/authors

http://vides.nanotcad.com/vides/

https://wiki.fysik.dtu.dk/ase/



Version
----

0.1


License
----

Copyright Gabriele Penazzi, University of Bremen 2013

Distributed under LGPL
