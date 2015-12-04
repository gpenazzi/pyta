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

To this end some kind of encapsulation is needed. I achieve the result with properties rather than getter/setter (even though I previously did it in this way). I found drawbacks and advantages in both methods, but the main reason I switched is that I find the syntax more elegant and in general properties substitute the getter/setter without relevant problems.    

In general, the idea you will not need to explicitly run the calculation (you
can still do it via private methods, if you like). It is left to the internal machinery whether the actual calculation is invoked in initialization, or just
in time.

In a similar spirit, a solver has 3 different kind of members: parameters,
input variables and output variables. Parameters are supposed not to change
during the lifetime of a solver, input variables may change and output parameters may change (what a surprise!). Input and output variables are encapsulated; when an input variable is assigned the solver should take care to reset (set to None) all the output variables which depend upon it.

As an example, in a Green function solver setting a different temperature should leave untouched the equilibrium Green's function and reset the Keldysh Green's function.



Is the coding philosophy respected all over?
=============================
Not completely. A trivial problem is that the scheme above is completely consistent if the input quantities are immutable types, so that we can be sure that once the output variables are calculated, they will stay consistent with the input ones. Even though this would enable a philosophy closer to a functional paradigma (input/output conistency independent on instance internal state), we would loose the flexibility of OO. Therefore I don't do anything to force a choice rather than another. The only thing which is assumed is a JIC logic.

A JIC implementation is not trivial in the case of SCC loops, as it may lead to recursion and I consider it undesirable. However, there may be some temporary solution based on explicit calls rather than JIC. One example is now the Self Consitent Born Approximation, which has to be explicitely invoked (yet).
tion is electron phonon self energy)


Pyta is terrible, what else can I use?
=============================

Pyta is still a very personal toy and changes continuosly, and I am not sure whether I am working on it more for the sake of science or for the sake of experimenting software design. If you are interested in quantum transport and you are looking for some more mature tool already used to produce actual science, and still like to work with Python, you can give a look to these other packages (I am not affiliated in any way with the authors):

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
