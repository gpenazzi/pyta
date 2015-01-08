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
Pyta relies heavily on object oriented programming, with a philosophy close to top-down dynamic programming.
The main goal of Pyta is helping the user to avoid the pain of refresh/rerun specific parts of calculation when some input variables is changed. To achieve this goal, a specific set of operations (some kind of 'model' if you like) are wrapped in classes, called Solvers. Every solver apply a JIT (Just In Time) execution phylosophy, i.e. a calculation is run when is needed to provide a specific output quantity. Some quantities are always calculated on the fly (typically post-processing quantities) while some other are stored unless the Solver input doesn''t make them obsolete. In this way it 'should' (see below) e ensured that the output is usually consistent with the input. 

In order to achieve this result, we need to heavily rely on data encapsulation, i.e. everything goes through setters and getters. I do realize that this is not really pythonic, and to some extent properties may be used succesfully. However, I don''t like the idea of breaking consistency between getter which accept arguments (only implemented as method) and getters which don''t (which may be implemented as properties). However I admit that I am not still sure about this point, therefore this may change in the near future.

Of course JIT execution may impact performance, therefore it should be clear that Pyta is born for algorithm prototyping, not as production code. 

Every Solver is derived by the base class, which provides two fundamental acces point: set() and get()
Intuitively, set() is used to pass input information, and get() to retrieve output. 
The base class implements a general set/get method which works in most cases. set/get are driven by specifying the appropriate class member names. Example

> stress=solver.get('stress')  #Return the quantity stress from the solver
> solver.set('stiffness_constants', some_matrix)  #Set stiffness constants

If you are not a fan of this JIT philosophy, you can still use the classes in a procedural way: the operations to be run are set in a quite strict way so that you could reimplement some steps. 



Is the coding philosophy respected all over?
=============================
No, it is not. Of course the separation between a Solver and a normal function is arbitrary. Moreover, the design principle behind the JIT philosophy may be too strict and do not always work. An example is mixer or SCC loops, which needs to be explicitely invoked (for example, Born Approximation).

Up to now I am allowing objects as input types, therefore it is clear that this may give two problems:

1) We can end up in some recursive definition. Avoiding it is demanded to the developer. 
2) The consistency between input and output can not be always ensured because the output of a solver should depend on the sate of an input object. However I think that managin this point is too cumbersome, therefore sometimes you may still need some manual refresh (again, an example in the current implementation is electron phonon self energy)
 

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
