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
Whenever possible, a Class is defined to represent a Solver, where a Solver is just a generic object which accept some inputs and parameters and knows how to generate a set of given outputs. 
Every Solver is derived by the base class, which provides two fundamental acces point: set() and get()
Intuitively, set() is used to pass input information, and get() to retrieve output.
There is no explicit solve or run commands (here comes the dynamic part). The computation routines are hidden inside a solver class and are invoked anytime a get() operation finds that the variable to be retrieved has not been defined. Hence the top-down approach: variables are 'requested' at high level and computed only when needed. The final aim is to have slim scipts capable to do complicated stuff. Performances are of course affected, therefore it is better to make clear that Pyta is not designed to be fast, it is designed to be consistent and to simplify the coding of complicated workflows.  


Is the coding philosophy respected all over?
=============================
No, it is not. Some simple analytics or utilities are written in a plain non object-oriented style, it is not worth it to try to make everything more complicated than already is. 
Also not everything can fit in such a simple functional like design. This solver structure works fine for sequential workflows but is prone to ill-defined recursive cross dependencies. This tipically happens when two solver depend on each other, a simple case is an SCC loop. Therefore I already needed to add a slightly different class to mix solvers (for the SCBA loop, in my case). These kind of class works in place, therefore they don't have a defined output. Rather, they change the state of the solvers.


Pyta is terrible, what else can I use?
=============================

Pyta is still a very personal toy and changes continuosly. If you are looking for some more mature tool already used to produce actual science, and still like to work with Python, you can give a look to these other packages (I am not affiliated in any way with the authors):

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
