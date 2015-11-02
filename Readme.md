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

Pyta does not aim to be a tool used for production run (not yet), no focus on performance is given. It is rather a tool useful to implement prototypes of quantum transport algorithms and to implement them quickly. To this end, it tries to avoid the pain of refresh/rerun specific parts of calculation when some input variables is changed. In the most recent version, this is achieved by a logical separation of input variables (mutable during the lifetime of the instance), input parameters (immutable during the lifetime of the instance) and output variables. Every solver apply a JIT (Just In Time) execution phylosophy, i.e. a calculation is run when is needed to provide a specific output quantity. Therefore the output variables are encapsulated in properties, invoking the calculation when necessary, and the input variables are also encapsulated in properties, resetting the output variables when needed. The solvers may contain output methods when arguments are needed (e.g. green.transmission(leads=(initial,final)). 

In previous versions I implemented this via set/get methods, however I quickly realizaed that those are not really needed and can be easily substituted by properties, resulting in more readable code. 

In general, the idea you will not need to explicitely run the calculation (you can still do it via private methods, if you like). It is left to the internal machinery wheter the actual calculation is invoked in initialization, or just in time.



Is the coding philosophy respected all over?
=============================
Not completely. A trivial problem is that the scheme above is completely consistent if the input quantities are immutable types, so that we can be sure that once the output variables are calculated, they will stay consistent with the input ones. Even though this would enable a philosophy closer to a functional paradigma (input/output conistency independent on instance internal state), we would loose the flexibility of OO. Therefore I don't do anything to force a choice rather than another. The only thing which is assumed is a JIT logic. 

A JIT implementation is not trivial in the case of SCC loops, as it may lead to recursion and I consider it undesirable. However, there may be some temporary solution based on explicit calls rather than JIT. One example is now the Self Consitent Born Approximation, which has to be explicitely invoked (yet). 
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
