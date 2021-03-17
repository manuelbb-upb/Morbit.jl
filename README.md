# Morbit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/dev)
[![Build Status](https://github.com/manuelbb-upb/Morbit.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/Morbit.jl/actions)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Morbit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Morbit.jl)

The package `Morbit.jl` provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives.
It is meant to find a **single** Pareto-critical point, not a good covering of the global Pareto Set.

“Morbit” stands for **M**ultiobjective **O**ptimization by **R**adial **B**asis **F**unction **I**nterpolation **i**n **T**rust-regions. 
The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al.  
There is a [preprint in the arXiv](https://arxiv.org/abs/2102.13444) that explains what is going on inside.
It has been submitted to the MCA journal.

This was my first project using Julia and there have been many messy rewrites.
Nonetheless, the solver should now work sufficiently well to tackle most problems.

For extensive usage information please refer to the [documentation](https://manuelbb-upb.github.io/Morbit.jl/dev).

## Features
* Applicable to unconstrained and finitely box-constrained problems with one or more objectives.
* Treat the objectives as exact (and benefit from automatic differencing) or use surrogate models.  
  Available surrogates are
  * First and second degree Taylor polynomials, either using exact derivatives or finite-difference approximations.
  * Fully-linear Lagrange polynomials of degree 1 or 2.
  * Fully-linear Radial basis function models with a polynomial tail of degree 1 or less.
* The surrogate construction algorithms try to avoid evaluating the true objectives unnecessarily.  
  Evaluation data is stored in a database and can be retrieved afterwards.
* At the moment, the trust region sub-problems are solved using either a multiobjective **steepest descent 
  direction** (default) or the Pascoletti-Serafini scalarization 
  (see [here](https://www.tu-ilmenau.de/fileadmin/media/mmor/thomann/SIAM_MHT_TE.pdf)).  
  *From prior experiments we know the “directed search” method and the nonlinear conjugate gradient steps to work well, too. 
  They need re-implementation. Directed search cannot be guaranteed to converge to a critical points and for CG we need the strong Wolfe conditions.*
* Objectives can be defined with parallelism in mind, i.e., they then receive sampling sites in batches when possible.

## ToDo's

* Provide more examples.
* Finish the MathOptInterface. `AbstractMOP` already is `MOI.ModelLike`, but for the solver we wait on [this issue](https://github.com/jump-dev/JuMP.jl/issues/2099).
* Re-enable the sampling from [PointSampler.jl](https://github.com/manuelbb-upb/PointSampler.jl) for surrogate construction.
* Saving of results and evaluation data needs re-implementation.
* Maybe provide some plotting recipes?
  



