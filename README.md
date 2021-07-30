# Morbit

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/dev)
[![Build Status](https://github.com/manuelbb-upb/Morbit.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/Morbit.jl/actions)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Morbit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Morbit.jl)

The package `Morbit.jl` provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives.
It is meant to find a **single** Pareto-critical point, not a good covering of the global Pareto Set.

“Morbit” stands for **M**ultiobjective **O**ptimization by **R**adial **B**asis **F**unction **I**nterpolation **i**n **T**rust-regions. 
The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al.  
~~There is a [preprint in the arXiv](https://arxiv.org/abs/2102.13444) that explains what is going on inside.
It has been submitted to the MCA journal.~~  
We have a [paper](https://www.mdpi.com/2297-8747/26/2/31) explaining the algorithm!

This was my first project using Julia and there have been many messy rewrites.
Nonetheless, the solver should now work sufficiently well to tackle most problems.

This project was founded by the European Region Development Fund.
<img alt="EFRE Logo EU" src="https://www.efre.nrw.de/fileadmin/Logos/EU-Fo__rderhinweis__EFRE_/EFRE_Foerderhinweis_englisch_farbig.jpg" width=45% />
<img alt="EFRE Logo NRW" src="https://www.efre.nrw.de/fileadmin/Logos/Programm_EFRE.NRW/Ziel2NRW_RGB_1809_jpg.jpg" width=45% />

## Installation 
This package is not registered (yet), so please install via 
```
using Pkg
Pkg.add(; url = "https://github.com/manuelbb-upb/Morbit.jl.git")
# or, using ssh:
# Pkg.add(; url = "git@github.comm/manuelbb-upb/Morbit.jl.git" )
```
**Use at least version 2.1.7.**
In version 2.1.6 (and probably some minor releases prior) there is a bug 
prohibiting early stopping.
The version used for our paper was not affected.

## Quick Usage Example

Let's find a critical point for the unconstrained minimization problem with objectives
```math
f₁(x) = (x₁ - 1)² + (x₂ - 1)², f₂(x) = (x₁ + 1)² + (x₂ + 1)².
```
The critical points coincide with globally Pareto optimal points and lie on the line connecting the individual minima (1,1) and (-1,-1).

Setting up Morbit for this problem is fairly easy:
```julia
using Morbit

f1 = x -> sum( (x .- 1 ).^2 )
f2 = x -> sum( (x .+ 1 ).^2 )

mop = MixedMOP()
add_objective!(mop, f1, :cheap )
add_objective!(mop, f2, :cheap )

x0 = [ π; -ℯ ]
optimize(mop, x0)
```
In the above case, both functions are treated as cheap and their gradients are determined using `ForwardDiff`.
If you wanted to model a objective, say the function `f₂`, using radial basis functions, you could pass a `SurrogateConfig`:
```julia
rbf_cfg = RbfConfig(;kernel = :multiquadric)
add_objective!(mop, f1, rbf_cfg)
``` 

Box constraints can easily be defined at initialization of the `MixedMOP`:
```
lb = fill(-4, 2)
ub = -lb
mop_con = MixedMOP(lb, ub)
```
# Morbit

## Warning
**I am currently doing a large refactoring!**
I hope to be finished at the end of ~~July~~ August (2021), after my 3 week vacation. 
In my opinion, the package will be much better afterwards, but there will be ~~some~~ many breaking changes.
Up until then, you can still use version 2.1.8, but keep in mind, that some scripts won't work anymore once the next version releases (and I found some bugs that I won't fix for the 2.1.X versions). 
I have also made it so that any new documentation is hosted here: [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/dev)
* There you can see the re-implementation of the RBFModels and
* a notebook on how I plan to redo the Taylor Models.
* Lagrange models will be redone soon.

## README
[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/dev)
[![Build Status](https://github.com/manuelbb-upb/Morbit.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/Morbit.jl/actions)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Morbit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Morbit.jl)

The package `Morbit.jl` provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives.
It is meant to find a **single** Pareto-critical point, not a good covering of the global Pareto Set.

“Morbit” stands for **M**ultiobjective **O**ptimization by **R**adial **B**asis **F**unction **I**nterpolation **i**n **T**rust-regions. 
The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al.  
~~There is a [preprint in the arXiv](https://arxiv.org/abs/2102.13444) that explains what is going on inside.
It has been submitted to the MCA journal.~~  
We have a [paper](https://www.mdpi.com/2297-8747/26/2/31) explaining the algorithm!

This was my first project using Julia and there have been many messy rewrites.
Nonetheless, the solver should now work sufficiently well to tackle most problems.

This project was founded by the European Region Development Fund.
<img alt="EFRE Logo EU" src="https://www.efre.nrw.de/fileadmin/Logos/EU-Fo__rderhinweis__EFRE_/EFRE_Foerderhinweis_englisch_farbig.jpg" width=45% />
<img alt="EFRE Logo NRW" src="https://www.efre.nrw.de/fileadmin/Logos/Programm_EFRE.NRW/Ziel2NRW_RGB_1809_jpg.jpg" width=45% />

## Installation 
This package is not registered (yet), so please install via 
```
using Pkg
Pkg.add(; url = "https://github.com/manuelbb-upb/Morbit.jl.git")
# or, using ssh:
# Pkg.add(; url = "git@github.comm/manuelbb-upb/Morbit.jl.git" )
```
**Use at least version 2.1.7.**
In version 2.1.6 (and probably some minor releases prior) there is a bug 
prohibiting early stopping.
The version used for our paper was not affected.

## Quick Usage Example

Let's find a critical point for the unconstrained minimization problem with objectives
```math
f₁(x) = (x₁ - 1)² + (x₂ - 1)², f₂(x) = (x₁ + 1)² + (x₂ + 1)².
```
The critical points coincide with globally Pareto optimal points and lie on the line connecting the individual minima (1,1) and (-1,-1).

Setting up Morbit for this problem is fairly easy:
```julia
using Morbit

f1 = x -> sum( (x .- 1 ).^2 )
f2 = x -> sum( (x .+ 1 ).^2 )

mop = MixedMOP()
add_objective!(mop, f1, :cheap )
add_objective!(mop, f2, :cheap )

x0 = [ π; -ℯ ]
optimize(mop, x0)
```
In the above case, both functions are treated as cheap and their gradients are determined using `ForwardDiff`.
If you wanted to model a objective, say the function `f₂`, using radial basis functions, you could pass a `SurrogateConfig`:
```julia
rbf_cfg = RbfConfig(;kernel = :multiquadric)
add_objective!(mop, f1, rbf_cfg)
``` 

Box constraints can easily be defined at initialization of the `MixedMOP`:
```
lb = fill(-4, 2)
ub = -lb
mop_con = MixedMOP(lb, ub)
```

There are many options to configure both the algorithm behavior and the 
surrogate modelling techniques.
Please see the [docs](https://manuelbb-upb.github.io/Morbit.jl/dev).

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
  They need re-implementation. Directed search cannot be guaranteed to converge to critical points and for CG we need the strong Wolfe conditions.*
* Objectives can be defined with parallelism in mind, i.e., they then receive sampling sites in batches when possible.

## ToDo's

* Provide more examples.
* Finish the MathOptInterface. `AbstractMOP` already is `MOI.ModelLike`, but for the solver we wait on [this issue](https://github.com/jump-dev/JuMP.jl/issues/2099).
* Re-enable the sampling from [PointSampler.jl](https://github.com/manuelbb-upb/PointSampler.jl) for surrogate construction.
* Saving of results and evaluation data needs re-implementation.
* Maybe provide some plotting recipes?
  




There are many options to configure both the algorithm behavior and the 
surrogate modelling techniques.
Please see the [docs](https://manuelbb-upb.github.io/Morbit.jl/dev).

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
  They need re-implementation. Directed search cannot be guaranteed to converge to critical points and for CG we need the strong Wolfe conditions.*
* Objectives can be defined with parallelism in mind, i.e., they then receive sampling sites in batches when possible.

## ToDo's

* Provide more examples.
* Finish the MathOptInterface. `AbstractMOP` already is `MOI.ModelLike`, but for the solver we wait on [this issue](https://github.com/jump-dev/JuMP.jl/issues/2099).
* Re-enable the sampling from [PointSampler.jl](https://github.com/manuelbb-upb/PointSampler.jl) for surrogate construction.
* Saving of results and evaluation data needs re-implementation.
* Maybe provide some plotting recipes?
  
