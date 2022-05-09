# Morbit

[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://manuelbb-upb.github.io/Morbit.jl/dev)
[![Build Status](https://github.com/manuelbb-upb/Morbit.jl/workflows/CI/badge.svg)](https://github.com/manuelbb-upb/Morbit.jl/actions)
[![Coverage](https://codecov.io/gh/manuelbb-upb/Morbit.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/manuelbb-upb/Morbit.jl)


The package `Morbit.jl` provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives.
It is meant to find a **single** Pareto-critical point.

“Morbit” stands for **M**ultiobjective **O**ptimization by **R**adial **B**asis **F**unction **I**nterpolation **i**n **T**rust-regions. 
The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al. 
We have a [paper](https://www.mdpi.com/2297-8747/26/2/31) explaining the algorithm!

This was my first project when I started using Julia and has since then undergone several rewrites.

This project was founded by the European Region Development Fund.
<img alt="EFRE Logo EU" src="https://www.efre.nrw.de/fileadmin/Logos/EU-Fo__rderhinweis__EFRE_/EFRE_Foerderhinweis_englisch_farbig.jpg" width=40% />
<img alt="EFRE Logo NRW" src="https://www.efre.nrw.de/fileadmin/Logos/Programm_EFRE.NRW/Ziel2NRW_RGB_1809_jpg.jpg" width=40% />

## New Features in Version 3.1+

Constraints :)
* Box constraints are supported natively and respected during model construction.
* Relaxable linear constraints are supported natively, i.e., propagated to the internal solver.
* Relaxable nonlinear constraints are supported via a filter mechanism.

## Installation 
This package is not registered (yet), so please install via 
```
using Pkg
Pkg.add(; url = "https://github.com/manuelbb-upb/Morbit.jl.git")
# or, using ssh:
# Pkg.add(; url = "git@github.comm/manuelbb-upb/Morbit.jl.git" )
```

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

mop = MOP(2) # problem in 2 variables
add_exact_objective!(mop, f1)
add_exact_objective!(mop, f2)

x0 = [ π; -ℯ ]
x, fx, ret_code, database = optimize(mop, x0)
```
The optimize method accepts either an `AlgorithmConfig` object via the 
`algo_config` keyword argument or concrete settings as keyword arguments.
E.g., 
```
x, fx, ret_code, database = optimize(mop, x0; max_iter=20, fx_tol_rel=1e-3)
```
sets two stopping criteria.

In the above case, both functions are treated as cheap and their gradients are determined using `FiniteDiff`.
To use automatic differentiation (via `ForwardDiff.jl`), use 
```julia
add_objective!(mop, f1; 
  n_out=1, model_cfg=ExactConfig(), 
  diff_method=Morbit.AutoDiffWrapper)
```
Gradients can be provided with the `gradients` keyword argument.

If you wanted to model a objective, say the function `f₂`, using radial basis functions, you could pass a `SurrogateConfig`:
```julia
rbf_cfg = RbfConfig(;kernel = :multiquadric)
add_objective!(mop, f1; n_out = 1, model_cfg = rbf_cfg)
```
Alternatively, there is 
```julia
add_rbf_objective!(mop, f1)
```
for scalar objective functions using the default configuration `RbfConfig()`.

Of course, vector-valued objectives are also supported:
```julia
F = x -> [f1(x); f2(x)]
add_rbf_objectives!(mop, F; n_out = 2)
# or 
# add_objective!(mop, F; n_out = 2, model_cfg = RbfConfig())
```

Instead of RBF models, Lagrange models (`LagrangeConfig()`) and Taylor polynomials (`TaylorConfig()`) are also supported.

Box constraints can easily be defined at initialization of the `MOP`:
```
lb = fill(-4, 2)
ub = -lb
mop_con = MOP(lb, ub)
```

Linear constraints of the form `A * x <= b` or `A * x == b` can be added via 
```julia
add_eq_constraint!(mop, A, b)
add_ineq_constraint!(mop, A, b)
```

Nonlinear constraints `g(x) <= 0` or `h(x) == 0` are added like the objectives:
```julia
add_nl_ineq_constraint!(mop, g; n_out = 1 model_cfg = RbfConfig())
add_nl_eq_constraint!(mop, h; n_out = 1 model_cfg = TaylorConfig())
```

There are many options to configure both the algorithm behavior and the 
surrogate modelling techniques.
Please see the [docs](https://manuelbb-upb.github.io/Morbit.jl/dev).

