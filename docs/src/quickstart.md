
# Quick Start
Below are the examples from the README file:

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
```julia
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
