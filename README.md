# Morbit

## Installation
### 0. Option
Use the [Singularity Container](https://github.com/manuelbb-upb/of_morbit_singularity).

### 1. Option
Download this repository (and extract it if you chose `.zip`ed download).
In Julia, `cd('/download/locaction/Morbit')` and then activate the Package manager via `]` to finally do
```
activate .
```

### 2. Option
In Julia, activate the Package manager via `]` and finally do
```
add https://url/to/this/repo
```
Might require credentials.

## Basic Usage
See the files in the `test` and `examples` folder.
A very basic script might look like
```julia
using Morbit  # include the module

# define objective functions ℝ² → ℝ
f1(x) = sum( (x .- 1).^2 )
f2(x) = sum( (x .+ 1).^2 )

# setup a box unconstrained problem
mop = MixedMOP( lb = -4 .* ones(2), ub = 4 .* ones(2) )   # unconstrained: mop = MixedMOP()
add_objective!(mop, f1)
add_objective!(mop, f2)

# optimize
opt_obj = AlgoConfig(
  max_iter = 10
)
x0 = [ 2.5; -3.0 ]
X,FX = optimize!( opt_obj, mop, x0 )

save_config( opt_obj, 'results.jld')  # save results
```

### Vector Format
We distinguish between (input) sites and (output) values. 
Input sites must be of type `Vector{Float64}`. 
Thus even in the 1D case you would have to provide `x0 = [ 2.0 ]` or similar.
Output values are of the same type.
When providing scalar objectives, they must return a `Float64`.
`opt_obj.iter_data.sites_db` is a `Vector{Vector{Float64}}` and its elements are all sites evaluated during optimization.
`opt_obj.iter_data.values_db` has the corresponding value vectors.

### Adding Objetives
Let `g` be a scalar-valued objective function and `G` be vector-valued. 
Both types of objectives are supported.
Additionally, you can specify the gradient function `dg` ℝ^n → ℝ^n of `g`.

* `add_objective!(mop, g, :expensive)` adds `g` as an *expensive* objective, that is internally replaced by a RBF surrogate model.
* `add_objective!(mop, g, :cheap)` allows for `g` to be automatically differentiated.
* `add_objective!(mop, g, dg)` adds `g` as a cheap objective with predefined gradient `dg`.
* `add_objective!(mop, G, :expensive, 2)` adds `G` as a 2-output-function.
* `add_objective!(mop, G, :expensive, 2, true)` additionally tells the algorithm that `G` can distinguish between a single input vector `x::Vector{Float64}` and a list of input vectors `x::Vector{Vector{Float64}}` and provide output in the same format respectively (useful for parallel execution of `G`).

### Internal settings
The `AlgoConfig` object above is used to provide the internal algorithm settings.
An incomplete table of settings:

| Setting name | dtype | default | description | 
| ------------ | ----- | ------- | ----------- |
| max_iter | Int | 1000 | max number of iterations |
| max_evals | Int or Float64 | Inf|  max number of objective evaluations |
| max_critical_loops | Int | 30 | max number of loops to perform in criticallity test before aborting |
| rbf_kernel | Symbol | :multiquadric | one of `:multiquadric`, `:exp`, `:cubic`, `thin_plate_spline` |
| rbf_poly_deg | Int64 | 1 | Degree of polynomial surrogate tail, either -1, 0 or 1 |
| rbf_shape_parameter | T where T<:Function | config_struct -> 1.0 | function to determine the shape parameter |
| max_model_points | Int64 | 2*n_vars^2 + 1 | maximum number of points to be included in the construction of one model |
| use_max_points | Bool | false | if `true` always use max number of allowed model points |
| descent_method | Symbol | :steepest | :steepest or :direct_search |
| ideal_point | Vector{Float64} | [] | ideal point to be used with `:direct_search` descent (if empty, calculate local minima)|
| all_objectives_descent | Bool | false | if `true` compute ρ as the minimum of descent ratios for ALL objetives |
| ν_success | Float64 | 0.4 | threshold for very successfull descent |
| ν_accept | Float64 | 0.0 | threshold for acceptable descent |
| γ_grow | Float64 | 2.0 | trust region growth factor |
| γ_shrink | Float64 | 0.8 | trust region shrink factor |
| γ_shrink_much | Float64 | 0.4 | severe shrinking factor |
| Δ₀ | Float64 | 0.4 | intial trust region radius |
| Δ_max | Float64 | 1 | maximum trust region radius |
| sampling_algorithm | Symbol | :orthogonal | `:orthogonal` or `:monte_carlo` |
| Δ_critical | Float64 | 1e-2 | stop if either `Δ <= Δ_critical && stepsize <= stepsize_min` or … |
| stepsize_min | Float64 | 1e-2 * Δ_critical | … |
| Δ_min | Float64 | Δ_critical * 1e-3 | … if `Δ<= Δ_min` |



