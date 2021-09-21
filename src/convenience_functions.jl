# Convenient methods for adding objectives to a MOP
# Intended for use by the enduser.
# Requires concrete types `MOP` and `VectorObjectiveFunction`.

export add_objective!, add_vector_objective!

# adding objectives to a mixed problem
@doc "Add a scalar objective to `mop::MOP` modelled according to `model_config`."
function add_objective!(mop :: MOP, func :: Function;
        model_cfg :: SurrogateConfig = RbfConfig(), kwargs... )
    return add_objective!(mop, VectorObjectiveFunction, func; model_cfg, n_out = 1, kwargs... )
end

# FUNCTIONS TO ADD VECTOR OBJECTIVES TO A MOP
@doc "Add a vector objective to `mop::MOP` modelled according to `model_config`."
function add_vector_objective!(mop :: MOP, func :: Function;
        model_cfg :: SurrogateConfig = RbfConfig(), n_out :: Int64, kwargs... )
    return add_objective!( mop, VectorObjectiveFunction, func; model_cfg, n_out,  kwargs... )
end
#=
# OLD FUNCTIONS TO ADD OBJECTIVES TO AN MOP
# # these functions are less customizable and are mainly kept for convenience
@doc """
    add_objective!( mop :: MOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )

Add scalar-valued objective function `func` to `mop` structure.
`func` must take an `Vec` as its (first) argument, i.e. represent a function ``f: ℝ^n → ℝ``.
`type` must either be `:expensive` or `:cheap` to determine whether the function is replaced by a surrogate model or not.

If `type` is `:cheap` and `func` takes 1 argument only then its gradient is calculated by ForwardDiff.
A cheap function `func` with custom gradient function `grad` (representing ``∇f : ℝ^n → ℝ^n``) is added by

    add_objective!(mop, func, grad)

The optional argument `n_out` allows for the specification of vector-valued objective functions.
This is mainly meant to be used for *expensive* functions that are in some sense inter-dependent.

The flag `can_batch` defaults to false so that the objective function is simply looped over a bunch of arguments if required.
If `can_batch == true` then the objective function must be able to return an array of results when provided an array of input vectors
(whilst still returning a single result, not a singleton array containing the result, for a single input vector).

# Examples
```
# Define 2 scalar objective functions and a MOP ℝ^2 → ℝ^2

f1(x) =  x[1]^2 + x[2]

f2(x) = exp(sum(x))
∇f2(x) = exp(sum(x)) .* ones(2);

mop = MOP()
add_objective!(mop, f1, :cheap)     # gradient will be calculated using ForwardDiff
add_objective!(mop, f2, ∇f2 )       # gradient is provided
```
"""
function add_objective!( mop :: MOP, func :: Function, type :: Symbol = :expensive; 
        n_out :: Int = 1, kwargs... )

    if type == :expensive || type == :rbf
        model_cfg = RbfConfig()
    elseif type == :cheap || type == :exact
        model_cfg = ExactConfig()
    elseif type == :lagrange1
        model_cfg = LagrangeConfig(;degree = 1)
    elseif type == :lagrange || type == :lagrange2
        model_cfg = LagrangeConfig(;degree = 2)
    elseif type == :taylor1
        model_cfg = TaylorConfig(;degree = 1)
    elseif type == :lagrange || type == :lagrange2
        model_cfg = TaylorConfig(;degree = 2) 
    else
        @warn "Model type `$(type)` not known, using `RbfConfig`."
        model_cfg = RbfConfig()
    end

    return add_vector_objective!( mop, func; model_cfg, n_out, kwargs...)
end

@doc """
    add_objective!( mop :: MOP, func :: T where{T <: Function}, grad :: T where{T <: Function})

Add scalar-valued objective function `func` and its vector-valued gradient `grad` to `mop` struture.
"""
function add_objective!( mop :: MOP, func :: Function, grad :: Function; kwargs... )
    model_cfg = ExactConfig()
    add_objective!(mop, func; model_cfg, gradients = [grad,], kwargs... )
end

=#