# adding objectives to a mixed problem
@doc "Add a scalar objective to `mop::MixedMOP` modelled according to `model_config`."
function add_objective!(mop :: MixedMOP, func :: Function,
        model_config :: SurrogateConfig; batch_eval = false )
    return _add_objective!( mop, VectorObjectiveFunction, func, model_config, 1, batch_eval )
end

# FUNCTIONS TO ADD VECTOR OBJECTIVES TO A MixedMOP
@doc "Add a vector objective to `mop::MixedMOP` modelled according to `model_config`."
function add_vector_objective!(mop :: MixedMOP, func :: Function,
        model_config :: SurrogateConfig; n_out :: Int64, batch_eval = false )
    return _add_objective!( mop, VectorObjectiveFunction, func, model_config, n_out, batch_eval )
end

# OLD FUNCTIONS TO ADD OBJECTIVES TO AN MOP
# # these functions are less customizable and are mainly kept for convenience
@doc """
    add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )

Add scalar-valued objective function `func` to `mop` structure.
`func` must take an `RVec` as its (first) argument, i.e. represent a function ``f: ℝ^n → ℝ``.
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

mop = MixedMOP()
add_objective!(mop, f1, :cheap)     # gradient will be calculated using ForwardDiff
add_objective!(mop, f2, ∇f2 )       # gradient is provided
```
"""
function add_objective!( mop :: MixedMOP, func :: Function, type :: Symbol = :expensive, 
        n_out :: Int64 = 1, can_batch :: Bool = false )
    if type == :expensive
        objf_config = RbfConfig();
    elseif type == :cheap
        objf_config = ExactConfig(
            gradients = :autodiff
        )
    else
        error( "Type must be either `:expensive` or `:cheap`.")
    end

    if n_out > 1
    	add_vector_objective!( mop, func, objf_config; n_out = n_out, batch_eval = can_batch )
    else
        add_objective!(mop, func, objf_config; batch_eval = can_batch)
    end
end

@doc """
    add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})

Add scalar-valued objective function `func` and its vector-valued gradient `grad` to `mop` struture.
"""
function add_objective!( mop :: MixedMOP, func :: Function, grad :: Function)
    objf_config = ExactConfig(
        gradients = grad,
    )
    add_objective!(mop, func, objf_config)
end
