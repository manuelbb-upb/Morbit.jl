import Base: broadcasted
using Parameters: @with_kw

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

# HELPER FUNCTIONS TO STAY IN FEASIBLE SET
const ε_bounds = 1e-15; # tolerancy for projection into feasible set

@doc "Put `x` back into box defined by `lb` and `ub`."
function intobounds( x :: Vector{R}, lb::Vector{L}, ub :: Vector{U}, constrained :: Val{true} ) where{
        R<:Real,L<:Real,U<:Real
    }
    global ε_bounds
    return min.( max.(lb .+ ε_bounds, x), ub .- ε_bounds  )
end
function intobounds( x :: Vector{R}, lb::Vector{L}, ub :: Vector{U}, constrained :: Val{false} )  where{
        R<:Real,L<:Real,U<:Real
    }
    x
end

function broadcasted( f::typeof(intobounds), X :: Vector{Vector{R}},
        lb::Vector{L}, ub :: Vector{U}, constrained :: Union{Val{true}, Val{false}} ) where{
            R<:Real,L<:Real,U<:Real
            }
    f.(X, Ref(lb), Ref(ub), constrained)
end

# this is required for BatchObjectiveFunction's
function intobounds( X :: Vector{Vector{R}}, lb::Vector{L}, ub :: Vector{U},
    constrained :: Union{Val{true}, Val{false}} )  where{
            R<:Real,L<:Real,U<:Real
        }
    intobounds.(X,lb,ub,constrained)
end

# evaluation of a BatchObjectiveFunction
function (objf::BatchObjectiveFunction)(args...)
    objf.function_handle(args...)
end

# overload broadcasting for BatchObjectiveFunction's
# that are assumed to handle arrays themselves
function broadcasted( objf::BatchObjectiveFunction, args... )
    objf.function_handle( args... )
end

# for 'usual' function evaluation calls (args = single vector)
# simply delegate to function_handle
function (objf:: VectorObjectiveFunction )(x :: Vector{R} where{R<:Real})
    objf.n_evals += 1
    objf.function_handle(x)
end

function new_batch( func1 :: F, func2 :: T ) where{ F <:Function ,T <: Function }
    return BatchObjectiveFunction(
        function( x :: Union{Vector{Vector{R}} , Vector{R}} ) where{R <: Real}
            if isa( x, Vector{R})
                [ func1(x); func2(x) ]
            else
                f1 = func1.(x)
                f2 = func2.(x)
                [ [f1[i];f2[i]] for i = eachindex(x) ]
            end
        end
    )
end

function new_func(func1 :: F, func2 :: T ) where{ F <:Function ,T <: Function }
    if F == BatchObjectiveFunction || T == BatchObjectiveFunction
        return new_batch( func1, func2 )
    else
        return function( x :: Vector{R} ) where{R <: Real}
            [ func1(x); func2(x) ]
        end
    end
end

function combine( objf1 :: VectorObjectiveFunction, objf2 :: VectorObjectiveFunction )
    # Create a new function handle
    new_objf = VectorObjectiveFunction(
        n_out = objf1.n_out + objf2.n_out,
        n_evals = max( objf1.n_evals, objf2.n_evals ),
        model_config = objf1.model_config,
        function_handle = new_func( objf1.function_handle, objf2.function_handle ),
        internal_indices = [ objf1.internal_indices; objf2.internal_indices ],
    )
    return new_objf
end

# FUNCTIONS TO ADD A SCALAR OBJECTIVE TO A MixedMOP
# # helpers
function wrap_func( mop :: MixedMOP,  func :: T where{T <: Function}, batch_eval :: Bool )
    fx = x -> func( intobounds( x, mop.lb, mop.ub, Val(mop.is_constrained) ) );
    wrapped_func = batch_eval ? BatchObjectiveFunction( fx ) : fx;
end

function init_objective( mop :: MixedMOP, func :: T where{T <: Function},
        model_config :: M where{M <: ModelConfig}, batch_eval :: Bool , n_out :: Int64 )
    wrapped_func = wrap_func(mop, func, batch_eval)

    objf = VectorObjectiveFunction(
        n_out = n_out,
        max_evals = max_evals( model_config ),
        function_handle = wrapped_func,
        model_config = model_config,
        internal_indices = collect( mop.n_objfs + 1 : mop.n_objfs + n_out ),
    );
end

function combine_objectives!(mop :: MixedMOP, objf :: VectorObjectiveFunction ,
        model_config :: Union{RbfConfig, LagrangeConfig, TaylorConfig})
    # check if there is some other objective with same settings to save work
    for (other_objf_index, other_objf) ∈ enumerate(mop.vector_of_objectives)
        if other_objf.model_config == model_config
            deleteat!( mop.vector_of_objectives, other_objf_index )
            new_objf = combine( other_objf, objf )
            push!(mop.vector_of_objectives, new_objf)
            new_objf.problem_position = length(mop.vector_of_objectives)
            return
        end
    end
    push!(mop.vector_of_objectives, objf)
    objf.problem_position = length(mop.vector_of_objectives)
end

function combine_objectives!(mop :: MixedMOP, objf :: VectorObjectiveFunction ,
        model_config :: ExactConfig)
    push!(mop.vector_of_objectives, objf)
end

# # user methods
@doc "Add a scalar objective to `mop::MixedMOP` modelled according to `model_config`."
function add_objective!(mop :: MixedMOP, func :: T where{T <: Function},
        model_config :: M where M <: ModelConfig; batch_eval = false )
    objf = init_objective( mop, func, model_config, batch_eval, 1 )
    
    push!( mop.original_functions, (func, 1) )
    
    combine_objectives!(mop, objf, model_config)
    mop.n_objfs += 1;
end


# FUNCTIONS TO ADD VECTOR OBJECTIVES TO A MixedMOP
@doc "Add a vector objective to `mop::MixedMOP` modelled according to `model_config`."
function add_vector_objective!(mop :: MixedMOP, func :: T where{T <: Function},
        model_config :: M where M <: ModelConfig; n_out :: Int64, batch_eval = false )
    if n_out < 1
        @error "You must specify the number (positive integer) of outputs of `func` with the mandatory keyword argument `n_out`."
    end

    objf = init_objective( mop, func, model_config, batch_eval, n_out )
    
    push!( mop.original_functions, (func, n_out) )

    combine_objectives!(mop, objf, model_config)
    mop.n_objfs += n_out;
end


# Helper functions …
@doc "Set `n_evals` to 0 for each VectorObjectiveFunction in `m.vector_of_objectives`."
function reset_evals!(m :: MixedMOP)
    for objf ∈ m.vector_of_objectives
        objf.n_evals = 0
    end
end

function max_evals!( m :: MixedMOP, M :: Int64 )
    for objf ∈ m.vector_of_objectives
        objf.max_evals = min( objf.max_evals, M )
    end
end

# to internally scale sites to the unit hypercube [0,1]^n
# or unscale them based on variable boundaries given in a `MixedMOP`
scale( mop :: MixedMOP, x :: Vector{Float64} , :: Val{true} ) = ( x .- mop.lb ) ./ ( mop.ub .- mop.lb );
scale( mop :: MixedMOP, x :: Vector{Float64}, :: Val{false} ) = x
scale( mop :: MixedMOP, x :: Vector{Float64} ) = scale( mop, x, Val( mop.is_constrained ) )
function scale!( mop :: MixedMOP, x :: Vector{Float64} )
    x[:] = scale( mop, x, Val( mop.is_constrained ))
end

unscale( mop :: MixedMOP, x :: Vector{Float64}, :: Val{true} ) = mop.lb .+ ( x .* ( mop.ub .- mop.lb ) );
unscale( mop :: MixedMOP, x :: Vector{Float64}, :: Val{false}) = x
unscale( mop :: MixedMOP, x :: Vector{Float64} ) = unscale( mop, x, Val( mop.is_constrained ))
function unscale!( mop :: MixedMOP, x :: Vector{Float64} )
    x[:] = unscale( mop, x, Val( mop.is_constrained ))
end

# … to internally evaluate all objectives …
# … at one single site `ξ` from the original domain
function eval_all_objectives( mop :: MixedMOP, ξ :: Vector{Float64}, unscale :: Val{false} )
    vcat( [ func(ξ) for func ∈ mop.vector_of_objectives ]... )
end

# … at one single (scaled) site `x`
function eval_all_objectives( mop :: MixedMOP, x :: Vector{Float64} )
    ξ = unscale( mop, x )
    return eval_all_objectives( mop, ξ, Val(false))
end

# # custom broadcasts
function broadcasted( f :: Union{ typeof(eval_all_objectives)}, mop :: MixedMOP, Ξ :: Vector{Vector{Float64}}, unscale :: Val{false} )
    [ vcat(z...) for z ∈ zip( [ func.(Ξ) for func ∈ mop.vector_of_objectives ]... ) ]
end

function broadcasted( f :: Union{ typeof(eval_all_objectives)}, mop :: MixedMOP, X :: Vector{Vector{Float64}})
    Ξ = unscale.(mop, X)
    f.(mop, Ξ, Val(false))
end


# Functions to sort image vectors from ℝ^n_objfs
function set_sorting!(mop :: MixedMOP)
    internal_sorting = Int64[];
    for objf ∈ mop.vector_of_objectives
        push!(internal_sorting, objf.internal_indices...)
    end
    mop.internal_sorting = internal_sorting;
    mop.reverse_sorting = sortperm( internal_sorting )
end

@doc "Set field `non_exact_indices` of argument `mop::MixedMOP`."
function set_non_exact_indices!( mop :: MixedMOP )
    mop.non_exact_indices = [];
    for ( objf_index, objf ) ∈ enumerate( mop.vector_of_objectives )
        if !isa( objf.model_config, ExactConfig )
            push!(mop.non_exact_indices, (objf_index : (objf_index + length(objf.internal_indices) - 1))...)
        end
    end
end

# # helper function to avoid repeated checking
function check_sorting!(mop::MixedMOP)
    if isempty(mop.internal_sorting)
        set_sorting!(mop)
    end
end

@doc "Sort image vector `y` to correspond to internal objective sorting."
function apply_internal_sorting(mop :: MixedMOP, y :: Vector{Float64})
    check_sorting!(mop)
    return y[mop.internal_sorting]
end

@doc "In place sort image vector `y` to correspond to internal objective sorting."
function apply_interal_sorting!( mop :: MixedMOP, y :: Vector{Float64} )
    check_sorting!(mop)
    y[:] = y[mop.internal_sorting]
end

@doc "Sort image vector `y` to correspond to internal objective sorting and
DON'T check indices."
function apply_internal_sorting(mop :: MixedMOP, y :: Vector{Float64}, check :: Val{false} )
    y[mop.internal_sorting]
end

@doc """
In-place sort image vector `y` to correspond to internal objective sorting
and DON'T check indices.
"""
function apply_internal_sorting!(mop :: MixedMOP, y :: Vector{Float64}, check :: Val{false} )
    y[:] = y[mop.internal_sorting]
end

@doc "Reverse sorting of vector `y` to correspond to original objective sorting."
function reverse_internal_sorting(mop :: MixedMOP, y :: Vector{Float64})
    check_sorting!(mop)
    y[ mop.reverse_sorting ]
end


@doc """
In-place reverse sorting of vector `y` to correspond to original objective sorting.
"""
function reverse_internal_sorting!(mop :: MixedMOP, y :: Vector{Float64})
    check_sorting!(mop)
    y[ mop.reverse_sorting ]
end

@doc "Reverse sorting of vector `y` to correspond to original objective sorting and DON'T check indices."
reverse_internal_sorting(mop :: MixedMOP, y :: Vector{Float64}, check :: Val{false} ) = y[mop.reverse_sorting]
function reverse_internal_sorting!(mop :: MixedMOP, y :: Vector{Float64}, check :: Val{false} )
    y[:] = y[mop.reverse_sorting]
end

# custom broadcast behavior for `apply_internal_sorting` and
# `reverse_internal_sorting` to only check once for sorting
function broadcasted( f :: Union{ typeof( apply_internal_sorting ), typeof(reverse_internal_sorting) },
        mop :: MixedMOP, Y :: Vector{Vector{Float64}} )
    check_sorting!(mop)
    f.( mop, Y, Val(false))
end

# # Other helper functions for calculation of decrease ratio ρ
function check_non_exact_indices!(mop::MixedMOP)
    if isempty( mop.non_exact_indices )
        set_non_exact_indices!( mop )
    end
end

@doc "Return components of `y` at `mop.non_exact_indices`."
function expensive_components( y :: Vector{T} where{T<:Real}, mop :: MixedMOP)
    check_non_exact_indices!( mop )
    y[ mop.non_exact_indices ]
end

@doc "Return components of `y` at `mop.non_exact_indices` and DON'T perform check if indices are set."
function expensive_components( y :: Vector{T} where{T<:Real}, mop :: MixedMOP, check :: Val{ false } )
    y[ mop.non_exact_indices ]
end

# OLD FUNCTIONS TO ADD OBJECTIVES TO AN MOP
# # these functions are less customizable and are mainly kept for convenience
@doc """
    add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )

Add scalar-valued objective function `func` to `mop` structure.
`func` must take an `Vector{Float64}` as its (first) argument, i.e. represent a function ``f: ℝ^n → ℝ``.
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
```jldoctest
# Define 2 scalar objective functions and a MOP ℝ^2 → ℝ^2

f1(x) =  x[1]^2 + x[2]

f2(x) = exp(sum(x))
∇f2(x) = exp(sum(x)) .* ones(2);

mop = MixedMOP()
add_objective!(mop, f1, :cheap)     # gradient will be calculated using ForwardDiff
add_objective!(mop, f2, ∇f2 )       # gradient is provided
```
"""
function add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )
    if type == :expensive
        objf_config = RbfConfig();
    elseif type == :cheap
        objf_config = ExactConfig(
            gradients = :autodiff
        )
    else
        @error "Type must be either `:expensive` or `cheap`."
    end

    if n_out > 1
    	add_vector_objective!( mop, func, obfj_config; n_out = n_out, batch_eval = can_bacth )
    else
        add_objective!(mop, func, objf_config; batch_eval = can_batch)
    end
end

@doc """
    add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})

Add scalar-valued objective function `func` and its vector-valued gradient `grad` to `mop` struture.
"""
function add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})
    objf_config = ExactConfig(
        gradients = grad,
    )
    add_objective!(mop, func, objf_config)
end
