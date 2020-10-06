#module Objectives

import Base: broadcasted
using Parameters: @with_kw

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

#=
export MixedMOP, add_objective!, add_vector_objective!
export RbfConfig, ExactConfig, TaylorConfig, LagrangeConfig
export ObjectiveFuntion, BatchObjectiveFunction, VectorObjectiveFunction
=#

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

abstract type ModelConfig end

@with_kw mutable struct RbfConfig <: ModelConfig
    kernel :: Symbol = :multiquadric;
    shape_parameter :: Union{F,Float64} where {F<:Function} = 1.0;
    polynomial_degree :: Int64 = 1;

    θ_enlarge_1 :: Float64 = 2.0;
    θ_enlarge_2 :: Float64 = -1.0;  # reset
    θ_pivot :: Float64 = 1.0 / θ_enlarge_1;
    θ_pivot_cholesky :: Float64 = 1e-7;

    max_model_points :: Int64 = -1; # is probably reset in the algorithm
    use_max_points :: Bool = false;

    sampling_algorithm :: Symbol = :orthogonal # :orthogonal or :monte_carlo

    constrained :: Bool = false;    # restrict sampling of new sites

    @assert sampling_algorithm ∈ [:orthogonal, :monte_carlo] "Sampling algorithm must be either `:orthogonal` or `:monte_carlo`."
    @assert kernel ∈ Symbol.(["exp", "multiquadric", "cubic", "thin_plate_spline"]) "Kernel '$kernel' not supported yet."
end

import Base: ==
==( config1 :: RbfConfig, config2 :: RbfConfig ) = begin
    if  all( getfield( config1, fname ) == getfield( config2, fname )
            for fname in fieldnames(RbfConfig) if fname != :shape_parameter
        )

        if isa( config1.shape_parameter, Float64 )
            if isa( config2.shape_parameter, Float64 )
                return config1.shape_parameter == config2.shape_parameter
            else
                return all( isapprox.( config1.shape_parameter, config2.shape_parameter.(0:0.1:1.0)))
            end
        else
            return all(
                isapprox.(
                    config1.shape_parameter.(0:0.1:1.0),
                    config2.shape_parameter.(0:0.1:1.0)
                )
            )
        end
    else
        return false
    end
end

@with_kw mutable struct ExactConfig <: ModelConfig
    gradient :: Union{Symbol, Vector{F}, F} where{F<:Function} = :autodiff
    jacobian :: Union{Nothing, Symbol, F where{F<:Function} } = nothing
end

@with_kw mutable struct TaylorConfig <: ModelConfig
    degree :: Int64 = 1;
    gradient :: Union{Symbol, Vector{F}, F} where{F<:Function} = :fdm       # allow for array of functions too
    jacobian :: Union{Nothing, Symbol, F where{F<:Function} } = nothing
    hessian :: Union{Symbol, F, Vector{F}} where{F<:Function} = :fdm
    n_out :: Int64 = 0; # used internally when setting hessians
end

@with_kw mutable struct LagrangeConfig <: ModelConfig
    degree :: Int64 = 1;
    constrained :: Bool = false;    # restrict sampling of new sites
end

#---

# … to allow for batch evaluation outside of julia or to exploit parallelized objective functions
@with_kw struct BatchObjectiveFunction <: Function
    function_handle :: Union{T, Nothing} where{T<:Function} = nothing
end

function (objf::BatchObjectiveFunction)(args...)
    objf.function_handle(args...)
end

# overload broadcasting for BatchObjectiveFunction who are assumed to handle arrays themselves
function broadcasted( objf::BatchObjectiveFunction, args... )
    objf.function_handle( args... )
end

# Main type of function used internally
@with_kw mutable struct VectorObjectiveFunction <: Function
    n_out :: Int64 = 0;
    n_evals :: Int64 = 0;   # true function evaluations (also counts finite difference evaluations)

    model_config :: Union{ Nothing, C } where {C <: ModelConfig } = nothing;

    function_handle :: Union{T, Nothing} where{T <: Function, F <: Function} = nothing

    gradient_handles :: Union{Nothing, Vector{Any}} = nothing;
    hessian_handles :: Union{Nothing, Vector{Any}} = nothing;   # possibly needed for Taylor models

    jacobian_handle :: Union{Nothing, F} where {F <: Function } = nothing;  # needed for TaylorModel and ExactModel
    jacobian_cache :: Union{ Nothing, Dict{ Vector{Float64}, Array{Float64,2} } } = nothing

    internal_indices :: Vector{Int64} = [];
end

# for 'usual' function evaluation calls (args = single vector), simply delegate to function_handle
function (objf:: VectorObjectiveFunction )(x :: Vector{R} where{R<:Real})
    objf.n_evals += 1
    objf.function_handle(x)
end

# overload broadcasting for VectorObjectiveFunction
# # desired bevavior : f.( [[x11; x12]; [x21;x22]] ) =  [f11 f21;f12 f22]
function broadcasted( objf::VectorObjectiveFunction, X :: Vector{Vector{R}} where{R<:Real} )
    objf.n_evals += length(X)
    return objf.function_handle.( X );
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

#---

@with_kw mutable struct MixedMOP
    #vector_of_objectives :: Vector{ Union{ ObjectiveFunction, VectorObjectiveFunction } } = [];
    vector_of_objectives :: Vector{ VectorObjectiveFunction } = [];
    n_objfs :: Int64 = 0;

    internal_sorting :: Vector{Int64} = [];
    reverse_sorting :: Vector{Int64} = [];
    non_exact_indices ::Vector{Int64} = [];     # indices of vector returned by `eval_all_objectives` corresponding to non-exact objectives

    x_0 :: Vector{Float64} = [];
    lb :: Union{Nothing,Vector{Float64}} = nothing;
    ub :: Union{Nothing,Vector{Float64}} = nothing;
    is_constrained = !( isnothing(lb) || isnothing(ub) )
end
Broadcast.broadcastable(m::MixedMOP) = Ref(m);

function grad_from_jacobian( objf :: VectorObjectiveFunction, ℓ :: Int64 = 1 )
    grad_fun = function (x :: Vector{R} where{R<:Real} )
        if isnothing(objf.jacobian_cache) || x != keys( objf.jacobian_cache )[1]
            objf.jacobian_cache = Dict(
                x => objf.jacobian_handle(x)
            )
        end
        return vec(objf.jacobian_cache[x][ℓ, :])
    end
    return grad_fun
end

function set_gradients!( objf :: VectorObjectiveFunction, model_config :: Union{ExactConfig, TaylorConfig} )

    if isa( model_config.gradient, F where {F<:Function } ) && objf.n_out == 1
        obj.gradient_handles = [ model_config.gradient ]
    elseif isa( model_config.gradient, Vector{F} where {F<:Function } )
        if length( model_config.gradient ) != objf.n_out
            @error "Length of gradient array does not match number of vector objective outputs."
        else
            objf.gradient_handles = model_config.gradient ;
        end
    elseif isa( model_config.jacobian, F where{F<:Function} )
        objf.jacobian_handle = function (x :: Vector{R} where{R<:Real} )
            J = model_config.jacobian(x)
            objf.jacobian_cache = Dict( x => J )
        end
        @goto build_grads
    elseif isa( model_config.gradient, Symbol )
        mode = model_config.gradient
        @goto build_jacobian
    elseif isa( model_config.jacobian, Symbol )
        mode = model_config.jacobian
        @goto build_jacobian
    else
        @error "No jacobian method (:fdm or :autodiff) or gradient function handle(s) provided."
    end

    return nothing

    @label build_jacobian
    if objf.n_out > 1 @warn "Using FiniteDifferences or AutomaticDifferentiation for vector objectives is really really ineffective!!!" end
    if mode == :fdm
        objf.jacobian_handle = function (x :: Vector{R} where{R<:Real})
            # taking difference quotients of `objf` (instead of `objf.function_handle`) increases `n_evals` of the objective
            FD.finite_difference_jacobian( objf, x)
        end
    elseif mode == :autodiff
        if objf.n_out > 1
            objf.jacobian_handle = function (x :: Vector{R} where{R<:Real})
                AD.jacobian( objf, x )
            end
        else
            objf.jacobian_handle = function (x :: Vector{R} where{R<:Real})
                transpose(AD.gradient( objf, x ))
            end
        end
    else
        @error "No jacobian method (:fdm or :autodiff) or gradient function handle(s) provided."
    end

    @label build_grads
    gradient_handles = [];
    for i = 1 : objf.n_out
        push!( gradient_handles, grad_from_jacobian( objf, i) )
    end
    objf.gradient_handles = gradient_handles;
    return nothing
end

function set_hessians!( objf :: VectorObjectiveFunction, model_config :: TaylorConfig )
    for i = 1 : model_config.n_out
        if model_config.hessian == :fdm
            hessian_handle = function (x :: Vector{R} where{R<:Real})
                # taking difference quotients of `objf` (instead of `objf.function_handle`) increases `n_evals` of the objective
                FD.finite_difference_hessian( ξ -> objf(ξ)[i], x)
            end
        elseif model_config.hessian == :autodiff
            hessian_handle = function (x :: Vector{R} where{R<:Real})
                AD.hessian( ξ -> objf(ξ)[i], x )
            end
        elseif isa( model_config.hessian, Vector{F} where{F<:Function} )
            hessian_handle = model_config.hessian_handle[i]
        else
            @error "No Hessian method (:fdm or :autodiff) or function handle for output $i provided."
        end
        objf.hessian_handles[i] = hessian_handle;
    end
    if isa( model_config.hessian , Symbol )
        @warn "Using FiniteDifferences or AutomaticDifferentiation for vector objectives is really really ineffective!!!"
    end
end

# FUNCTIONS TO ADD A SCALAR OBJECTIVE TO A MixedMOP
# # helpers
function wrap_func( mop :: MixedMOP,  func :: T where{T <: Function}, batch_eval :: Bool )
    fx = x -> func( intobounds( x, mop.lb, mop.ub, Val(mop.is_constrained) ) );
    wrapped_func = batch_eval ? BatchObjectiveFunction( fx ) : fx;
end

function init_objective( mop :: MixedMOP, func :: T where{T <: Function}, model_config :: M where{M <: ModelConfig}, batch_eval :: Bool , n_out :: Int64 )
    wrapped_func = wrap_func(mop, func, batch_eval)

    objf = VectorObjectiveFunction(
        n_out = n_out,
        function_handle = wrapped_func,
        model_config = model_config,
        internal_indices = collect( mop.n_objfs + 1 : mop.n_objfs + n_out ),
    );
end

# # user methods
@doc "Add a scalar objective to `mop::MixedMOP` that should be modeled by an RBF network or Lagrange polynomials."
function add_objective!(mop :: MixedMOP, func :: T where{T <: Function}, model_config :: Union{RbfConfig, LagrangeConfig}; batch_eval = false )
    objf = init_objective( mop, func, model_config, batch_eval, 1 )
    # check if there is some other RBF objective with same settings and combine to save work
    for (other_objf_index, other_objf) ∈ enumerate(mop.vector_of_objectives)
        if other_objf.model_config == model_config
            deleteat!( mop.vector_of_objectives, other_objf_index )
            new_objf = combine( other_objf, objf )
            push!(mop.vector_of_objectives, new_objf)
            @goto objf_added
        end
    end
    push!(mop.vector_of_objectives, objf)
    @label objf_added
    mop.n_objfs += 1;
end

@doc "Add a scalar objective to `mop::MixedMOP` that should be evaluated exactely or modeled by a Taylor Polynomial."
function add_objective!(mop :: MixedMOP, func :: T where{T <: Function}, model_config :: Union{ExactConfig, TaylorConfig}; batch_eval = false )
    objf = init_objective( mop, func, model_config , batch_eval, 1)

    set_gradients!(objf, model_config)

    if isa(model_config, TaylorConfig) && model_config.degree > 1
        set_hessian!(objf, model_config)
    end

    push!(mop.vector_of_objectives, objf)
    @label objf_added
    mop.n_objfs += 1;
end

# FUNCTIONS TO ADD VECTOR OBJECTIVES TO A MixedMOP
@doc "Add a vector objective to `mop::MixedMOP` that should be modeled by an RBF network or Lagrange polynomials."
function add_vector_objective!(mop :: MixedMOP, func :: T where{T <: Function}, model_config :: Union{RbfConfig, LagrangeConfig}; n_out :: Int64, batch_eval = false )
    if n_out < 1
        @error "You must specify the number (positive integer) of outputs of `func` with the mandatory keyword argument `n_out`."
    end

    objf = init_objective( mop, func, model_config, batch_eval, n_out )

    # check if there is some other RBF/Lagrange objective with same settings
    # and combine to save work when building surrogates
    for (other_objf_index, other_objf) ∈ enumerate(mop.vector_of_objectives)
        if other_objf.model_config == model_config
            deleteat!( mop.vector_of_objectives, other_objf_index )
            new_objf = combine( other_objf, objf )
            push!(mop.vector_of_objectives, new_objf)
            @goto objf_added
        end
    end
    push!(mop.vector_of_objectives, objf)
    @label objf_added
    mop.n_objfs += n_out;
end

@doc "Add a scalar objective to `mop::MixedMOP` that should be evaluated exactely or modeled by a Taylor Polynomial."
function add_vector_objective!(mop :: MixedMOP, func :: T where{T <: Function}, model_config :: Union{ExactConfig, TaylorConfig}; n_out :: Int64, batch_eval = false )
    if n_out < 1
        @error "You must specify the number (positive integer) of outputs of `func` with the mandatory keyword argument `n_out`."
    end

    objf = init_objective( mop, func, model_config, n_out )

    set_gradients!(objf, model_config)

    if isa(model_config, TaylorConfig) && model_config.degree > 1
        model_config.n_out = n_out;
        set_hessians!(objf, model_config)
    end

    push!(mop.vector_of_objectives, objf)
    @label objf_added
    mop.n_objfs += n_out;
end


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

# Helper functions to scale sites to the unit hypercube [0,1]^n or unscale them
# based on variable boundaries given in a `MixedMOP`
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

# to internally evaluate all objectives …
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

# Get derivative information
function get_jacobian( objf :: VectorObjectiveFunction, x :: Vector{R} where{R<:Real} )
    if !isnothing(objf.jacobian_handle)
        return objf.jacobian_handle(x)
    else
        return transpose( (hcat( g(x) for g in objf.gradient_handles ) )... )
    end
end

# Functions to sort image vectors from ℝ^n_objfs

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
            gradient = :autodiff
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
        gradient = grad,
    )
    add_objective!(mop, func, objf_config)
end

#############################################################

#end#module

#=
# poor man's ~ Unit Testing ~
import .Objectives
o = Objectives
f1counter = 0;
f2counter = 0;
x0 = rand(2)
X = [rand(2) for i = 1 : 10]

function f1(x :: Vector{R}) where{R<:Real}
    global f1counter += 1;
    return 1.0
end

function f2(x :: Union{Vector{Float64}, Vector{Vector{Float64}}})
    global f2counter += 1;
    if isa(x,Vector{Float64})
        return 2.0
    else
        return 2 .* ones( length(x) )
    end
end

bo1 = o.BatchObjectiveFunction( f2 )
#---
mop = o.MixedMOP(lb=zeros(2), ub = ones(2))

rbfconf = o.RbfConfig()
o.add_objective!(mop, f1, rbfconf)
o.add_objective!(mop, x -> ones(2), o.ExactConfig() )
#o.add_objective!(mop, f2, rbfconf; batch_eval = true)

#@show y = o.eval_all_objectives( mop, rand(2) )
#Y = o.eval_all_objectives.(mop, X)
#@show o.reverse_internal_sorting.(mop, Y)

#o.add_objective!( mop, f1, :cheap)
#o.add_objective!(mop, f1, x -> 1.0)
=#
