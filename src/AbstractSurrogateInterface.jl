# make a configuration broadcastable
Broadcast.broadcastable( sm::AbstractSurrogate ) = Ref(sm);
Broadcast.broadcastable( sc::AbstractSurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from AbstractSurrogateConfig
max_evals( :: AbstractSurrogateConfig ) ::Int = typemax(Int)

# return data that is stored in iter data in each iteration
num_outputs( :: AbstractSurrogate ) :: Int = nothing

fully_linear( :: AbstractSurrogate ) :: Bool = false

set_fully_linear!( :: AbstractSurrogate, bool_val ) = nothing

# can objective functions with same configuration types be combined 
# to a new vector objective?
combinable( :: AbstractSurrogateConfig ) :: Bool = false

needs_gradients( :: AbstractSurrogateConfig ) :: Bool = false
needs_hessians( :: AbstractSurrogateConfig ) :: Bool = false 

_variables( :: AbstractSurrogate ) :: Union{Nothing, AbstractVector{<:VarInd} } = nothing
# TODO make combinable bi-variate to check for to concrete configs if they are combinable

## TODO: make `prepare_init_model` and `_init_model` have a `ensure_fully_linear` kwarg too
function prepare_init_model( model_cfg, func_indices, mop, scal, x_it, sdb, algo_config; kwargs...) :: AbstractSurrogateMeta
    nothing
end

function init_model( meta, model_cfg, func_indices, mop, scal, x_it, sdb, algo_config; kwargs...) :: Tuple{<:AbstractSurrogate,<:AbstractSurrogateMeta}
    nothing 
end

function update_model( mod, meta, model_cfg, func_indices, mop, scal, x_it, sdb, algo_config; kwargs... )
    mod, meta 
end

eval_models( :: AbstractSurrogate, :: AbstractAffineScaler, ::Vec ) ::Vec = nothing 

# (Partially) Derived 

# either `get_gradient` or `get_jacobian` has to be implemented
# the other is derived 
function get_gradient( mod :: AbstractSurrogate, scal :: AbstractAffineScaler, x_scaled ::Vec, ℓ )
    return vec( get_jacobian( mod, scal, x_scaled, [ℓ,]) )
end

function _get_jacobian_from_grads( mod :: AbstractSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec, rows = nothing )
    indices = isnothing(rows) ? (1:num_outputs(mod)) : rows 
    return transpose( hcat( (get_gradient(mod, scal, x_scaled, ℓ) for ℓ = indices)...) )
end

function get_jacobian( mod:: AbstractSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec)
    return _get_jacobian_from_grads( mod, scal, x_scaled, nothing )
end

function get_jacobian( mod:: AbstractSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec, rows)
    return _get_jacobian_from_grads( mod, scal, x_scaled, rows )
end

get_saveable_type( :: AbstractSurrogateConfig, x, y ) = Nothing
get_saveable( :: AbstractSurrogateMeta ) = nothing

requires_update( cfg :: AbstractSurrogateConfig ) = true
requires_improve( cfg :: AbstractSurrogateConfig ) = true

prepare_update_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = meta
prepare_improve_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = meta

# overwrite if possible, this is inefficient:
""" 
    eval_models( mod, scal, x_scaled, ℓ)

Evaluate output(s) `ℓ` of model `mod` at scaled site `x_scaled`.
"""
function eval_models( sm :: AbstractSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec, ℓ) 
    eval_models(sm, scal, x_scaled)[ℓ]
end

improve_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = mod, meta

# check if surrogate configurations are equal (only really needed if combinable)
function Base.:(==)( cfg1 :: T, cfg2 :: T ) where T <: AbstractSurrogateConfig
    all( getfield(cfg1, fname) == getfield(cfg2, fname) for fname ∈ fieldnames(T) )
end

function Base.:(==)( cfg1 :: T, cfg2 :: F ) where {T <: AbstractSurrogateConfig, F<:AbstractSurrogateConfig}
    false 
end

## derived 
_eval_models_vec( mod :: AbstractSurrogate, args...) = _ensure_vec( eval_models(mod,args...) )

## NOTE This was added after introducing `_variables` to replace `eval_models`
eval_surrogate( :: Nothing, mod, id, x, args... ) = _eval_models_vec( mod, x, args... )
function eval_surrogate( var_inds :: AbstractVector{<:VarInd}, mod, id, x, args... )
	ξ = get_x_scaled( id, var_inds )
	return _eval_models_vec( mod, ξ, args... )
end

function eval_surrogate( mod, id, x = get_x_scaled(id), args...)
	return eval_surrogate( _variables(mod), mod, id, x, args... )
end

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `model`.
That is, if `model` is a surrogate for two scalar objectives, then `ℓ` must 
be either 1 or 2.
"""
function _get_optim_handle( model :: AbstractSurrogate, scal :: AbstractAffineScaler, ℓ )
    # Return an anonymous function that modifies the gradient if present
    function (x :: Vec, g :: Vec)
        if !isempty(g)
            g[:] = get_gradient( model, scal, x, ℓ)
        end
        return eval_models( model, scal, x, ℓ )
    end
end

# ## Helper Implementations 
""" 
    RefSurrogate(model_ref, output_indices)

A `RefSurrogate` holds a reference `model_ref` to some `AbstractSurrogate`
object and delegates most of the evaluation to this inner model.
`RefSurrogate`s are meant to model a single (vector-valued) objective or 
constraint function. 
As an `AbstractSurrogate` might model more than one objetive or constraint,
we have a field `output_indices` for filtering out the right output indices 
from the inner model output.

[`RefVecFun`](@ref)
"""
@with_kw struct RefSurrogate{W <: Base.RefValue} <: AbstractSurrogate
	model_ref :: W
	output_indices :: Vector{Int}
	inner_index :: InnerIndex
	num_outputs :: Int = num_outputs(inner_index)
	@assert num_outputs == length(output_indices)
end

function RefSurrogate( m :: AbstractSurrogate, output_indices, inner_index ) 
	return RefSurrogate(; model_ref = Ref(m), output_indices, inner_index )
end

num_outputs(r :: RefSurrogate) = r.num_outputs

@with_kw struct CompositeSurrogate{
	I <: Base.RefValue{<:AbstractSurrogate}, 
	O <: Base.RefValue{<:AbstractVecFun},
	# X, Y
} <: AbstractSurrogate
	
	model_ref :: I 
	outer_ref :: O 
	
	inner_output_indices = Int[]

	inner_index :: InnerIndex
	
	num_outputs :: Int = num_outputs( outer_ref[] )

	# caching of last inner model evaluation result
	# cache_in :: Vector{X} = MIN_PRECISION[]
	# cache_out :: Vector{Y} = MIN_PRECISION[]
end 

fully_linear( r :: Union{CompositeSurrogate,RefSurrogate} ) = fully_linear( r.model_ref[] )
set_fully_linear!( r :: Union{CompositeSurrogate,RefSurrogate}, val ) = set_fully_linear!( r.model_ref[], val )

function eval_models( m :: RefSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec )
	eval_models( m.model_ref[], scal, x_scaled, m.output_indices )
end
function eval_models( m :: RefSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec, ℓ )
	eval_models( m.model_ref[], scal, x_scaled, m.output_indices[ℓ] )
end

function _eval_inner( m :: CompositeSurrogate, scal, x_scaled )
	#= 
	if m.cache_in != x_scaled || isempty(m.cache_out)
		empty!(m.cache_in)
		append!(m.cache_in, x_scaled)
		empty!(m.cache_out)
		append!(m.cache_out,
			eval_models( m.model_ref[], scal, x_scaled, m.inner_output_indices )
		)
	end
	return m.cache_out=#
	return [
		untransform(x_scaled, scal);
		eval_models( m.model_ref[], scal, x_scaled, m.inner_output_indices )
	]
end
function eval_models( m :: CompositeSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec )
	return eval_vfun(
		m.outer_ref[],
		_eval_inner(m, scal, x_scaled )
	)
end

function get_gradient( m :: RefSurrogate, scal :: AbstractAffineScaler, x_scaled ::Vec, ℓ = 1 )
	return get_gradient( m.model_ref[], scal, x_scaled, m.output_indices[ℓ] )
end

function _composite_jac( Dφ, Dg, scal, x_scaled)
	# computing the jacobian of ``f(x) = φ( T(x), g(x) )``
	# (``x`` is from the scaled domain.)
	# ``T`` = affine unscaling of ``x``.
	# Let $ξ_0 = [t_0; g_0] = [T(x); g(x)].
	# We are provided with `Dφ`, a horizontal block matrix with entries 
	#   ``D_t φ(ξ_0)`` and ``D_g φ(ξ_0)``.
	# `Dg` is the jacobian of ``g`` at ``x``.
	# Let ``J`` be the jacobian of ``T`` at ``x``.
	# Then (using the chain rule):
	# ``Df(x) = D_t φ(ξ_0) J(x) + D_g φ(ξ_0) Dg(x)``.
	n_vars = length(x_scaled)
	J = jacobian_of_unscaling(scal) # n_vars × n_vars
	return Dφ[:, 1:n_vars] * J .+ Dφ[:, n_vars+1:end] * Dg
end

function get_gradient( m :: CompositeSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec, ℓ = 1)
	# Df = Dφ(g(x))Dg(x)
	gx = _eval_inner(m,scal,x_scaled)
	∇φ = _get_gradient( m.outer_ref[], gx, ℓ )
	Dg = get_jacobian(m.model_ref[], scal, x_scaled, m.inner_output_indices )
	return vec( _composite_jac(∇φ', Dg, scal, x_scaled) )
end

function get_jacobian( m :: RefSurrogate, scal :: AbstractAffineScaler, x_scaled ::Vec )
	get_jacobian( m.model_ref[], scal, x_scaled, m.output_indices)
end
function get_jacobian( m :: RefSurrogate, scal :: AbstractAffineScaler, x_scaled ::Vec, rows )
	get_jacobian( m.model_ref[], scal, x_scaled, m.output_indices[rows])
end

function get_jacobian( m :: CompositeSurrogate, scal :: AbstractAffineScaler, x_scaled :: Vec, args...)
	gx = _eval_inner(m,scal,x_scaled)
	Dφ = _get_jacobian( m.outer_ref[], gx, args... )
	Dg = get_jacobian(m.model_ref[], scal, x_scaled, m.inner_output_indices )
	return _composite_jac( Dφ, Dg, scal, x_scaled )
end
