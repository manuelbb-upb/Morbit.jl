# make a configuration broadcastable
Broadcast.broadcastable( sm::AbstractSurrogate ) = Ref(sm);
Broadcast.broadcastable( sc::AbstractSurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from AbstractSurrogateConfig
max_evals( :: AbstractSurrogateConfig ) ::Int = typemax(Int)

# return data that is stored in iter data in each iteration
num_outputs( :: AbstractSurrogate ) :: Int = nothing

fully_linear( :: AbstractSurrogate ) :: Bool = false

set_fully_linear!( :: AbstractSurrogate, :: Bool ) = nothing

# can objective functions with same configuration types be combined 
# to a new vector objective?
combinable( :: AbstractSurrogateConfig ) :: Bool = false

needs_gradients( :: AbstractSurrogateConfig ) :: Bool = false
needs_hessians( :: AbstractSurrogateConfig ) :: Bool = false 

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

eval_models( :: AbstractSurrogate, :: AbstractVarScaler, ::Vec ) ::Vec = nothing 

# (Partially) Derived 

# either `get_gradient` or `get_jacobian` has to be implemented
# the other is derived 
function get_gradient( mod :: AbstractSurrogate, scal :: AbstractVarScaler, x_scaled ::Vec, ℓ )
    return vec( get_jacobian( mod, scal, x_scaled, [ℓ,]) )
end

function _get_jacobian_from_grads( mod :: AbstractSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec, rows = nothing )
    indices = isnothing(rows) ? (1:num_outputs(mod)) : rows 
    return transpose( hcat( (get_gradient(mod, scal, x_scaled, ℓ) for ℓ = indices)...) )
end

function get_jacobian( mod:: AbstractSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec)
    return _get_jacobian_from_grads( mod, scal, x_scaled, nothing )
end

function get_jacobian( mod:: AbstractSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec, rows)
    return _get_jacobian_from_grads( mod, scal, x_scaled, rows )
end

get_saveable_type( :: AbstractSurrogateConfig, x, y ) = Nothing
get_saveable( :: AbstractSurrogateMeta ) = nothing

prepare_update_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = meta
prepare_improve_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = meta

# overwrite if possible, this is inefficient:
""" 
    eval_models( mod, scal, x_scaled, ℓ)

Evaluate output(s) `ℓ` of model `mod` at scaled site `x_scaled`.
"""
function eval_models( sm :: AbstractSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec, ℓ) 
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

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `model`.
That is, if `model` is a surrogate for two scalar objectives, then `ℓ` must 
be either 1 or 2.
"""
function _get_optim_handle( model :: AbstractSurrogate, scal :: AbstractVarScaler, ℓ )
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
struct RefSurrogate{W <: Base.RefValue} <: AbstractSurrogate
	model_ref :: W
	output_indices :: Vector{Int}
end

function RefSurrogate( m :: AbstractSurrogate, output_indices ) 
	return RefSurrogate( Ref(m), output_indices )
end

""" 
    ExprSurrogate( model_ref, generated_function, output_indices)

Similar to `RefSurrogate`, `ExprSurrogate` holds a reference `model_ref` 
to some other surrogate.
But instead of evaluating this inner model directly, the function `generated_function`
is used for evaluation and differentiation.

[`RefSurrogate`](@ref)
"""
struct ExprSurrogate{W <: Base.RefValue,F} <: AbstractSurrogate
	model_ref :: W
	generated_function :: F
	output_indices :: Vector{Int}
end

""" 
    ExprSurrogate( model, expr_str, scal, output_indices)

Initialize an `ExprSurrogate` from `model` (and `scal :: AbstractVarScaler`) 
by turning `expr_str` into a function, where each occurence of `VRFE(x)` is 
replaced by a call to model and extracting the values at positions `output_indices`.

[`str2func`](@ref)
"""
function ExprSurrogate( m :: AbstractSurrogate, expr_str :: String, scal, output_indices )
	model_ref = Ref(m)
	gen_func = str2func(expr_str,m, scal, output_indices; register_adjoint = true)
	return ExprSurrogate( model_ref, gen_func, output_indices)
end


fully_linear( r :: Union{ExprSurrogate, RefSurrogate} ) = fully_linear( r.model_ref[] )
set_fully_linear!( r :: Union{ExprSurrogate, RefSurrogate}, val ) = set_fully_linear!( r.model_ref[], val )

function eval_models( m :: RefSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec )
	eval_models( m.model_ref[], scal, x_scaled, m.output_indices )
end
function eval_models( m :: ExprSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec )
	return Base.invokelatest(m.generated_function, x_scaled)
end

function get_gradient( m :: RefSurrogate, scal :: AbstractVarScaler, x_scaled ::Vec, ℓ = 1 )
	get_gradient( m.model_ref[], scal, x_scaled, m.output_indices[ℓ] )
end
function get_gradient( m :: ExprSurrogate, scal :: AbstractVarScaler, x_scaled :: Vec, ℓ = 1 )
	return Zygote.gradient( ξ -> m.generated_function(ξ)[m.output_indices[ℓ]],x_scaled)[1]
end

function get_jacobian( m :: RefSurrogate, scal :: AbstractVarScaler, x_scaled ::Vec )
	get_jacobian( m.model_ref[], scal, x_scaled, m.output_indices)
end
function get_jacobian( m :: RefSurrogate, scal :: AbstractVarScaler, x_scaled ::Vec, rows )
	get_jacobian( m.model_ref[], scal, x_scaled, m.output_indices[rows])
end

function get_jacobian( m :: ExprSurrogate, scal :: AbstractVarScaler, x_scaled ::Vec )
	return Zygote.jacobian( ξ -> m.generated_function(ξ)[m.output_indices],x_scaled)[1]
end
function get_jacobian( m :: ExprSurrogate, scal :: AbstractVarScaler, x_scaled ::Vec, rows )
	return Zygote.jacobian( ξ -> m.generated_function(ξ)[m.output_indices[rows]],x_scaled)[1]
end

"""
	str2func(expr_str, model, scal, output_indices; register_adjoint = true)

Parse a user provided string describing some function of `x` and 
return the resulting function.
Each occurence of "VREF(x)" in `expr_str` is replaced by a function 
evaluating outputs `output_indices` of `mod::AbstractSurrogate` at `x`.
If `register_adjoint == true`, then we register a custom adjoint for 
`vfunc` that uses the `get_jacobian` method.

The user may also use custom functions in `expr_str` hat have been 
registered with `register_func`.

[`register_func`](@ref)
"""
function str2func(expr_str, vfunc :: AbstractSurrogate, scal, output_indices; 
	register_adjoint = true
)
	global registered_funcs

	parsed_expr = Meta.parse(expr_str)
	reg_funcs_expr = Expr[ :($k = $v) for (k,v) = registered_funcs ]

	gen_func = @eval begin 
		let $(reg_funcs_expr...), scal = $scal, output_indices = $output_indices, mod = $vfunc, 
			VREF = ( x -> eval_models( mod, scal, x, output_indices ) );
			
			if $register_adjoint
				Zygote.@adjoint VREF(x) = VREF(x), y -> (get_jacobian(mod,scal,x,output_indices)'y,);
				Zygote.refresh();
			end
			x -> $(parsed_expr)
		end
	end
	return CountedFunc( gen_func )	# TODO can_batch ?
end
