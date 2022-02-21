#src depends on abstract type AbstractSurrogateConfig & combinable( :: AbstractSurrogateConfig )

# Include helpers to differentiate wrapped functions:
include("DiffFn.jl")

Broadcast.broadcastable( objf::AbstractVecFun ) = Ref( objf );

# The main implementations:

"""
    VecFun(; n_out, model_config, func_handle, diff_wrapper = nothing)

Wrap the function `func_handle` to ensure vector output.
A `VecFun` also provides a unified differentiation API.
"""
@with_kw struct VecFun{
    SC <: AbstractSurrogateConfig,
    D <: Union{Nothing,DiffFn},
    F <: CountedFunc,
} <: AbstractVecFun

    n_out :: Int = 0

    model_config :: SC

    function_handle :: F 

    diff_wrapper :: D = nothing
end

"""
    RefVecFun( function_ref, nl_index = nothing )

A `RefVecFun` simply stores a reference to a `VecFun` object.
Evaluation methods are delegated to this object.
`nl_index` is an optional information field used by `MOP`.

[`VecFun`](@ref), [`MOP`](@ref)
"""
struct RefVecFun{R} <: AbstractVecFun
	function_ref :: R 
	nl_index :: Union{Nothing, NLIndex}

	function RefVecFun( fr :: R, i = nothing ) where R
		return new{R}(fr, i)
	end
	#TODO use own eval counter like in expr vec fun?
end

RefVecFun( F :: AbstractVecFun, nl_index = nothing ) = RefVecFun( Ref(F), nl_index )

"""
    ExprVecFun(function_ref, generated_function, num_outputs, nl_index = nothing)

Like `RefVecFun`, an `ExprVecFun` object stores a reference to a `VecFun`.
But this reference is only used for retrieval of basic information. 
For evaluation, we have a `generated_function`.

[`RefVecFun`](@ref), [`VecFun`](@ref), [`str2func`](@ref)
"""
struct ExprVecFun{R,F,S} <: AbstractVecFun
	function_ref :: R
	generated_function :: F
    substitutor :: S
    expr_str :: String
	num_outputs :: Int
	nl_index :: Union{Nothing, NLIndex}
	function ExprVecFun(fr::R, f::F, s::S, e, n, i = nothing) where{R,F,S}
		return new{R,F,S}(fr,f,s,e,n,i)
	end
end

"""
    ExprVecFun( vf :: AbstractVecFun, expr_str :: String,
        n_out :: Int, nl_index = nothing )

Initialize an `ExprVecFun` object from another vector function.
An evaluation function is created from `expr_str` by replacing each occurence 
of "VREF(x)" to a call to `vf`.
"""
function ExprVecFun( vf :: AbstractVecFun, expr_str :: String, n_out :: Int, nl_index = nothing)
	@assert n_out > 0 "Need a positive output number."

	gen_func, substitutor = str2func( expr_str, vf )

	return ExprVecFun(Ref(vf),gen_func,substitutor,expr_str,n_out,nl_index)
end

# The user will only ever have to generate a `VecFun`, and `make_vec_fun`
# turns a function into a `VecFun`:
"""
    make_vec_fun( fn; 
    model_cfg, n_out, can_batch = false,
    gradients = nothing, jacobian = nothing, hessian = nothing,
    diff_method = FiniteDiffWrapper )

Pack the function `fn::CountedFunc` into a `VecFun` and ensure that 
appropriate derivative information can be queried, as needed by `model_cfg`.
"""
function make_vec_fun( fn :: CountedFunc; 
        model_cfg::AbstractSurrogateConfig, n_out :: Int,
        gradients :: Union{Nothing,Function,AbstractVector{<:Function}} = nothing, 
        jacobian :: Union{Nothing,Function} = nothing, 
        hessians :: Union{Nothing,AbstractVector{<:Function}} = nothing,
        diff_method :: Union{Type{<:DiffFn}, Nothing} = FiniteDiffWrapper
    )
      
    if needs_gradients( model_cfg )
        if ( isnothing(gradients) && isnothing(jacobian) )
            if isnothing(diff_method)
                error("""
                According to `model_cfg` we need gradient information.
                You can provide a list of functions with the `gradients` keyword or a 
                `jacobian` function. 
                Alternatively, you can use the keyword argument `diff_method` with 
                `Morbit.FiniteDiffWrapper` or `Morbit.AutoDiffWrapper`.
                """)
            else
                @warn "Using $(diff_method) for gradients."
            end
        end
    end

    if needs_hessians( model_cfg )
        if isnothing(hessians)
            if isnothing(diff_method)
                error("""
                According to `model_cfg` we need hessian information.
                You can provide a list of functions with the `hessians` keyword.
                Alternatively, you can use the keyword argument `diff_method` with 
                `Morbit.FiniteDiffWrapper` or `Morbit.AutoDiffWrapper`.
                """)
            else
                @warn "Using $(diff_method) for hessians."
            end
        end
    end

    diff_wrapper = if (needs_gradients(model_cfg) || needs_hessians(model_cfg)) && !isnothing(diff_method)
        diff_method(;
            objf = fn,
            gradients = gradients,
            jacobian = jacobian,
            hessians = hessians
        )
    else
        nothing
    end
    
    return VecFun(;
        n_out = n_out,
        function_handle = fn, 
        model_config = model_cfg,
        diff_wrapper
    )
end

"""
    make_vec_fun( fn; 
    model_cfg, n_out, can_batch = false,
    gradients = nothing, jacobian = nothing, hessian = nothing,
    diff_method = FiniteDiffWrapper )

Pack the function `fn::Function` into a `VecFun` and ensure that 
appropriate derivative information can be queried, as needed by `model_cfg`.
"""
function make_vec_fun( fn :: Function; 
    can_batch = false, kwargs... 
    )
    wrapped_fn = CountedFunc( fn; can_batch )
    return make_vec_fun(wrapped_fn; kwargs...)
end

# ## Information retrieval methods:

num_vars( objf :: VecFun ) = objf.n_in
num_vars( r :: Union{RefVecFun,ExprVecFun} ) = num_vars( r.function_ref[] )

num_outputs( objf :: VecFun ) = objf.n_out
num_outputs( r :: RefVecFun ) = num_outputs( r.function_ref[] )
num_outputs( e :: ExprVecFun ) = e.num_outputs

model_cfg( objf :: VecFun ) = objf.model_config
model_cfg( r :: Union{ExprVecFun,RefVecFun} ) = model_cfg( r.function_ref[] )

# (`wrapped_function` should return a function of type `CountedFunc`)
wrapped_function( vfun::VecFun ) = vfun.function_handle
wrapped_function( r :: RefVecFun ) = wrapped_function( r.function_ref[] )
wrapped_function( e :: ExprVecFun ) = e.generated_function 

# evaluation
function eval_objf( vfun :: VecFun, x :: Vec )
    return wrapped_function( vfun )(x)
end

function Broadcast.broadcasted( ::typeof(eval_objf), objf :: AbstractVecFun, X :: VecVec )
    return wrapped_function(objf).(X)
end

eval_objf( r :: RefVecFun, args... ) = eval_objf( r.function_ref[], args... )
eval_objf( e :: ExprVecFun, args... ) = wrapped_function(e)(args...)

# ## Derivatives
# (optional) only required for certain SorrogateConfigs
function _get_gradient( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, args... )
    return get_gradient( objf.diff_wrapper, x, args... )
end
_get_gradient( r :: RefVecFun, x :: Vec, args ...) = _get_gradient( r.function_ref[], x, args...)
function _get_gradient( ef :: ExprVecFun, x :: Vec, ℓ = 1 )
	return Zygote.gradient( ξ -> ef.generated_function(ξ)[ℓ], x )[1]
end

function _get_jacobian( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, args... )
    return get_jacobian( objf.diff_wrapper, x, args... )
end
_get_jacobian( r :: RefVecFun, x :: Vec, args ...) = _get_jacobian( r.function_ref[], x, args...)
function _get_jacobian( ef :: ExprVecFun, x :: Vec )
	return Zygote.jacobian( ef.generated_function, x )[1]
end
function _get_jacobian( ef :: ExprVecFun, x :: Vec, rows )
	return Zygote.jacobian( ξ -> ef.generated_function(ξ)[rows], x )[1]
end

function _get_hessian( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, args... )
    return get_hessian( objf.diff_wrapper, x, args... )
end
_get_hessian( r :: RefVecFun, x :: Vec, args ...) = _get_hessian( r.function_ref[], x, args...)
function _get_hessian( ef :: ExprVecFun, x :: Vec, ℓ = 1 )
	return Zygote.gradient( ξ -> ef.generated_function(ξ)[ℓ], x )[1]
end

# ## Derived Methods
has_gradients( vf::AbstractVecFun ) = needs_gradients( model_cfg(vf) )
has_hessians( vf::AbstractVecFun ) = needs_hessians( model_cfg(vf) )

num_evals( objf :: AbstractVecFun ) = getfield( wrapped_function(objf), :counter )[]
"(Soft) upper bound on the number of function calls. "
max_evals( objf :: AbstractVecFun) = max_evals( model_cfg(objf) );

# Does the model type allow for combination with models of same type?
combinable( objf :: AbstractVecFun ) = combinable( model_cfg(objf) );

function combinable( objf1 :: T, objf2 :: F ) where {T<:AbstractVecFun, F<:AbstractVecFun}
    return ( 
        combinable( objf1 ) && combinable( objf2 ) && 
        isequal(model_cfg( objf1 ), model_cfg( objf2 ))
    )
end

#= 
TODO how to pass down broadcasting for gradient, jacobian and hessians 
if `can_batch=true`
low priority

function Broadcast.broadcasted( ::typeof(_get_gradient), objf :: VecFun, X :: VecVec, l :: Int )
    return gradient_handle(objf, l).(X)
end
function Broadcast.broadcasted( ::typeof(_get_hessian), objf :: VecFun, X :: VecVec, l :: Int )
    return hessian_handle(objf, l).(X)
end
=#

"""
	str2func(expr_str, vfunc)

Parse a user provided string describing some function of `x` and 
return the resulting function.
Each occurence of "VREF(x)" in `expr_str` is replaced by a function 
evaluating `vfunc::AbstractVecFun` at `x`.

The user may also use custom functions in `expr_str` hat have been 
registered with `register_func`.

[`register_func`](@ref)
"""
function str2func(expr_str, vfunc :: AbstractVecFun)
	global registered_funcs

	parsed_expr = Meta.parse(expr_str)
	reg_funcs_expr = Expr[ :($k = $v) for (k,v) = registered_funcs ]

	gen_func = @eval begin 
		let $(reg_funcs_expr...), vfunc = $vfunc, VREF = ( x -> eval_objf( vfunc, x ) );
            # register custom adjoint (important for `ExactModel`)
            Zygote.@adjoint VREF(x) = VREF(x), y -> (_get_jacobian(vfunc, x)'y,);
			Zygote.refresh();
            # return evaluator function
			x -> $(parsed_expr)
		end
	end

    subst_func = function( x, y )
        result = @eval begin 
		    let $(reg_funcs_expr...), x = $x, y = $y, VREF = (x -> y);
			    $(parsed_expr)
            end
		end
        return result
	end
    
	return CountedFunc( gen_func ), subst_func	# TODO can_batch ?
end

#=

# if cache is used
function eval_objf( vfun :: VecFun, x :: Vec )
    inner_func = wrapped_function(vfun)
    _multi_val_cache = _can_batch(inner_func) # alternative test: vfun.cache_in isa Vector{Vector}
    
    if _multi_val_cache
        x_ind = findfirst( ξ -> ξ == x, vfun.cache_in )
        if !isnothing(x_ind)
            return vfun.cache_out[x_ind]
        end
    else
        x == vfun.cache_in && return vfun.cache_out
    end
    
    empty!(vfun.cache_in)
    empty!(vfun.cache_out)
    
    y = inner_func(x)

    if _multi_val_cache
        push!(vfun.cache_in, x)
        push!(vfun.cache_out, y)
    else
        append!(vfun.cache_in, x)
        append!(vfun.cache_out, y)
    end

    return y    
end

function Broadcast.broadcasted( ::typeof(eval_objf), objf :: VecFun{<:Any,<:Any,<:Any,<:Nothing,<:Nothing}, X :: VecVec )
    return wrapped_function(objf).(X)
end

function Broadcast.broadcasted( ::typeof(eval_objf), objf :: VecFun, X :: VecVec )
    inner_func = wrapped_function(vfun)
    _multi_val_cache = _can_batch(inner_func) # alternative test: vfun.cache_in isa Vector{Vector}
    
    X_indices = Union{Int,Nothing}[]
    if _multi_val_cache
        for x = X
            x_ind = findfirst( ξ -> ξ == x, vfun.cache_in )
            push!(X_indices, x_ind)
        end
    else
        ξ = vfun.cache_in
        for (i,x) = enumerate(X)
            if x == ξ
                push!(X_indices, i)
            else
                push!(X_indices, nothing)
            end
        end
    end
    
    uneval_indices = isnothing.(X_indices)
    Y_new = inner_func.( X_indices[uneval_indices] )
    
    Y = [ isnothing(x_ind) ? Y_new[j] : vfun.cache_out[x_ind] for (j,x_ind) = enumerate(X_indices) ]

    empty!(vfun.cache_in)
    empty!(vfun.cache_out)
    
    if _multi_val_cache
        append!(vfun.cache_in, X)
        append!(vfun.cache_out, Y)
    else
        append!(vfun.cache_in, X[end])
        append!(vfun.cache_out, Y[end])
    end

    return y    
end
=#