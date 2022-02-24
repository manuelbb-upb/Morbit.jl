#src depends on abstract type AbstractSurrogateConfig & combinable( :: AbstractSurrogateConfig )

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
    RefVecFun( inner_ref, nl_index = nothing )

A `RefVecFun` simply stores a reference to a `VecFun` object.
Evaluation methods are delegated to this object.
`nl_index` is an optional information field used by `MOP`.

[`VecFun`](@ref), [`MOP`](@ref)
"""
struct RefVecFun{R} <: AbstractVecFun
	inner_ref :: R 
	nl_index :: Union{Nothing, NLIndex}

	function RefVecFun( fr :: R, i = nothing ) where R
		return new{R}(fr, i)
	end
	#TODO use own eval counter like in expr vec fun?
end

RefVecFun( F :: AbstractVecFun, nl_index = nothing ) = RefVecFun( Ref(F), nl_index )

"""
    ExprVecFun(inner_ref, generated_function, num_outputs, nl_index = nothing)

Like `RefVecFun`, an `ExprVecFun` object stores a reference to a `VecFun`.
But this reference is only used for retrieval of basic information. 
For evaluation, we have a `generated_function`.

[`RefVecFun`](@ref), [`VecFun`](@ref), [`str2func`](@ref)
"""
struct ExprVecFun{R <: Base.RefValue{<:AbstractVecFun},F,S} <: AbstractVecFun
	inner_ref :: R
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

    #expr_str = replace(_expr_str, "VREF(x)" => "VREF(x̂)")
	gen_func, substitutor = str2func( expr_str, vf )

	return ExprVecFun(Ref(vf),gen_func,substitutor,expr_str,n_out,nl_index)
end

"""
    CompositeVecFun(; inner_ref, outer_ref,
        inner_nl_index = nothing, outer_nl_index = nothing,
        num_outputs = num_outputs( outer_ref ) 
    )

A `CompositeVecFun` is a special `AbstractVecFun` to use for 
functions ``f:ℝⁿ→ℝᵏ, f = φ ∘ g``, where ``g:ℝⁿ → ℝᵐ`` is 
an expensive function (that is possibly used elsewhere too) 
and ``φ:ℝᵐ→ℝᵏ`` is an *outer* function.
Hence, when evaluating a `CompositeVecFun`, the value of the function 
referenced by `inner_ref` is plugged into `outer_ref`.
When evaluating the overall jacobian at ``x``, the chain rule 
provides
```math 
\\begin{aligned}
ℝ^{k×n} \\ni Df(x) &= Dφ(g(x))Dg(x), \\quad Dφ ∈ ℝ^{k×m}, Dg ∈ ℝ^{m×n},
∇f_ℓ(x) &= (Dφ(g(x))Dg(x))_{ℓ,:}^T = (∇φ_ℓ(g(x))^T Dg(x))^T = Dg(x)^T ∇φ_ℓ(g(x)).
```

For the Hessian of output ``f_ℓ`` it holds that 
```math
Hf_ℓ(x) = D(∇f_ℓ(x))
= 
D(Dφ_ℓ(g(x)) \\cdot Dg(x))
= (Hφ_ℓ(g(x)) \\ Dg(x))\\cdot Dg(x) + Dφ_ℓ(g(x)) \\cdot Hg(x)
```
"""
@with_kw struct CompositeVecFun{
    O <: Base.RefValue{<:AbstractVecFun},
    I <: Base.RefValue{<:AbstractVecFun},
} <: AbstractVecFun

    outer_ref :: O
    inner_ref :: I

    num_outputs :: Int = num_outputs(outer_ref[])

    nl_index :: Union{Nothing,NLIndex} = nothing
end

function CompositeVecFun( 
    outer :: AbstractVecFun, inner :: AbstractVecFun, 
    nl_index = nothing 
)
    return CompositeVecFun(;
        outer_ref = Ref(outer), inner_ref = Ref(inner), 
        nl_index,
    )
end

# `make_vec_fun` turns a function into a `VecFun`:
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
num_vars( r :: Union{RefVecFun,ExprVecFun} ) = num_vars( r.inner_ref[] )

num_outputs( objf :: VecFun ) = objf.n_out
num_outputs( r :: RefVecFun ) = num_outputs( r.inner_ref[] )
num_outputs( e :: Union{ExprVecFun,CompositeVecFun} ) = e.num_outputs

# `model_cfg` should in practice only ever be called for `VecFun`s (??? not anymore)
# it refers to the config for the inner AbstractVecFun which is considered expensive
model_cfg( vfun :: VecFun ) = vfun.model_config
model_cfg( vfun :: RefVecFun ) = model_cfg(vfun.inner_ref[])
model_cfg( vfun :: ExprVecFun ) = model_cfg(vfun.inner_ref[])
model_cfg( vfun :: CompositeSurrogate ) = model_cfg(vfun.inner_ref[])

function eval_vfun( vfun :: VecFun, x :: Vec )
    return vfun.function_handle(x)
end
eval_vfun( vfun :: RefVecFun, x ) = eval_vfun( vfun.inner_ref[], x)
eval_vfun( vfun :: ExprVecFun, x ) = vfun.generated_function(x)
eval_vfun( vfun :: CompositeVecFun, x ) = eval_vfun( vfun.outer_ref[], eval_vfun( vfun.inner_ref[], x) )

function Broadcast.broadcasted( ::typeof(eval_vfun), vfun :: VecFun, X :: VecVec )
    return vfun.function_handle.(X)
end
function Broadcast.broadcasted( ::typeof(eval_vfun), vfun :: RefVecFun, X :: VecVec )
    return vfun.inner_ref[].function_handle.(X)
end
function Broadcast.broadcasted( ::typeof(eval_vfun), s :: CompositeVecFun, X :: VecVec )
    X̃ = eval_vfun.( s.inner_ref[], X)
    return eval_vfun.(s.outer_ref[], X̃)
end

# ## Derivatives
# (optional) only required for certain SorrogateConfigs (`needs_gradients` == true)
# Note also, that in practice, the functions `get_gradient`, `_get_jacobian` 
# and `_get_hessian` are called only for `VecFun`, because we only allows those 
# to be added to an MOP for modelling
function _get_gradient( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, ℓ )
    return get_gradient( objf.diff_wrapper, x, ℓ )
end
#=
_get_gradient( r :: RefVecFun, x :: Vec, args ...) = _get_gradient( r.inner_ref[], x, args...)
function _get_gradient( ef :: ExprVecFun, x :: Vec, ℓ )
	return Zygote.gradient( ξ -> ef.generated_function(ξ)[ℓ], x )[1]
end

using LRUCache
@memoize LRU(maxsize=1) function _eval_inner( s :: CompositeVecFun, x :: Vec )
    return eval_vfun( s.inner_ref[], x )
end
# TODO the above caching is really improvised 
# it should not really matter in practice, as it is only used 
# for `get_jacobian` of an `ExactModel` or during the construction 
# of an `TaylorCallbackConfig` model.
function _get_gradient( s :: CompositeVecFun, x :: Vec, ℓ )
    gx = _eval_inner(s, x)
    ∇φ = _get_gradient( s.outer_ref[], gx, ℓ )
    Jg = _get_jacobian( s.inner_ref[], x )
    return vec( ∇φ'Jg )
end 
=#

function _get_jacobian( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, args... )
    return get_jacobian( objf.diff_wrapper, x, args... )
end
#=
_get_jacobian( r :: RefVecFun, x :: Vec, args ...) = _get_jacobian( r.inner_ref[], x, args...)
function _get_jacobian( ef :: ExprVecFun, x :: Vec )
	return Zygote.jacobian( ef.generated_function, x )[1]
end
function _get_jacobian( ef :: ExprVecFun, x :: Vec, rows )
	return Zygote.jacobian( ξ -> ef.generated_function(ξ)[rows], x )[1]
end
function _get_jacobian( s :: CompositeVecFun, x :: Vec, rows )
    gx = _eval_inner(s, x)
    Jφ = _get_jacobian( s.outer_ref[], gx, rows )
    Jg = _get_jacobian( s.inner_ref[], x )
    return Jφ*Jg
end
=#

function _get_hessian( objf :: VecFun{<:Any, <:DiffFn, <:Any}, x :: Vec, args... )
    return get_hessian( objf.diff_wrapper, x, args... )
end
#=
_get_hessian( r :: RefVecFun, x :: Vec, args ...) = _get_hessian( r.inner_ref[], x, args...)
function _get_hessian( ef :: ExprVecFun, x :: Vec, ℓ = 1 )
	return Zygote.gradient( ξ -> ef.generated_function(ξ)[ℓ], x )[1]
end

function _get_hessian( s :: CompositeVecFun, x :: Vec, ℓ)
    # (Hφ_ℓ(g(x)) \\ Dg(x))\\cdot Dg(x) + Dφ_ℓ(g(x)) \\cdot Hg(x)
    gx = _eval_inner(s,x)
    Hφ = _get_hessian(s.outer_ref[], gx, ℓ)
    Jg = _get_jacobian(s.inner_ref[], x)
    Jφ = _get_jacobian(s.outer_ref[], gx, [ℓ,])
    Hg = _get_hessian(s.inner_ref[], x)
    return (Hφ * Jg)*Jg + Jφ * Hg
end
=#

# this is used as a stopping criterion in algorithm and needs to be defined for any 
# `AbstractVecFun`
function _budget_okay( vfun :: AbstractVecFun, upper_bound = Inf ) 
    return num_evals(vfun) < min( max_evals(vfun), upper_bound )
end
function _budget_okay( vfun :: CompositeVecFun, upper_bound = Inf )
    return _budget_okay(vfun.inner_ref[], upper_bound) &&
        _budget_okay(vfun.outer_ref[], upper_bound)
end

# Derived Methods specific to `VecFun`
# Note: `num_evals` and `max_evals` are defined for all (but for printing only)
num_evals( vfun :: VecFun ) = getfield( vfun.function_handle, :counter )[]
num_evals( vfun :: Union{RefVecFun, ExprVecFun} ) = num_evals(vfun.inner_ref[])
num_evals( vfun :: CompositeVecFun ) = max( num_evals(vfun.inner_ref[]), num_evals(vfun.outer_ref[]) )

"(Soft) upper bound on the number of function calls. "
max_evals( objf :: AbstractVecFun) = max_evals( model_cfg(objf) );
max_evals( vfun :: CompositeVecFun ) = min( max_evals(vfun.inner_ref[]), max_evals(vfun.outer_ref[]) )

function reset_evals!( vfun :: VecFun, N = 0 )
    vfun.function_handle.counter[] = N
    nothing 
end
reset_evals!( vfun :: Union{RefVecFun,ExprVecFun}, N = 0 ) = reset_evals!( vfun.inner_ref[], N)
function reset_evals!( vfun :: CompositeVecFun, N = 0 )
    reset_evals!(vfun.inner_ref[], N)
    reset_evals!(vfun.outer_ref[], N)
    nothing 
end

# Does the model type allow for combination with models of same type?
combinable( objf :: VecFun ) = combinable( model_cfg(objf) );

function combinable( objf1 :: T, objf2 :: F ) where {T<:VecFun, F<:VecFun}
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
function str2func(expr_str, inner_vfunc :: AbstractVecFun; register_adjoint = false)
	global registered_funcs

    evaluator =  x -> eval_vfun( inner_vfunc, x ) 
    jacobian_handle =  x -> _get_jacobian( inner_vfunc, x)

    parsed_expr = Meta.parse(expr_str)
	reg_funcs_expr = Expr[ :($k = $v) for (k,v) = registered_funcs ]

    if register_adjoint
        @eval begin 
            let $(reg_funcs_expr...);
                Zygote.@adjoint $(evaluator)(x) = $(evaluator)(x), y -> ($jacobian_handle(x)'y,);
            end
        end
	    Zygote.refresh();
    end

	gen_func = mk_function( @__MODULE__, :( x -> begin  
		let $(reg_funcs_expr...), VREF = $evaluator ;  
			return $(parsed_expr)
		end
	end))

    subst_func = mk_function( @__MODULE__, :( (x,y) -> begin
        let $(reg_funcs_expr...), VREF = x -> y;
			return $(parsed_expr)
        end
	end))

    #=
    subst_func = function( x, y )
        result = @eval begin 
		    let $(reg_funcs_expr...), x = $x, y = $y, VREF = (x -> y);
			    $(parsed_expr)
            end
		end
        return result
	end
    =#
    
	return CountedFunc( gen_func ), subst_func	# TODO can_batch ?
end
