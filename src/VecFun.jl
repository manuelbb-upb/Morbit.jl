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

"""
    make_outer_fun( func :: Function; n_vars, n_out, kwargs...)

Helper function to build a `VecFun` from a function `func` taking 
two vector valued arguments: First, the argument vector ``x`` 
of length `n_vars` and second, the value of an inner function evaluated at ``x``.
"""
function make_outer_fun( fn;
    n_vars :: Int,
    jacobian_1 :: Union{Function, Nothing} = nothing, 
    jacobian_2 :: Union{Function, Nothing} = nothing,
    kwargs...
)
    @assert n_vars > 0 "The number of variables `n_vars` must be positive."
    
    # ξ = [x;y]; 
    # we have to use this concatenation because VecFuns were initially designed 
    # for univariate vector input
    func = ξ -> fn( ξ[1:n_vars], ξ[n_vars+1:end] )
    jacobian = if !(isnothing(jacobian_1) || isnothing(jacobian_2))
        ξ -> hcat( jacobian_1(ξ), jacobian_2(ξ) )
    else
        nothing
    end
    return make_vec_fun( func; jacobian, model_cfg = ExactConfig(), kwargs...)
end


"""
    make_outer_fun( expr_str :: String; n_vars, kwargs...)

Helper function to build a `VecFun` from a an expression string.
"""
function make_outer_fun( expr_str :: AbstractString;
    kwargs...
)
    fn = outer_fn_from_expr(expr_str)
    return make_outer_fun(fn; kwargs...)
end   

# ## Information retrieval methods:

num_outputs( objf :: VecFun ) = objf.n_out
num_outputs( r :: RefVecFun ) = num_outputs( r.inner_ref[] )
num_outputs( e :: CompositeVecFun ) = e.num_outputs

# `model_cfg` should in practice only ever be called for `VecFun`s (??? not anymore)
# it refers to the config for the inner AbstractVecFun which is considered expensive
model_cfg( vfun :: VecFun ) = vfun.model_config
model_cfg( vfun :: RefVecFun ) = model_cfg(vfun.inner_ref[])
model_cfg( vfun :: CompositeVecFun ) = model_cfg(vfun.inner_ref[])

function eval_vfun( vfun :: VecFun, x :: Vec )
    return vfun.function_handle(x)
end
eval_vfun( vfun :: RefVecFun, x ) = eval_vfun( vfun.inner_ref[], x)

function eval_vfun( vfun :: CompositeVecFun, x )
    return eval_vfun( 
        vfun.outer_ref[], 
        [x; eval_vfun( vfun.inner_ref[], x)] 
    )
end

function Broadcast.broadcasted( ::typeof(eval_vfun), vfun :: VecFun, X :: VecVec )
    return vfun.function_handle.(X)
end
function Broadcast.broadcasted( ::typeof(eval_vfun), vfun :: RefVecFun, X :: VecVec )
    return vfun.inner_ref[].function_handle.(X)
end
function Broadcast.broadcasted( ::typeof(eval_vfun), s :: CompositeVecFun, X :: VecVec )
    X̃ = [[x,y] for (x,y) = zip(X, eval_vfun.( s.inner_ref[], X))]
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
num_evals( vfun :: RefVecFun ) = num_evals(vfun.inner_ref[])
num_evals( vfun :: CompositeVecFun ) = num_evals(vfun.inner_ref[])
dont_count!( vfun :: VecFun ) = dont_count!( vfun.function_handle )
dont_count!( vfun :: RefVecFun ) = dont_count!( vfun.inner_ref[] )
dont_count!( vfun :: CompositeVecFun ) = begin 
    dont_count!( vfun.inner_ref[] )
    dont_count!( vfun.outer_ref[] )
end

do_count!( vfun :: VecFun ) = do_count!( vfun.function_handle )
do_count!( vfun :: RefVecFun ) = do_count!( vfun.inner_ref[] )
do_count!( vfun :: CompositeVecFun ) = begin 
    do_count!( vfun.inner_ref[] )
    do_count!( vfun.outer_ref[] )
end

#num_evals( vfun :: CompositeVecFun ) = max( num_evals(vfun.inner_ref[]), num_evals(vfun.outer_ref[]) )

"(Soft) upper bound on the number of function calls. "
max_evals( objf :: AbstractVecFun) = max_evals( model_cfg(objf) );
max_evals( vfun :: CompositeVecFun ) = max_evals(vfun.inner_ref[])
#max_evals( vfun :: CompositeVecFun ) = min( max_evals(vfun.inner_ref[]), max_evals(vfun.outer_ref[]) )

function reset_evals!( vfun :: VecFun, N = 0 )
    vfun.function_handle.counter[] = N
    nothing 
end
reset_evals!( vfun :: RefVecFun, N = 0 ) = reset_evals!( vfun.inner_ref[], N)
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


function outer_fn_from_expr(expr_str)
	global registered_funcs

    parsed_expr = Meta.parse(expr_str)
	reg_funcs_expr = Expr[ :($k = $v) for (k,v) = registered_funcs ]

    outer_fn = mk_function( @__MODULE__, :( (x,gx) -> begin
        let $(reg_funcs_expr...), VREF = gx;
			return $(parsed_expr)
        end
	end))
    
	return outer_fn
end
