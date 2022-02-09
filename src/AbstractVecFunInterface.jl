# depends on abstract type SurrogateConfig & combinable( :: SurrogateConfig )

Broadcast.broadcastable( objf::AbstractVecFun ) = Ref( objf );

# MANDATORY METHODS
"A general constructor."
function _wrap_func( :: T, fn :: VecFuncWrapper;
    model_cfg :: SurrogateConfig, n_out :: Int, can_batch :: Bool = false,
    gradients = nothing, jacobian = nothing, hessians = nothing, diff_method = nothing) :: T where {T<:Type{<:AbstractVecFun}}
    nothing
end

num_vars( :: AbstractVecFun ) :: Int = nothing
num_outputs( :: AbstractVecFun ) :: Int = nothing

model_cfg( :: AbstractVecFun ) :: SurrogateConfig = nothing

wrapped_function( :: AbstractVecFun ) :: VecFuncWrapper = nothing

# (optional) only required for certain SorrogateConfigs
get_objf_gradient( :: AbstractVecFun, x :: Vec, l :: Int = 1) = nothing
get_objf_jacobian( ::AbstractVecFun, x :: Vec) = nothing 
get_objf_hessian(  :: AbstractVecFun, x :: Vec, l :: Int = 1) = nothing

# Derived 
has_gradients( vf::AbstractVecFun ) = needs_gradients( model_cfg(vf) )
has_hessians( vf::AbstractVecFun ) = needs_hessians( model_cfg(vf) )

function _wrap_func( T :: Type{<:AbstractVecFun}, fn :: Function; 
    can_batch = false, kwargs... 
    )
    wrapped_fn = VecFuncWrapper( fn; can_batch )
    return _wrap_func(T, wrapped_fn; kwargs...)
end

num_evals( objf :: AbstractVecFun ) = getfield( wrapped_function(objf), :counter )[]
"(Soft) upper bound on the number of function calls. "
max_evals( objf :: AbstractVecFun) = max_evals( model_cfg(objf) );

function eval_objf( objf :: AbstractVecFun, x :: Vec )
    return wrapped_function( objf )(x)
end

function Broadcast.broadcasted( ::typeof(eval_objf), objf :: AbstractVecFun, X :: VecVec )
    return wrapped_function(objf).(X)
end

#= 
TODO how to pass down broadcasting for gradient, jacobian and hessians 
if `can_batch=true`
low priority

function Broadcast.broadcasted( ::typeof(get_objf_gradient), objf :: AbstractVecFun, X :: VecVec, l :: Int )
    return gradient_handle(objf, l).(X)
end
function Broadcast.broadcasted( ::typeof(get_objf_hessian), objf :: AbstractVecFun, X :: VecVec, l :: Int )
    return hessian_handle(objf, l).(X)
end
=#

# Does the model type allow for combination with models of same type?
combinable( objf :: AbstractVecFun ) = combinable( model_cfg(objf) );

function combinable( objf1 :: T, objf2 :: F ) where {T<:AbstractVecFun, F<:AbstractVecFun}
    return ( 
        combinable( objf1 ) && combinable( objf2 ) && 
        isequal(model_cfg( objf1 ), model_cfg( objf2 ))
    )
end
