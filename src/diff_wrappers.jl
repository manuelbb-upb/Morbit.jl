struct HessWrapper <: DiffFn
    list_of_hess_fns :: Vector{<:Function}
end
HessWrapper( hf :: Function ) = HessWrapper( [hf,] );

function get_hessian( hw :: HessWrapper, x :: RVec, ℓ :: Int = 1)
    return list_of_hess_fns[ℓ](x);
end

struct GradWrapper <: DiffFn
    list_of_grads :: Vector{<:Function}
    jacobian_handle :: Union{Nothing, Function} 
end

GradWrapper( g :: Function, jh :: Union{Nothing, Function} ) = GradWrapper( [g,], jh );
GradWrapper( g :: Nothing, jh :: Union{Nothing, Function} ) = GradWrapper( Function[], jh );

# true =̂ gw.list_of_grads not empty
function _get_gradient( :: Val{true}, gw :: GradWrapper, x :: RVec, ℓ :: Int = 1)
    return vec( gw.list_of_grads[ℓ](x) );
end 
# false =̂ list_of_grads empty but jacobian handle provided
function _get_gradient( :: Val{false}, gw :: GradWrapper, x :: RVec, ℓ :: Int = 1)
    return vec( gw.jacobian_handle(x)[ℓ, :] );
end

function get_gradient( gw :: GradWrapper, x :: RVec, ℓ :: Int = 1)
    return _get_gradient( Val(!isempty(gw.list_of_grads)), gw, x, ℓ );
end

function get_jacobian( gw :: GradWrapper, x :: RVec )
    # prefer user defined handle over gradient iteration
    if !isnothing(gw.jacobian_handle)
        return gw.jacobian_handle( x )
    else
        return transpose( hcat( 
            ( _get_gradient( Val(true), gw, x, ℓ ) for ℓ = eachindex(gw.list_of_grads) )...
        ));
    end
end

struct AutoDiffWrapper <: DiffFn
    objf :: AbstractObjective 
end

function get_jacobian( adw :: AutoDiffWrapper, x :: RVec )
    return AD.jacobian( _eval_handle(adw.objf), x );
end

#=
# single output
function _get_gradient( :: Val{true}, adw :: AutoDiffWrapper, x :: RVec, :: Int )
    return AD.gradient( _eval_handle(adw.objf), x );
end

# multiple outputs
function _get_gradient( :: Val{false}, adw :: AutoDiffWrapper, x :: RVec, ℓ :: Int )
    return AD.gradient( _eval_handle(adw.objf, ℓ), x );
end
=#

function get_gradient( adw :: AutoDiffWrapper, x :: RVec, ℓ :: Int = 1)
    return AD.gradient( _eval_handle(adw.objf, ℓ), x );
end  

# not optimized for multiple outputs
function get_hessian( adw :: AutoDiffWrapper, x :: RVec, ℓ :: Int = 1)
    return AD.hessian( _eval_handle(adw.objf, ℓ), x )
end

struct FiniteDiffWrapper <: DiffFn
    objf :: AbstractObjective
end

function get_jacobian( fdw :: FiniteDiffWrapper, x :: RVec )
    return FD.finite_difference_jacobian( _eval_handle( fdw.objf), x );
end

#=
# single output
function _get_gradient( :: Val{true}, fdw :: FiniteDiffWrapper, x :: RVec, :: Int )
    return FD.finite_difference_gradient( _eval_handle(fdw.objf), x );
end

# multiple outputs  TODO use jacobian rows instead?
function _get_gradient( :: Val{false}, fdw :: FiniteDiffWrapper, x :: RVec, ℓ :: Int )
    return FD.finite_difference_gradient( _eval_handle(fdw.objf, ℓ), x );
end
=#

function get_gradient( fdw :: FiniteDiffWrapper, x :: RVec, ℓ :: Int = 1)
    return FD.finite_difference_gradient( _eval_handle(fdw.objf, ℓ), x );
end

# NOTE this is clearly is not optimized for multiple outputs 
# (and atm is not meant to be optimized)
# one could differentiate the gradients here and use the same evaluation sites 
# for all hessians
function get_hessian( fdw :: FiniteDiffWrapper, x :: RVec, ℓ :: Int = 1 )
    return FD.finite_difference_hessian(  _eval_handle( adw.objf, ℓ), x );
end
