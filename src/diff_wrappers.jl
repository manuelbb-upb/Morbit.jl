# Types to help with differentiation of Abstract Objectives 
# Note: Differatiates the objective handle for scaled input

struct HessWrapper <: DiffFn
    mop :: AbstractMOP;
    list_of_hess_fns :: Vector{<:Function};
end
HessWrapper(mop :: AbstractMOP, hf :: Function ) = HessWrapper(mop, [hf,] );

struct GradWrapper <: DiffFn
    mop :: AbstractMOP;
    list_of_grads :: Vector{<:Function}
    jacobian_handle :: Union{Nothing, Function} 
end

GradWrapper(mop :: AbstractMOP, g :: Function, jh :: Union{Nothing, Function} ) = GradWrapper(mop, [g,], jh );
GradWrapper(mop :: AbstractMOP, g :: Nothing, jh :: Union{Nothing, Function} ) = GradWrapper(mop, Function[], jh );

function _get_transformer_jacobian( gw :: Union{HessWrapper,GradWrapper}, x̂ :: RVec) :: RMat 
    tfn = _get_transformer_fn(gw.mop);
    return _jacobian_unscaling( tfn, x̂ )
end

# true =̂ gw.list_of_grads not empty
function _get_gradient( :: Val{true}, gw :: GradWrapper, x̂ :: RVec, ℓ :: Int = 1)
    J = _get_transformer_jacobian( gw, x̂ );
    return vec(J'gw.list_of_grads[ℓ](unscale(x̂,gw.mop)));
end 
# false =̂ list_of_grads empty but jacobian handle provided
function _get_gradient( :: Val{false}, gw :: GradWrapper, x̂ :: RVec, ℓ :: Int = 1)
    J = _get_transformer_jacobian( gw, x̂ );
    return vec( J'gw.jacobian_handle(unscale(x̂, gw.mop))[ℓ, :] );
end

function get_gradient( gw :: GradWrapper, x̂ :: RVec, ℓ :: Int = 1)
    return _get_gradient( Val(!isempty(gw.list_of_grads)), gw, x̂, ℓ );
end

function get_jacobian( gw :: GradWrapper, x̂ :: RVec )
    # prefer user defined handle over gradient iteration
    if !isnothing(gw.jacobian_handle);
        J = _get_transformer_jacobian( gw, x̂ );
        return J'gw.jacobian_handle( unscale(x̂, gw.mop) )
    else
        return transpose( hcat( 
            ( _get_gradient( Val(true), gw, x̂, ℓ ) for ℓ = eachindex(gw.list_of_grads) )...
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
    return FD.finite_difference_hessian( _eval_handle( fdw.objf, ℓ), x );
end

###### Hessians ########################

function get_hessian( hw :: HessWrapper, x̂ :: RVec, ℓ :: Int = 1)
    J = _get_transformer_jacobian( hw, x̂ ); # does only work for affine linear variable scaling!!
    return J'hw.list_of_hess_fns[ℓ](unscale( x̂, hw.mop))*J;
end

struct HessFromGrads <: DiffFn 
    gw :: GradWrapper
    method :: Symbol
end

@memoize ThreadSafeDict function _get_grad_func( gw :: GradWrapper, ℓ :: Int) :: Function
    return x -> _get_gradient( gw, x, ℓ)
end

function get_hessian( hw :: HessFromGrads, x :: RVec, ℓ :: Int = 1 )
    if hw.method == :autodiff
        return AD.jacobian( _get_grad_func( hw.gw, ℓ), x )
    elseif hw.method == :fdm
        return FD.finite_difference_jacobian( _get_grad_func( hw.gw, ℓ), x )
    end
end

