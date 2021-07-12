# Types to help with differentiation of Abstract Objectives 
# Needs `AbstractMOP` and `AbstractObjective` to be defined.
# `TransformerFn` is also needed.
# Note: Differatiates the objective handle for scaled input.
struct GradWrapper{
        TT<:TransformerFn, 
        FT<:Union{AbstractVector{<:Function},Nothing}, 
        JT<:Union{Function, Nothing}
    } <: DiffFn
    tfn :: TT
    list_of_grads :: FT
    jacobian_handle :: JT   # JT either Nothing or <:Function
end

GradWrapper(tfn :: TransformerFn, g :: Function, jh :: Union{Nothing, Function} ) = GradWrapper(tfn, [g,], jh );

# true =̂ gw.list_of_grads not empty
function _get_gradient( :: Val{true}, gw :: GradWrapper, x̂ :: Vec, ℓ :: Int = 1)
    Jtfn = _jacobian_unscaling( gw.tfn, x̂ )
    g = gw.list_of_grads[ℓ](gw.tfn(x̂))
    @assert length(g) == length(x̂) "Gradient $(ℓ) must have length $(length(x̂))."
    return vec(Jtfn'g);
end 

# false =̂ list_of_grads empty but jacobian handle provided
function _get_gradient( :: Val{false}, gw :: GradWrapper, x̂ :: Vec, ℓ :: Int = 1)
    Jtfn = _jacobian_unscaling( gw.tfn, x̂ )
    return vec( Jtfn'gw.jacobian_handle( gw.tfn(x̂) )[ℓ, :] );
end

function get_gradient( gw :: GradWrapper, x̂ :: Vec, ℓ :: Int = 1)
    return _get_gradient( Val(!isempty(gw.list_of_grads)), gw, x̂, ℓ );
end

# prefer user defined handle over gradient iteration
function get_jacobian( gw :: GradWrapper, x̂ :: Vec )
    Jtfn = _jacobian_unscaling( gw.tfn, x̂ )
    return Jtfn'gw.jacobian_handle( gw.tfn(x̂) )
end

function get_jacobian( gw :: GradWrapper{TT,FT,JT}, x̂ :: Vec ) where{TT,FT,JT<:Nothing}
    return transpose( hcat( 
        ( _get_gradient( Val(true), gw, x̂, ℓ ) for ℓ = eachindex(gw.list_of_grads) )...
    ))    
end

struct AutoDiffWrapper{ 
        O <: AbstractObjective,
        TT <: TransformerFn,
        JT <: Union{Nothing,Function} 
    } <: DiffFn
    objf :: O
    tfn :: TT
    jacobian :: JT
end

# no jacobian handle set
function get_jacobian( adw :: AutoDiffWrapper{O,TT,JT}, x̂ :: Vec ) where{O,TT,JT<:Nothing}
    return AD.jacobian( eval_handle(adw.objf) ∘ adw.tfn, x̂ );
end

# jacobian handle set
function get_jacobian( adw :: AutoDiffWrapper{O,TT,JT}, x̂ :: Vec ) where{O,TT,JT<:Function}
    Jtfn = _get_transformer_jacobian( adw, x̂ )
    return Jtfn'gw.jacobian_handle( adw.tfn(x̂) )
end

function get_gradient( adw :: AutoDiffWrapper, x̂ :: Vec, ℓ :: Int = 1)
    return AD.gradient( x -> eval_handle(adw.objf)( adw.tfn(x) )[ℓ], x̂ )
end  

# not optimized for multiple outputs
function get_hessian( adw :: AutoDiffWrapper, x̂ :: Vec, ℓ :: Int = 1)
    return AD.hessian( x -> eval_handle(adw.objf)( adw.tfn(x) )[ℓ], x̂ )
end

struct FiniteDiffWrapper{
        O<:AbstractObjective,
        TT<:AbstractMOP,
        JT<:Function
    } <: DiffFn
    objf :: O
    tfn :: TT
    jacobian :: JT
end

# no jacobian handle set
function get_jacobian( fdw :: FiniteDiffWrapper{O,TT,JT}, x̂ :: Vec ) where{O,TT,JT<:Nothing}
    return FD.finite_difference_jacobian( x -> eval_handle(fdw.objf)( adw.tfn(x) ) , x̂ )
end

# jacobian handle set
function get_jacobian( fdw :: FiniteDiffWrapper{O,TT,JT}, x̂ :: Vec ) where{O,TT,JT<:Function}
    Jtfn = _jacobian_unscaling( fdw.tfn, x̂ )
    return Jtfn'gw.jacobian_handle( fdw.tfn(x̂) )
end

function get_gradient( fdw :: FiniteDiffWrapper, x̂ :: Vec, ℓ :: Int = 1)
    return FD.finite_difference_gradient( x -> eval_handle(fdw.objf)( adw.tfn(x) )[ℓ], x̂ );
end

# NOTE this is clearly is not optimized for multiple outputs 
# (and atm is not meant to be optimized)
# one could differentiate the gradients here and use the same evaluation sites 
# for all hessians
function get_hessian( fdw :: FiniteDiffWrapper, x̂ :: Vec, ℓ :: Int = 1 )
    return FD.finite_difference_hessian(x -> eval_handle(fdw.objf)( adw.tfn(x) )[ℓ], x̂ );
end

###### Hessians ########################

struct HessWrapper{TT<:TransformerFn, FT<:AbstractVector{<:Function}} <: DiffFn
    tfn :: TT
    list_of_hess_fns :: FT
end

HessWrapper(tfn :: TransformerFn, hf :: Function ) = HessWrapper(tfn, [hf,] );

function get_hessian( hw :: HessWrapper, x̂ :: Vec, ℓ :: Int = 1)
    Jtfn = _jacobian_unscaling( hw.tfn, x̂ )
    return Jtfn'hw.list_of_hess_fns[ℓ](hw.tfn(x̂))*Jtfn;
end

struct HessFromGrads{GW <: GradWrapper} <: DiffFn 
    gw :: GW
    method :: Symbol
end

#@memoize ThreadSafeDict 
function _get_grad_func( gw :: GradWrapper, ℓ :: Int) :: Function
    return x -> _get_gradient( gw, x, ℓ)
end

function get_hessian( hw :: HessFromGrads, x̂ :: Vec, ℓ :: Int = 1 )
    if hw.method == :autodiff
        return AD.jacobian( _get_grad_func( hw.gw, ℓ), x̂ )
    elseif hw.method == :fdm
        return FD.finite_difference_jacobian( _get_grad_func( hw.gw, ℓ), x̂ )
    end
end
