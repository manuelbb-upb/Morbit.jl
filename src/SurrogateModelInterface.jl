# make a configuration broadcastable
Broadcast.broadcastable( sm::SurrogateModel ) = Ref(sm);
Broadcast.broadcastable( sc::SurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from SurrogateConfig
max_evals( :: SurrogateConfig ) ::Int = typemax(Int)

# return data that is stored in iter data in each iteration
# get_saveable(::SurrogateMeta) = nothing :: Union{Nothing, <:SurrogateMeta};

fully_linear( :: SurrogateModel ) :: Bool = false

set_fully_linear!( :: SurrogateModel, :: Bool ) = nothing

# can objective functions with same configuration types be combined 
# to a new vector objective?
combinable( :: SurrogateConfig ) :: Bool = false

needs_gradients( :: SurrogateConfig ) :: Bool = false
needs_hessians( :: SurrogateConfig ) :: Bool = false 

# TODO make combinable bi-variate to check for to concrete configs if they are combinable

## TODO: make `prepare_init_model` and `_init_model` have a `ensure_fully_linear` kwarg too
function prepare_init_model( model_cfg, func_indices, mop, scal, x_it, sdb, algo_config; kwargs...) :: SurrogateMeta
    #=::SurrogateConfig, ::FunctionIndexIterable, :: AbstractMOP,
    :: AbstractVarScaler,
    :: AbstractIterate, ::AbstractSuperDB, :: AbstractConfig; kwargs... ) :: SurrogateMeta =#
    nothing
end

function init_model( meta, model_cfg, func_indices, mop, scal, x_it, sdb, algo_config; kwargs...) :: Tuple{<:SurrogateModel,<:SurrogateMeta}
    #=::SurrogateMeta, ::SurrogateConfig, FunctionIndexIterable, :: AbstractMOP, 
    :: AbstractVarScaler,
    :: AbstractIterate, ::AbstractSuperDB, :: AbstractConfig; kwargs... ) :: Tuple{<:SurrogateModel,<:SurrogateMeta}
    =#
    nothing 
end

function update_model( mod, meta, model_cfg, func_indices, mop, scal, x_it, sdb, algo_config; kwargs... )
    #= mod :: SurrogateModel, meta ::SurrogateMeta, ::SurrogateConfig, :: FunctionIndexIterable, 
    :: AbstractMOP, :: AbstractVarScaler,
    :: AbstractIterate, :: AbstractSuperDB, :: AbstractConfig; kwargs... ) :: Tuple{<:SurrogateModel,<:SurrogateMeta}
    =#
    mod, meta 
end

eval_models( :: SurrogateModel, :: AbstractVarScaler, ::Vec ) ::Vec = nothing 
get_gradient( :: SurrogateModel, :: AbstractVarScaler, ::Vec, :: Int ) :: Vec = nothing
get_jacobian( :: SurrogateModel, :: AbstractVarScaler, :: Vec ) :: Mat = nothing 

# DEFAULTS

get_saveable_type( :: SurrogateConfig, x, y ) = Nothing
get_saveable( :: SurrogateMeta ) = nothing

prepare_update_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = meta
prepare_improve_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = meta

# overwrite if possible, this is inefficient:
eval_models( sm :: SurrogateModel, scal :: AbstractVarScaler, x_scaled :: Vec, ℓ :: Int) = eval_models(sm, scal, x_scaled)[ℓ]

improve_model( mod, meta, cfg, func_indices, mop, scal, iter_data, db, algo_config; kwargs...) = mod, meta

# check if surrogate configurations are equal (only really needed if combinable)
function Base.:(==)( cfg1 :: T, cfg2 :: T ) where T <: SurrogateConfig
    all( getfield(cfg1, fname) == getfield(cfg2, fname) for fname ∈ fieldnames(T) )
end

function Base.:(==)( cfg1 :: T, cfg2 :: F ) where {T <: SurrogateConfig, F<:SurrogateConfig}
    false 
end

## derived 

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `model`.
That is, if `model` is a surrogate for two scalar objectives, then `ℓ` must 
be either 1 or 2.
"""
function _get_optim_handle( model :: SurrogateModel, scal :: AbstractVarScaler, ℓ :: Int )
    # Return an anonymous function that modifies the gradient if present
    function (x :: Vec, g :: Vec)
        if !isempty(g)
            g[:] = get_gradient( model, scal, x, ℓ)
        end
        return eval_models( model, scal, x, ℓ )
    end
end