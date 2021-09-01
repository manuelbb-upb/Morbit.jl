# make a configuration broadcastable
Broadcast.broadcastable( sm::SurrogateModel ) = Ref(sm);
Broadcast.broadcastable( sc::SurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from SurrogateConfig
max_evals( :: SurrogateConfig ) = typemax(Int) :: Int;

# return data that is stored in iter data in each iteration
# get_saveable(::SurrogateMeta) = nothing :: Union{Nothing, <:SurrogateMeta};

fully_linear( :: SurrogateModel ) = false :: Bool;

# can objective functions with same configuration types be combined 
# to a new vector objective?
combinable( :: SurrogateConfig ) = false :: Bool    

# TODO make combinable bi-variate to check for to concrete configs if they are combinable


## TODO: make `prepare_init_model` and `_init_model` have a `ensure_fully_linear` kwarg too
function prepare_init_model( ::SurrogateConfig, :: AbstractObjective, :: AbstractMOP, 
    :: AbstractIterData, ::AbstractDB, :: AbstractConfig; kwargs... ) :: SurrogateMeta 
    nothing
end

function _init_model( ::SurrogateConfig, :: AbstractObjective, :: AbstractMOP, 
    :: AbstractIterData, ::AbstractDB, :: AbstractConfig, :: SurrogateMeta; kwargs... ) :: Tuple{<:SurrogateModel,<:SurrogateMeta}
    nothing 
end

## TODO: Allow to pass a SurrogateConfig here as well. (ATM use `model_cfg(objf)`) #src
## In general, the function signatures are somewhat messy. We should unify them a bit. #src
function update_model( :: SurrogateModel, :: AbstractObjective, :: SurrogateMeta, :: AbstractMOP, 
    :: AbstractIterData, :: AbstractDB, :: AbstractConfig; kwargs... ) :: Tuple{<:SurrogateModel,<:SurrogateMeta}
    nothing 
end

eval_models( :: SurrogateModel, ::Vec ) ::Vec = nothing 
get_gradient( :: SurrogateModel, ::Vec, :: Int ) :: Vec = nothing
get_jacobian( :: SurrogateModel, :: Vec ) :: Mat = nothing 

# DEFAULTS

get_saveable_type( :: SurrogateMeta ) = Nothing 
get_saveable( :: SurrogateMeta ) = nothing

prepare_update_model( mod, objf, meta, mop, iter_data, db, algo_config; kwargs...) = meta
prepare_improve_model( mod, objf, meta, mop, iter_data, db, algo_config; kwargs...) = meta

# overwrite if possible, this is inefficient:
eval_models( sm :: SurrogateModel, x̂ :: Vec, ℓ :: Int) = eval_models(sm, x̂)[ℓ]

function improve_model( mod :: SurrogateModel, objf:: AbstractObjective, meta :: SurrogateMeta,
    mop :: AbstractMOP, id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig;
    ensure_fully_linear = false)
    return mod, meta
end

# check if surrogate configurations are equal (only really needed if combinable)
function Base.:(==)( cfg1 :: T, cfg2 :: T ) where T <: SurrogateConfig
    all( getfield(cfg1, fname) == getfield(cfg2, fname) for fname ∈ fieldnames(T) )
end

function Base.:(==)( cfg1 :: T, cfg2 :: F ) where {T <: SurrogateConfig, F<:SurrogateConfig}
    false 
end

# only needed if combinable
combine( cfg :: SurrogateConfig, :: SurrogateConfig  ) = cfg

## derived 

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `model`.
That is, if `model` is a surrogate for two scalar objectives, then `ℓ` must 
be either 1 or 2.
"""
function _get_optim_handle( model :: SurrogateModel, ℓ :: Int )
    # Return an anonymous function that modifies the gradient if present
    function (x :: Vec, g :: Vec)
        if !isempty(g)
            g[:] = get_gradient( model, x, ℓ)
        end
        return eval_models( model, x, ℓ)
    end
end