# make a configuration broadcastable
Broadcast.broadcastable( sm::SurrogateModel ) = Ref(sm);
Broadcast.broadcastable( sc::SurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from SurrogateConfig
max_evals( :: SurrogateConfig ) = typemax(Int) :: Int;
max_evals!( :: SurrogateConfig, :: Int ) = 0 :: Nothing; # TODO do we still use this anywhere?

# return data that is stored in iter data in each iteration
# saveable(::SurrogateMeta) = nothing :: Union{Nothing, <:SurrogateMeta};

fully_linear( :: SurrogateModel ) = false :: Bool;

# can objective functions with same configuration types be combined 
# to a new vector objective?
combinable( :: SurrogateConfig ) = false :: Bool     

_init_model( ::SurrogateConfig, :: AbstractObjective, :: AbstractMOP, :: AbstractIterData ) = nothing :: Tuple{<:SurrogateModel,<:SurrogateMeta};
update_model( :: SurrogateModel,:: AbstractObjective, :: SurrogateMeta, :: AbstractMOP, :: AbstractIterData) = nothing :: Tuple{<:SurrogateModel,<:SurrogateMeta};

eval_models( :: SurrogateModel, ::RVec ) = nothing :: RVec
get_gradient( :: SurrogateModel, ::RVec, :: Int ) = nothing :: RVec
get_jacobian( :: SurrogateModel, :: RVec ) = nothing :: RMat

# DEFAULTS

function init_model( objf:: AbstractObjective, args...)
    _init_model( model_cfg(objf), objf, args...)
end

# overwrite, this is inefficient
eval_models( sm :: SurrogateModel, x̂ :: RVec, ℓ :: Int) = eval_models(sm, x̂)[ℓ]
function improve_model!( mod :: SurrogateModel, objf:: AbstractObjective, meta :: SurrogateMeta,
    mop :: AbstractMOP, id :: AbstractIterData;
    ensure_fully_linear) :: Tuple{SurrogateModel,SurrogateMeta}
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
combine( :: SurrogateConfig, :: SurrogateConfig  ) = nothing :: SurrogateConfig;
