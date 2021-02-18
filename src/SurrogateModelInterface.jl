# make a configuration broadcastable
Broadcast.broadcastable( sm::SurrogateModel ) = Ref(sm);
Broadcast.broadcastable( sc::SurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from SurrogateConfig
max_evals( :: SurrogateConfig ) = typemax(Int) :: Int;
max_evals!( :: SurrogateConfig, :: Int ) = 0 :: Nothing;

# return data that is stored in iter data in each iteration
# saveable(::SurrogateMeta) = nothing :: Union{Nothing, <:SurrogateMeta};

fully_linear( :: SurrogateModel ) = false :: Bool;

# prepare!( :: SurrogateConfig, :: AbstractObjective ) = nothing :: Nothing;

# can objective functions with same configuration types be combined 
# to a new vector objective
combinable( :: SurrogateConfig ) = false :: Bool     

init_model( :: AbstractObjective, :: AbstractMOP, ac :: Any ) = nothing :: Tuple{<:SurrogateModel,<:SurrogateMeta};
update_model( :: AbstractObjective, :: AbstractMOP, ac :: Any, ::Bool ) = nothing :: Tuple{<:SurrogateModel,<:SurrogateMeta};

eval_models( :: SurrogateModel, ::RVec ) = nothing :: RVec
get_gradient( :: SurrogateModel, ::RVec, :: Int ) = nothing :: RVec
get_jacobian( :: SurrogateModel, :: RVec ) = nothing :: RMat

# DEFAULTS

# check if surrogate configurations are equal (only really needed if combinable)
function Base.:(==)( cfg1 :: T, cfg2 :: T ) where T <: SurrogateConfig
    all( getfield(cfg1, fname) == getfield(cfg2, fname) for fname in fieldnames(C) )
end

function Base.:(==)( cfg1 :: T, cfg2 :: F ) where {T <: SurrogateConfig, F<:SurrogateConfig}
    false 
end

# only needed if combinable
combine( :: SurrogateConfig, :: SurrogateConfig  ) = nothing :: SurrogateConfig;
