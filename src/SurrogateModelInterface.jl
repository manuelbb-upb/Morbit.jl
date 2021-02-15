# each surrogate model type is configured via a data structure 
# that inherits from SurrogateConfig.
# such a configuration struct should be exported my the module 
abstract type SurrogateConfig end

# make a configuration broadcastable
Broadcast.broadcastable( sc::SurrogateConfig ) = Ref(sc);

# Methods to be implemented by each type inheriting from SurrogateConfig
max_evals( :: SurrogateConfig ) = typemax(Int) :: Int;

# Abstract super type for meta data that is collected 
# during build process
abstract type SurrogateMeta end

# return data that is stored in iter data in each iteration
saveable(::SurrogateMeta) = nothing :: Union{Nothing, SurrogateMeta};

# Abstract super type for the actual surrogate models
abstract type SurrogateModel end

fully_linear( :: SurrogateModel ) = false :: Bool;

prepare!( :: AbstractObjective, :: SurrogateConfig, ::AbstractConfig ) = nothing :: Nothing;

# build the model using information from algorithm configuration,
# the objective function and the configuration; 
# `crit_flag` if we are in the criticality loop 
function build_model( cfg :: AbstractConfig, objf :: AbstractObjective, 
    ::SurrogateConfig, crit_flag :: Bool ) :: Tuple{SurrogateModel,SurrogateMeta}
    nothing
end

