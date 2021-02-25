# each surrogate model type is configured via a data structure 
# that inherits from SurrogateConfig.
# such a configuration struct should be exported my the module 
abstract type SurrogateConfig end

# Abstract super type for meta data that is collected 
# during build process
abstract type SurrogateMeta end

# Abstract super type for the actual surrogate models
abstract type SurrogateModel end

abstract type AbstractObjective <: MOI.AbstractVectorFunction end;

abstract type AbstractMOP <: MOI.ModelLike end;

abstract type AbstractDB end;
abstract type AbstractIterData end;
abstract type AbstractConfig end;

abstract type DiffFn end;

abstract type Result end;

get_gradient( :: DiffFn, :: RVec, :: Int ) = nothing :: RVec;
get_jacobian( :: DiffFn, :: RVec ) = nothing :: RMat;
get_hessian( :: DiffFn, :: RVec, :: Int ) = nothing :: RMat;

include("SurrogateModelInterface.jl");
include("AbstractObjectiveInterface.jl");
include("AbstractMOPInterface.jl");

