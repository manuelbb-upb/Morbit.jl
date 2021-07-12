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

# T = true if MOP is modifyable
abstract type AbstractMOP{T} <: MOI.ModelLike end;

abstract type AbstractDB{F<:AbstractFloat} end;

abstract type AbstractIterData{F<:AbstractFloat} end;
abstract type AbstractIterSaveable{F<:AbstractFloat} end;

abstract type AbstractDescentConfig end

abstract type AbstractConfig{F<:AbstractFloat} end;

abstract type DiffFn end;

get_gradient( :: DiffFn, :: Vec, :: Int ) :: Vec = nothing
get_jacobian( :: DiffFn, :: Vec ) :: Mat = nothing
get_hessian( :: DiffFn, :: Vec, :: Int ) :: Mat = nothing;

abstract type AbstractResult{F<:AbstractFloat} end;

# classify the iterations 
# mainly for user information collection 
@enum ITER_TYPE begin
    ACCEPTABLE = 1;     # accept trial point, shrink radius 
    SUCCESSFULL = 2;    # accept trial point, grow radius 
    MODELIMPROVING = 3; # reject trial point, keep radius 
    INACCEPTABLE = 4;   # reject trial point, shrink radius (much)
end

@enum STOP_CODE begin
    CONTINUE = 1
    MAX_ITER = 2
    BUDGET_EXHAUSTED = 3
    CRITICAL = 4
    TOLERANCE = 5 
end

include("SurrogateModelInterface.jl");
include("AbstractObjectiveInterface.jl");
include("AbstractMOPInterface.jl");
include("AbstractConfigInterface.jl")
include("AbstractResultInterface.jl")
include("AbstractDBInterface.jl")
