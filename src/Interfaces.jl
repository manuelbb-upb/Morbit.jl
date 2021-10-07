# # Interfaces

# This file defines abstract super types and includes further method 
# definitions for these types so we can write our algorithm without 
# the actual implementations being available.
# This also avoids some import order complications.

# ## Abstract Super Types

# ### Surrogate Models
# Each surrogate model type is configured via a data structure 
# that implements `SurrogateConfig`.
# Such a configuration struct should be exported by the main module. 
"Abstract super type for a configuration type defining some surrogate model."
abstract type SurrogateConfig end

"Abstract super type for meta data that is used to build a model."
abstract type SurrogateMeta end

"Abstract super type for the actual surrogate models."
abstract type SurrogateModel end

"Abstract super type wrapping around an objective, its model and the surrogate meta."
abstract type AbstractSurrogateWrapper end

"Wrapper around a list of `AbstractSurrogateWrapper`s."
abstract type AbstractSurrogateContainer end

# ### MOPs
# Internally we use implementations of the `AbstractVecFun` for 
# managing the MOP objectives and evaluation.
"Abstract super type for any kind of (vector) objective."
abstract type AbstractVecFun{C <: SurrogateConfig} <: Function end

# Our actual problem is repersented by an `AbstractMOP`.
"""
    AbstractMOP{T}

Abstract super type for multi-objective optimization problems.
`T` is `true` if the problem is modifyable and `false elsewise.

The user should define a `MixedMOP<:AbstractMOP{true}`, see [`MixedMOP`](@ref).
"""
abstract type AbstractMOP{T} end

# ### Internal Data Managment

# For internal data management we make some effort to keep everything 
# structured.
# To save the intermediate results:
"Abstract super type for stuff stored in the database."
abstract type AbstractResult{XT <: VecF, YT <: VecF} end

# `AbstractIterData` is passed to functions and provides access to 
# the current site and value vectors and the trust region radius.
"Abstract super type for iteration data. Implemented by `IterData`. 
Has type parameters for the vector type of itaration site, value vector, constraint vectors and trust region radius."
abstract type AbstractIterData{XT <: VecF, YT <: VecF, ET <: VecF, IT <: VecF, DT <: NumOrVecF} end
"Abstract super type for some saveable representation of `AbstractIterData`."
abstract type AbstractIterSaveable{XT <: VecF, YT <: VecF, 
    ET <: VecF, IT <: VecF, DT <: NumOrVecF } end

# A shorthand for everything that is either nothing or an `AbstractIterSaveable`:
const NothingOrSaveable = Union{Nothing, AbstractIterSaveable}
const NothingOrMeta = Union{Nothing, SurrogateMeta}

# Everything is kept in a database:
"Abstract database super type. Implemented by `ArrayDB` and `MockDB`."
abstract type AbstractDB{R<:AbstractResult, S<:NothingOrMeta} end
abstract type AbstractSuperDB end

# ### Algorithm configuration.

# We have our own type to define the solution of the trust region sub-problems:
"Abstract super type for descent step configuration."
abstract type AbstractDescentConfig end

# This might be returned by the general algorithm configuration implementing `AbstractConfig`:
"Abstract super type for user configurable algorithm configuration."
abstract type AbstractConfig end

abstract type AbstractFilter end

abstract type AbstractVarScaler end

# ### Enums

# These codes should be availabe everywhere and that is why we 
# define them here:

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

# ### Interface Definitions
# Most of the interfaces are defined in subfiles and we include
# them here:
include("SurrogateModelInterface.jl");
include("AbstractSurrogateContainerInterface.jl")
include("AbstractVecFunInterface.jl");
include("AbstractMOPInterface.jl");
include("AbstractIterDataInterface.jl")
include("AbstractResultInterface.jl")
include("AbstractDBInterface.jl")
include("AbstractConfigInterface.jl")
include("AbstractFilterInterface.jl")
