# # Interfaces

# This file defines abstract super types,
# mainly to avoid some import order complications.

# ## Abstract Super Types

# ### Surrogate Models
# Each surrogate model type is configured via a data structure 
# that implements `AbstractSurrogateConfig`.
# Such a configuration struct should be exported by the main module. 
"Abstract super type for a configuration type defining some surrogate model."
abstract type AbstractSurrogateConfig end

"Abstract super type for meta data that is used to build a model."
abstract type AbstractSurrogateMeta end

"Abstract super type for the actual surrogate models."
abstract type AbstractSurrogate end

# ### MOPs
# Internally we use implementations of the `AbstractVecFun` for 
# managing the MOP objectives and evaluation.
"Abstract super type for any kind of (vector) objective."
abstract type AbstractVecFun <: Function end

# Our actual problem is repersented by an `AbstractMOP`.
"""
    AbstractMOP{T}

Abstract super type for multi-objective optimization problems.
`T` is `true` if the problem is modifyable and `false elsewise.

The user should define a `MOP<:AbstractMOP{true}`, see [`MOP`](@ref).
"""
abstract type AbstractMOP{T} end

# ### Internal Data Managment

# For internal data management we make some effort to keep everything 
# structured.

# `AbstractIterate` stores the current site and value vectors as well as the 
# trust region radius for meta models.
"Abstract super type for a container that stores the site and value vectors."
abstract type AbstractIterate end

"Abstract super type for some saveable iteration information."
abstract type AbstractIterSaveable end 

# A shorthand for everything that is either nothing or an `AbstractSurrogateMeta`:
const NothingOrMeta = Union{Nothing, AbstractSurrogateMeta}

# Everything is kept in databases:
abstract type AbstractResult end
"Abstract database super type. Implemented by `ArrayDB` and `MockDB`."
abstract type AbstractDB{R <: AbstractResult, S<:NothingOrMeta} end

# ### Algorithm configuration.

# We have our own type to define the solution of the trust region sub-problems:
"Abstract super type for descent step configuration."
abstract type AbstractDescentConfig end

# This might be returned by the general algorithm configuration implementing `AbstractConfig`:
"Abstract super type for user configurable algorithm configuration."
abstract type AbstractConfig end

#src #TODO documentation
abstract type AbstractSurrogateContainer end

abstract type AbstractFilter end

abstract type AbstractVarScaler end

abstract type DiffFn end