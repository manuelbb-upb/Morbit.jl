```@meta
EditURL = "<unknown>/src/Interfaces.jl"
```

# Interfaces

This file defines abstract super types and includes further method
definitions for these types so we can write our algorithm without
the actual implementations being available.
This also avoids some import order complications.

## Abstract Super Types

### Surrogate Models
Each surrogate model type is configured via a data structure
that implements `SurrogateConfig`.
Such a configuration struct should be exported by the main module.

````julia
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
````

Define a shorthand for the type of what we save in some `AbstractIterSaveable`.

````julia
const ContainerSaveable = Union{Nothing,Tuple{Vararg{<:Union{Nothing,SurrogateMeta}}}}
````

### MOPs
Internally we use implementations of the `AbstractObjective` for
managing the MOP objectives and evaluation.

````julia
"Abstract super type for any kind of (vector) objective."
abstract type AbstractObjective <: MOI.AbstractVectorFunction end
````

Our actual problem is repersented by an `AbstractMOP`.

````julia
"""
    AbstractMOP{T}

Abstract super type for multi-objective optimization problems.
`T` is `true` if the problem is modifyable and `false elsewise.

The user should define a `MixedMOP<:AbstractMOP{true}`, see [`MixedMOP`](@ref).
"""
abstract type AbstractMOP{T} <: MOI.ModelLike end
````

### Internal Data Managment

For internal data management we make some effort to keep everything
structured.
To save the intermediate results:

````julia
"Abstract super type for stuff stored in the database."
abstract type AbstractResult{XT <: VecF, YT <: VecF} end
````

`AbstractIterData` is passed to functions and provides access to
the current site and value vectors and the trust region radius.

````julia
"Abstract super type for iteration data. Implemented by `IterData`.
Has type parameters for the vector type of itaration site, value vector and trust region radius."
abstract type AbstractIterData{XT <: VecF, YT <: VecF, DT <: NumOrVecF} end
"Abstract super type for some saveable representation of `AbstractIterData`."
abstract type AbstractIterSaveable{XT <: VecF, YT <: VecF, DT <: NumOrVecF, C <: ContainerSaveable } end
````

A shorthand for everything that is either nothing or an `AbstractIterSaveable`:

````julia
const NothingOrSaveable = Union{Nothing, AbstractIterSaveable}
````

Everything is kept in a database:

````julia
"Abstract database super type. Implemented by `ArrayDB` and `MockDB`."
abstract type AbstractDB{R<:AbstractResult, S<:NothingOrSaveable} end
````

### Algorithm configuration.

We have our own type to define the solution of the trust region sub-problems:

````julia
"Abstract super type for descent step configuration."
abstract type AbstractDescentConfig end
````

This might be returned by the general algorithm configuration implementing `AbstractConfig`:

````julia
"Abstract super type for user configurable algorithm configuration."
abstract type AbstractConfig end
````

### Differentiation

````julia
"Super type for making (automatic) differentiation comfortable."
abstract type DiffFn end
````

Method definitions for `DiffFn`s:

````julia
get_gradient( :: DiffFn, :: Vec, :: Int ) :: Vec = nothing
get_jacobian( :: DiffFn, :: Vec ) :: Mat = nothing
get_hessian( :: DiffFn, :: Vec, :: Int ) :: Mat = nothing
````

### Enums

These codes should be availabe everywhere and that is why we
define them here:

````julia
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
````

### Interface Definitions
Most of the interfaces are defined in subfiles and we include
them here:

````julia
include("SurrogateModelInterface.jl");
include("AbstractSurrogateContainerInterface.jl")
include("AbstractObjectiveInterface.jl");
include("AbstractMOPInterface.jl");
include("AbstractIterDataInterface.jl")
include("AbstractResultInterface.jl")
include("AbstractDBInterface.jl")
include("AbstractConfigInterface.jl")
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

