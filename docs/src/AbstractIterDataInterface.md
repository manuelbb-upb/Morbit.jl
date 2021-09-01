```@meta
EditURL = "<unknown>/../src/AbstractIterDataInterface.jl"
```

# Iter Data Interface

Types (and instances of those types) that implement the `AbstractIterData`
interface are used to have a unified access to relevant iteration data in
all functions of the main algorithm.

First of all, we make it so that any `AbstractResult` is broadcasted wholly:

````julia
Base.broadcastable( id :: AbstractIterData ) = Ref( id )
````

## Mandatory Methods

Any subtype of `AbstractIterData` should implement these methods.

First, we need a constructor.
This constructor should take a base type implementing `AbstractIterData` and
its first states as arguments, where `x` and `fx` are vectors of floats and `Δ`
is either a float or a vector thereof:

````julia
function _init_iter_data( T :: Type{<:AbstractIterData}, x :: VecF, fx :: VecF, Δ :: NumOrVecF )
    return nothing
end
````

### Getters

There are Getters for the mathematical objects relevant during optimzation:

````julia
"Return current iteration site vector ``xᵗ``."
function get_x( :: AbstractIterData{XT,YT,DT} ) where {XT,YT,DT}
    return XT()
end

"Return current value vector ``f(xᵗ)``."
function get_fx(  :: AbstractIterData{XT,YT,DT} ) where {XT,YT,DT}
    return YT()
end

"Return current trust region radius (vector) ``Δᵗ``."
function get_delta( :: AbstractIterData{XT,YT,DT} ) where {XT,YT,DT}
    return zero(eltype(DT))
end
````

We also need the iteration result index for our database.
This should be implemented but works as is if only `MockDB` is used.

````julia
"Index (or `id`) of current iterate in database."
get_x_index( :: AbstractIterData ) :: Int = -1
````

For printing information and for stopping we need the iteration count:

````julia
"Return number of iterations so far."
get_num_iterations( :: AbstractIterData ) :: Int = 0
"Return the number of model improvement iterations so far."
get_num_model_improvements( :: AbstractIterData ) :: Int = 0
````

Finally, we also want to display the last iteration classification:

````julia
"Return the iteration classification of `ITER_TYPE`."
it_stat( :: AbstractIterData ) :: ITER_TYPE = SUCCESSFULL
````

### Setters

Implement the setters and note the leading underscore!
The algorithm will actually call `set_x!` instead of `_set_x!` etc.
These derived methods are implemented below and ensure that we actually
store copies of the input!

````julia
"Set current iteration site to `x̂`."
_set_x!( :: AbstractIterData, x̂ :: Vec ) :: Nothing = nothing

"Set current iteration value vector to `x̂`."
_set_fx!( :: AbstractIterData, ŷ :: Vec ) :: Nothing = nothing

"Set current trust region radius (vector?) to `Δ`."
_set_delta!( :: AbstractIterData, Δ :: NumOrVec ) :: Nothing = nothing

"Set the current iteration database id to `val`."
set_x_index!( :: AbstractIterData, val :: Int ) :: Nothing = nothing
````

The itaration counters are modified by the following methods:

````julia
"Set the iteration counter to `N`."
set_num_iterations!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing

"Set the improvement counter to `N`."
set_num_model_improvements!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing ;
nothing #hide
````

Of course, we also need a way to set the iteration classification:

````julia
"Set the iteration classification."
it_stat!( :: AbstractIterData, :: ITER_TYPE ) :: Nothing = nothing;
nothing #hide
````

## Derived Methods

The actual setters simply ensure proper copying:

````julia
set_x!( id :: AbstractIterData, x :: Vec ) = _set_x!(id, copy(x))
set_fx!( id :: AbstractIterData, fx :: Vec ) = _set_fx!(id, copy(fx))
set_delta!( id :: AbstractIterData, Δ :: NumOrVec ) = _set_delta!(id, copy(Δ))
````

These two auxillary methods are derived from the above definitions:

````julia
"Increase the iteration counter by `N`."
function inc_num_iterations!( id :: AbstractIterData, N :: Int = 1 )
    current_num_iter = get_num_iterations( id )
    return set_num_iterations!( id, current_num_iter + N)
end

"Increase the model improvement counter by `N`."
function inc_num_model_improvements!( id :: AbstractIterData, N :: Int = 1 )
    current_num_model_improvements = get_num_model_improvements( id )
    return set_num_iterations!( id, current_num_model_improvements + N )
end
````

The actual constructor ensures that all arguments have eltype<:AbstractFloat:

````julia
"""
    init_iter_data( T , x, fx, Δ )
Return an instance of "base" type `T` implementing `AbstractIterData` with
correct type parameters for `x`, `fx` and `Δ`.
`x` and `fx` should be vectors of floats and `Δ` can either be a float or
a vector of floats.
"""
function init_iter_data( T :: Type{<:AbstractIterData}, x :: Vec, fx :: Vec, Δ :: NumOrVec )
    global MIN_PRECISION
    base_type = @eval $(T.name.name)    # strip any type parameters from T
    XTe = Base.promote_eltype( x, MIN_PRECISION )
    YTe = Base.promote_eltype( fx, MIN_PRECISION )
    DTe = Base.promote_eltype( Δ, MIN_PRECISION )
	return _init_iter_data( base_type, XTe.(x), YTe.(y), DTe.(Δ) )
end
````

# `AbstractIterSaveable`

The only thing we require for an `AbstractIterSaveable` is a constructor (think "extractor")
that gets us a saveable object to store in the database.
It can have arbritrary keyword arguments but should accept `kwargs...` in any
case:

````julia
get_saveable( :: Type{<:AbstractIterSaveable}, id :: AbstractIterData; kwargs... ) = nothing
get_saveable( :: Type{<:Nothing}, id :: AbstractIterData; kwargs... ) = nothing
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

