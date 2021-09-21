# # Iter Data Interface 

# Types (and instances of those types) that implement the `AbstractIterData`
# interface are used to have a unified access to relevant iteration data in 
# all functions of the main algorithm.

# First of all, we make it so that any `AbstractResult` is broadcasted wholly:
Base.broadcastable( id :: AbstractIterData ) = Ref( id )

# ## Mandatory Methods 

# Any subtype of `AbstractIterData` should implement these methods.

# First, we need a constructor.
# This constructor should take a base type implementing `AbstractIterData` and 
# its first states as arguments, where `x` and `fx` are vectors of floats and `Δ`
# is either a float or a vector thereof:
function _init_iter_data( T :: Type{<:AbstractIterData}, x :: VecF, fx :: VecF, 
    c_e :: VecF, c_i :: VecF, Δ :: NumOrVecF, x_index_mapping; kwargs... )
    return nothing 
end

# ### Getters

# There are Getters for the mathematical objects relevant during optimzation:
"Return current iteration site vector ``xᵗ``."
function get_x( :: AbstractIterData{XT,YT,ET,IT,DT} ) where {XT,YT,ET,IT,DT}
    return XT()
end

"Return current value vector ``f(xᵗ)``."
function get_fx(  :: AbstractIterData{XT,YT,ET,IT,DT} ) where {XT,YT,ET,IT,DT}
    return YT()
end

"Return current equality constraint vector ``cₑ(xᵗ)``."
function get_eq_const( :: AbstractIterData{XT,YT,ET,IT,DT} ) where {XT,YT,ET,IT,DT}
    return ET()
end

"Return current inequality constraint vector ``cᵢ(xᵗ)``."
function get_ineq_const( :: AbstractIterData{XT,YT,ET,IT,DT} ) where {XT,YT,ET,IT,DT}
    return IT()
end

"Return current trust region radius (vector) ``Δᵗ``."
function get_delta( :: AbstractIterData{XT,YT,ET,IT,DT} ) where {XT,YT,ET,IT,DT}
    return zero(eltype(DT))
end

# We also need the iteration result index for our sub-database.
# This should be implemented but works as is if only `MockDB` is used.
"Index (or `id`) of current iterate in database."
get_x_index( :: AbstractIterData, :: FunctionIndexTuple ) :: Int = -1
get_x_index( id :: AbstractIterData, func_indices ) = get_x_index( id, Tuple(func_indices) )
# For printing information and for stopping we need the iteration count:
"Return number of iterations so far."
get_num_iterations( :: AbstractIterData ) :: Int = 0
"Return the number of model improvement iterations so far."
get_num_model_improvements( :: AbstractIterData ) :: Int = 0

# Finally, we also want to display the last iteration classification:
"Return the iteration classification of `ITER_TYPE`."
it_stat( :: AbstractIterData ) :: ITER_TYPE = SUCCESSFULL

# ### Setters

# Implement the setters and note the leading underscore!
# The algorithm will actually call `set_x!` instead of `_set_x!` etc.
# These derived methods are implemented below and ensure that we actually 
# store copies of the input!

"Set current iteration site to `x̂`."
_set_x!( :: AbstractIterData, x̂ :: Vec ) :: Nothing = nothing 

"Set current iteration value vector to `ŷ`."
_set_fx!( :: AbstractIterData, ŷ :: Vec ) :: Nothing = nothing 

"Set current equality constraint vector to `c`."
_set_eq_const!( :: AbstractIterData, c :: Vec ) :: Nothing = nothing 

"Set current inequality constraint vector to `c`."
_set_ineq_const!( :: AbstractIterData, c :: Vec ) :: Nothing = nothing 

"Set current trust region radius (vector?) to `Δ`."
_set_delta!( :: AbstractIterData, Δ :: NumOrVec ) :: Nothing = nothing
   
"Set the current iteration database id to `val`."
set_x_index!( :: AbstractIterData, :: FunctionIndexTuple, val :: Int ) :: Nothing = nothing

# The itaration counters are modified by the following methods:
"Set the iteration counter to `N`."
set_num_iterations!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing

"Set the improvement counter to `N`."
set_num_model_improvements!( :: AbstractIterData, N :: Int = 0 ) :: Nothing = nothing ;

# Of course, we also need a way to set the iteration classification:
"Set the iteration classification."
it_stat!( :: AbstractIterData, :: ITER_TYPE ) :: Nothing = nothing;

# ## Derived Methods

# The actual setters simply ensure proper copying:
set_x!( id :: AbstractIterData, x :: Vec ) = _set_x!(id, copy(x))
set_fx!( id :: AbstractIterData, fx :: Vec ) = _set_fx!(id, copy(fx))
set_eq_const!( id :: AbstractIterData, c :: Vec ) = _set_eq_const!(id, copy(c))
set_ineq_const!( id :: AbstractIterData, c :: Vec ) = _set_ineq_const!(id, copy(c))
set_delta!( id :: AbstractIterData, Δ :: NumOrVec ) = _set_delta!(id, copy(Δ))

# These two auxillary methods are derived from the above definitions:
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

# The actual constructor ensures that all arguments have eltype<:AbstractFloat:
"""
    init_iter_data( T , x, fx, Δ )
Return an instance of "base" type `T` implementing `AbstractIterData` with 
correct type parameters for `x`, `fx` and `Δ`.
`x` and `fx` should be vectors of floats and `Δ` can either be a float or 
a vector of floats.
"""
function init_iter_data( T :: Type{<:AbstractIterData}, x :: Vec, fx :: Vec, c_e :: Vec, c_i :: Vec, Δ :: NumOrVec,
    x_index_mapping ; kwargs... )
    base_type = @eval $(_typename(T))    # strip any type parameters from T
	return _init_iter_data( base_type, 
        ensure_precision(x), ensure_precision(fx), 
        ensure_precision(c_e), ensure_precision(c_i), 
        ensure_precision(Δ), x_index_mapping; kwargs... )
end

# # `AbstractIterSaveable`

# The only thing we require for an `AbstractIterSaveable` is a constructor (think "extractor")
# that gets us a saveable object to store in the database.
# It can have arbritrary keyword arguments but should accept `kwargs...` in any 
# case:

get_saveable( :: Type{<:AbstractIterSaveable}, id :: AbstractIterData; kwargs... ) = nothing
get_saveable( :: Type{<:Nothing}, id :: AbstractIterData; kwargs... ) = nothing