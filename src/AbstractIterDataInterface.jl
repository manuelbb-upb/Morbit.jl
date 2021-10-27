# # Iter Data Interface 

# Types (and instances of those types) that implement the `AbstractIterate`
# interface are used to have a unified access to relevant iteration data in 
# all functions of the main algorithm.

# First of all, we make it so that any `AbstractIterate` is broadcasted wholly:
Base.broadcastable( id :: AbstractIterate ) = Ref( id )

# ## Mandatory Methods 

# Any subtype of `AbstractIterate` should implement these methods.

# First, we need constructors.
"""
    _init_iterate( T , x, x_scaled, fx, 
        l_e, l_i, c_e, c_i, Δ, x_index_mapping )
    
Return an `AbstractIterate` of type `T`. The arguments 
`x` through `c_i` are site and value vectors. 
`Δ` is the associated trust region radius.
`x_index_mapping` is a `Dict` mapping `FunctionIndexTuple`s 
to the index of `x` in the current (sub)database.
"""
function _init_iterate( T :: Type{<:AbstractIterate},
    x :: VecF, x_scaled :: VecF , fx :: VecF,
    l_e :: VecF, l_i :: VecF,
    c_e :: VecF, c_i :: VecF, Δ :: NumOrVecF, x_index_mapping )
    nothing
end

# ### Getters

# There are Getters for the mathematical objects relevant during optimzation:
"Return current iteration site vector ``xᵗ``."
get_x( :: AbstractIterate ) = nothing 

"Return current iteration site vector ``xᵗ``."
get_x_scaled( :: AbstractIterate )=nothing 

"Return current value vector ``f(xᵗ)``."
get_fx(  :: AbstractIterate ) = nothing 

get_eq_const( :: AbstractIterate ) = nothing

get_ineq_const( :: AbstractIterate ) = nothing

"Return current equality constraint vector ``cₑ(xᵗ)``."
get_nl_eq_const( :: AbstractIterate ) = nothing

"Return current inequality constraint vector ``cᵢ(xᵗ)``."
get_nl_ineq_const( :: AbstractIterate ) = nothing 

"Return current trust region radius (vector) ``Δᵗ``."
get_delta( :: AbstractIterate ) = nothing 

# We also need the iteration result index for our sub-database.
# This should be implemented but works as is if only `MockDB` is used.
"Index (or `id`) of current iterate in database."
get_x_index( :: AbstractIterate, :: FunctionIndexTuple ) :: Int = -1
get_x_index( id :: AbstractIterate, func_indices ) = get_x_index( id, Tuple(func_indices) )

get_x_index_dict( id :: AbstractIterate ) = Base.ImmutableDict{FunctionIndexTuple,Int}()

### Setters

# Implement the setters and note the leading underscore!
# The algorithm will actually call `set_delta!` instead of `_set_delta!` etc.
# These derived methods are implemented below and ensure that we actually 
# store copies of the input!

"Set current trust region radius (vector?) to `Δ`."
_set_delta!( :: AbstractIterate, Δ :: NumOrVec ) :: Nothing = nothing

# ## Derived Methods

# The actual setters simply ensure proper copying:
set_delta!( id :: AbstractIterate, Δ :: NumOrVec ) = _set_delta!(id, copy(Δ))

# The actual constructor ensures that all arguments have eltype<:AbstractFloat:
"""
    init_iter_data( T , x, fx, Δ )
Return an instance of "base" type `T` implementing `AbstractIterate` with 
correct type parameters for `x`, `fx` and `Δ`.
`x` and `fx` should be vectors of floats and `Δ` can either be a float or 
a vector of floats.
"""
function init_iterate( T :: Type{<:AbstractIterate}, x :: Vec, x_scaled :: Vec, fx :: Vec, 
    l_e :: Vec, l_i :: Vec, c_e :: Vec, c_i :: Vec, Δ :: NumOrVec,
    x_index_mapping )
    base_type = @eval $(_typename(T))    # strip any type parameters from T
	return _init_iterate( base_type,
        ensure_precision(x),
        ensure_precision(x_scaled),
        ensure_precision(fx),
        ensure_precision(l_e), ensure_precision(l_i), 
        ensure_precision(c_e), ensure_precision(c_i), 
        ensure_precision(Δ), x_index_mapping,
    )
end

"Return only the parts of f(x) that are relevant for `func_indices`."
function get_vals( id :: AbstractIterate, sdb :: AbstractSuperDB, func_indices )
    x_index = get_x_index( id, func_indices )
    sub_db = get_sub_db( sdb, func_indices )
    return get_value( sub_db, x_index )
end

# # `AbstractIterSaveable`

# The only thing we require for an `AbstractIterSaveable` is a constructor (think "extractor")
# that gets us a saveable object to store in the database.

struct EmptyIterSaveable <: AbstractIterSaveable end

function get_saveable( :: Type{<:AbstractIterSaveable}, id; 
 it_stat, rho, steplength, omega) 
 return EmptyIterSaveable()
end

# The type `T` returned by `get_saveable_type` should be such
# that `get_saveable(T, id) :: T`
function get_saveable_type( T :: Type{<:AbstractIterSaveable}, id :: AbstractIterate )
    return typeof( get_saveable( T, id ) )
end

function get_saveable_type( id :: AbstractIterate, ac :: AbstractConfig )
    T = iter_saveable_type( ac )
    if T <: EmptyIterSaveable
        return EmptyIterSaveable
    else
        return get_saveable_type(T, id :: AbstractIterate)
    end
end