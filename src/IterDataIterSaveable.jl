#src Does not depend on anything but `AbstractIterate` and `AbstractIterSaveable`

# # Iter Data Interface 

# Types (and instances of those types) that implement the `AbstractIterate`
# interface are used to have a unified access to relevant iteration data in 
# all functions of the main algorithm.

# First of all, we make it so that any `AbstractIterate` is broadcasted wholly:
Base.broadcastable( id :: AbstractIterate ) = Ref( id )

mutable struct IterData{ 
        XT <: VecF, YT <: VecF, XS <: VecF,
        E <: VecF, I <: VecF,
        ET <: VecF, IT <: VecF, DT <: NumOrVecF, 
        XIndType, 
    } <: AbstractIterate
    
    x :: XT 
    x_scaled :: XS
    fx :: YT
    l_e :: E 
    l_i :: I 
    c_e :: ET 
    c_i :: IT 
    Δ :: DT 
    
    x_indices :: XIndType
end

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
function _init_iterate( ::Type{<:IterData}, x :: VecF, 
    x_scaled :: VecF, fx :: VecF,
    l_e :: VecF, l_i :: VecF, 
    c_e :: VecF, c_i :: VecF, Δ :: NumOrVecF, x_index_mapping )
    return IterData(x, x_scaled, fx, l_e, l_i, c_e, c_i, Δ, x_index_mapping)
end

# ### Getters

# There are Getters for the mathematical objects relevant during optimzation:
"Return current iteration site vector ``xᵗ``."
get_x( id :: IterData ) = id.x

"Return current iteration site vector ``xᵗ``."
get_x_scaled( id :: IterData ) = id.x_scaled

"Return current value vector ``f(xᵗ)``."
get_fx( id :: IterData ) = id.fx

"Return value vector of linear equality constraints."
get_eq_const( id :: IterData ) = id.l_e

"Return value vector of linear inequality constraints."
get_ineq_const( id :: IterData ) = id.l_i

"Return current equality constraint vector ``cₑ(xᵗ)``."
get_nl_eq_const( id :: IterData ) = id.c_e

"Return current inequality constraint vector ``cᵢ(xᵗ)``."
get_nl_ineq_const( id :: IterData ) = id.c_i

"Return current trust region radius (vector) ``Δᵗ``."
get_delta( id :: IterData ) = id.Δ

# We also need the iteration result index for our sub-database.
"Index (or `id`) of current iterate in database."
get_x_index( id:: IterData, indices :: FunctionIndexTuple ) = id.x_indices[indices]
get_x_index_dict( id :: IterData ) = id.x_indices

function Base.show( io :: IO, id :: I) where I<:AbstractIterate
    str = "AbstractIterate of type $(_typename(I))."
    if !get(io, :compact, false)
        x = get_x(id)
        fx = get_fx(id)
        Δ = get_delta(id)
        c_e = get_nl_eq_const(id)
        c_i = get_nl_ineq_const(id)
        l_e = get_eq_const(id)
        l_i = get_ineq_const(id)
        str *= """ 
        x   $(lpad(eltype(x), 8, " ")) = $(_prettify(x))
        fx  $(lpad(eltype(fx), 8, " ")) = $(_prettify(fx))
        Δ   $(lpad(eltype(Δ), 8, " ")) = $(Δ)
        l_e $(lpad(eltype(l_e), 8, " ")) = $(_prettify(l_e))
        l_i $(lpad(eltype(l_i), 8, " ")) = $(_prettify(l_i))
        c_e $(lpad(eltype(c_e), 8, " ")) = $(_prettify(c_e))
        c_i $(lpad(eltype(c_i), 8, " ")) = $(_prettify(c_i))"""
    end
    print(io, str)
end
### Setters

# Implement the setters and note the leading underscore!
# The algorithm will actually call `set_delta!` instead of `_set_delta!` etc.
# These derived methods are implemented below and ensure that we actually 
# store copies of the input!

"Set current trust region radius (vector?) to `Δ`."
function _set_delta!( id :: IterData, Δ :: NumOrVec ) :: Nothing
    id.Δ = Δ
    return nothing
end

# ## Derived Methods

# The actual setters simply ensure proper copying (if a vector is provided):
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
function get_vals( id :: AbstractIterate, sdb, func_indices )
    x_index = get_x_index( id, func_indices )
    sub_db = get_sub_db( sdb, func_indices )
    return get_value( sub_db, x_index )
end

# # `AbstractIterSaveable`

# The only thing we require for an `AbstractIterSaveable` is a constructor (think "extractor")
# that gets us a saveable object to store in the database.

struct EmptyIterSaveable <: AbstractIterSaveable end

# ## Default fallbacks
# By default, return an `EmptyIterSaveable()`:
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

# A more useful implementation:
struct IterSaveable{
        XT <: VecF, D <: NumOrVecF,
        XIndType
    } <: AbstractIterSaveable

    iter_counter :: Int
    it_stat :: ITER_TYPE

    x :: XT
    Δ :: D
    x_indices :: XIndType

    # additional information for stamping
    ρ :: Float64
    stepsize :: Float64
    ω :: Float64
end

function get_saveable( :: Type{<:IterSaveable}, id :: AbstractIterate;
    iter_counter, it_stat, rho, steplength, omega)
    return IterSaveable(
        iter_counter, it_stat,
        get_x( id ),
        get_delta( id ),
        get_x_index_dict( id ),
        Float64(rho), Float64(steplength), Float64(omega)
    )
end

function get_saveable_type( :: Type{<:IterSaveable}, id :: AbstractIterate )
    return IterSaveable{ 
        typeof( get_x(id) ), 
        typeof( get_delta(id) ),
        typeof( get_x_index_dict(id) )
     }
end
