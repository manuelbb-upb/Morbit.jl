# # Result Datatype
# This file describes the `Result` datatype used to represent iteration data and 
# database entries.

@with_kw struct Result{XT <: VecF, YT <: VecF} <: AbstractResult
    x :: XT = Float64[]
    y :: YT = Float64[]
    db_id :: Int = -1
end

# First of all, we make it so that any `AbstractResult` is broadcasted wholly:
Broadcast.broadcastable( res :: Result ) = Ref(res)

# A result should be compared by values:
#src #TODO: do i use this somewhere?
function Base.:(==)(r1 :: Result, r2::Result)
    return (
        r1.x == r2.x &&
        r1.y == r2.y &&
        r1.db_id == r2.db_id
    )
end

# ## Mandatory Methods

# The following getter methods should be implemented:
"""
    get_site( res :: Result{XT,YT} )

Return evaluation site of type `XT` associated with `res`.
"""
get_site( res :: Result ) = res.x

"""
    get_value( res :: Result{XT,YT} )

Return evaluation value vector of type `YT` associated with `res`.
"""
get_value( res :: Result ) = res.y

"""
	get_id( res :: Result ) :: Int

Return the `id` of a result such that for the database `db` 
containing `res` it holds that `get_result(db, id) == res`.
"""
get_id( res :: Result ) = res.db_id

# Also define these setters:
function set_site!( res :: Result, x )
    res.x[:] .= x[:]
    return nothing 
end

function set_value!( res :: Result, y )
    # `if-else` for if `res.y` is an MVector
    if length(res.y) == length(y)
        res.y[:] .= y[:]
    else
        empty!(res.y)
        append!(res.y, y)
    end
    return nothing 
end

# What should be placeholders?
empty_site( :: Type{<:Result{Vector{F}, YT}} ) where {F<:AbstractFloat,YT} = F[]
empty_site( :: Type{<:Result{ XT, YT}} ) where {XT <: StaticVector, YT} = fill!(XT(undef),NaN)
empty_value( :: Type{<:Result{XT, Vector{F}}} ) where {F<:AbstractFloat,XT} = F[]
empty_value( :: Type{<:Result{XT, YT}} ) where {XT, YT<:StaticVector} = fill!(YT(undef),NaN)

# An `Result` is created with the `init_res` constructor taking 
# an evaluation site `x`, a value vector `y` and a database id.
"""
    init_res( id, x, y)

Return of result with site `x`, value `y` and database id `id`.
"""
function init_res( R ::Type{<:Result}, id :: Int, x :: Vec, y :: AbstractVector = MIN_PRECISION[])
    _y = isempty(y) ? empty_value( R ) : y
    return R( x, _y, id )
end

# ## Derived Methods 

# Based on the above definitions, there are some useful defaults defined.
# These can be overwritten for specific types but you don't have to:

"""
	_equal_vals( r1 :: Result, r2 :: Result )

Return `true` if both the site and the value vectors of `r1` and `r2` are equal.
"""
function _equal_vals( r1 :: Result, r2 :: Result )
	return get_site(r1) == get_site(r2) && get_value(r1) == get_value(r2)
end

function _is_valid_vector( x :: Vec ) 
	any( isnan.(x) ) && return false 
	isempty(x) && return false 
	return true 
end

"""
    has_valid_site( r :: Result )

Return `true` if the site vector of `r` is neither empty nor NaN.
"""
function has_valid_site( r :: Result )
	site = get_site(r)
	return _is_valid_vector(site)
end

"""
    has_valid_value( r :: Result )

Return `true` if the value vector of `r` is neither empty nor NaN.
"""
function has_valid_value( r :: Result )
	value = get_value(r)
	return _is_valid_vector(value)
end