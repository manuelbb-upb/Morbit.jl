```@meta
EditURL = "<unknown>/src/AbstractResultInterface.jl"
```

# Result Interface
This file describes the methods that should or can be implemented
for subtypes of `AbstractResult`.

First of all, we make it so that any `AbstractResult` is broadcasted wholly:

````julia
Broadcast.broadcastable( res :: AbstractResult ) = Ref(res)
````

## Mandatory Methods

The following getter methods should be implemented:

````julia
"""
    get_site( res :: AbstractResult{XT,YT} )

Return evaluation site of type `XT` associated with `res`.
"""
function get_site( :: AbstractResult{XT,YT} ) :: XT where {XT,YT}
	return XT()
end

"""
    get_value( res :: AbstractResult{XT,YT} )

Return evaluation value vector of type `YT` associated with `res`.
"""
function get_value( :: AbstractResult{XT,YT} ) :: YT where {XT,YT}
	return YT()
end

"""
	get_id( res :: AbstractResult ) :: Int

Return the `id` of a result such that for the database `db`
containing `res` it holds that `get_result(db, id) == res`.
"""
get_id( :: AbstractResult ) :: Int = -1
````

Also define these setters:

````julia
set_site!(r :: AbstractResult, x) :: Nothing = nothing
set_value!(r :: AbstractResult, y) :: Nothing = nothing
````

An `AbstractResult` is created with the `init_res` constructor taking
an evaluation site `x`, a value vector `y` and a database id.

````julia
"""
    init_res( res_type, x, y, id)

Return of result of type `res_type` with site `x`, value `y` and database id `id`.
"""
function init_res( :: Type{<:AbstractResult}, :: Vec, :: Vec, :: Int )
	return nothing
end
````

## Derived Methods

Based on the above definitions, there are some useful defaults defined.
These can be overwritten for specific types but you don't have to:

````julia
"""
	_equal_vals( r1 :: AbstractResult, r2 :: AbstractResult )

Return `true` if both the site and the value vectors of `r1` and `r2` are equal.
"""
function _equal_vals( r1 :: AbstractResult, r2 :: AbstractResult )
	return get_site(r1) == get_site(r2) && get_value(r1) == get_value(r2)
end

"""
    has_valid_site( r :: AbstractResult )

Return `true` if the site vector of `r` is neither empty nor NaN.
"""
function has_valid_site( r :: AbstractResult )
	site = get_site(r)
	return !(isempty(site) || any( isnan.(site) ))
end

"""
    has_valid_value( r :: AbstractResult )

Return `true` if the value vector of `r` is neither empty nor NaN.
"""
function has_valid_value( r :: AbstractResult )
	value = get_value(r)
	isempty(value) && return false
	any( isnan.(value) ) && return false
	return true
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

