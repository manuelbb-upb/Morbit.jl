Broadcast.broadcastable( res :: AbstractResult ) = Ref(res)

"Return the site vector associated with a result."
( get_site( :: AbstractResult{F} ) :: AbstractVector{F} ) where F = F[]

"Return the value vector associated with a result."
( get_value( :: AbstractResult{F} ) :: AbstractVector{F} ) where F = F[]

set_site!(r :: AbstractResult, x) :: Nothing = nothing
set_value!(r :: AbstractResult, y) :: Nothing = nothing

"""
	get_id( res :: AbstractResult ) :: Int

Return the `id` of a result such that for the database `db` 
conataining `res` it holds that `get_result(db, id) == res`.
"""
get_id( :: AbstractResult ) :: Int = -1;

"Constructor for a result, taking site and value vector and id in database."
function init_res( :: T , :: Vec, :: Vec, :: Int ) :: T where T<:Type{<:AbstractResult}
	nothing
end

# derived 
function _equal_vals( r1 :: AbstractResult, r2 :: AbstractResult )
	return get_site(r1) == get_site(r2) && get_value(r1) == get_value(r2)
end

function has_valid_site( r :: AbstractResult )
	site = get_site(r)
	return !(isempty(site) || any( isnan.(site) ))
end

function has_valid_value( r :: AbstractResult )
	value = get_value(r)
	isempty(value) && return false 
	any( isnan.(value) ) && return false
end