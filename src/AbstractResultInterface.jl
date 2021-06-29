Broadcast.broadcastable( res :: AbstractResult ) = Ref(res)

# NOTE the site and value vectors should be mutable!
"Return the site vector associated with a result."
( get_site( :: AbstractResult{F} ) :: AbstractVector{F} ) where F = F[];

"Return the value vector associated with a result."
( get_value( :: AbstractResult{F} ) :: AbstractVector{F} ) where F = F[];

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