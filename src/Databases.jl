# # Databases
# This file describes the methods that should or can be implemented 
# for subtypes of `AbstractDB`.

# First of all, we make it so that any `AbstractDB` is broadcasted wholly:
Broadcast.broadcastable( db :: AbstractDB ) = Ref( db );

# There are two implementations of `AbstractDB`:
# First `MockDB`, that is used whenever we do not want 
# to save iteration data:
struct MockDB{R,I} <: AbstractDB{R,I} end 
# Note, that whenever a method is implemented for `AbstractDB`
# it usually describes the behavior for `MockDB` too.\
# Then there is `ArrayDB` that uses vectors to store data.
@with_kw mutable struct ArrayDB{
	R <: AbstractResult, I <: NothingOrMeta
} <: AbstractDB{R,I}
	
	res :: Vector{R} = R[]
	iter_info :: Vector{I} = I[]

	# counter of entries 
	num_entries :: Int = 0

	# transformed results indicator
	transformed :: Bool = false
	
	# list of ids for unevaluated results
	unevaluated_ids :: Vector{Int} = []

	@assert length(res) == num_entries "Number of entries does not match `num_entries`."
end

# ## Mandatory Methods

# A database should be constructed using the `init_db` method:
"Constructor for empty database of type `MockDB`."
init_db( :: MockDB, R, I, args... ) = MockDB{R,I}()

"Constructor for empty database of type `ArrayDB`."
function init_db(:: Type{<:ArrayDB}, R :: Type{<:AbstractResult},
		I :: Type{<:NothingOrMeta})
	return ArrayDB{R, I}()
end

# Internally the variables might be scaled (or *transformed*).
# The `is_transformed` method gives an indication to whether or 
# not the site vectors of the stored results are transformed or not.
# `set_transformed!` is used to set the flag.
"Bool indicating if the database data been transformed."
is_transformed( :: AbstractDB ) :: Bool = false
is_transformed( db :: ArrayDB ) = db.transformed 

"Set the flag indicating whether the database data has been transformed or not."
set_transformed!( :: AbstractDB, :: Bool ) :: Nothing = nothing
function set_transformed!( db :: ArrayDB, val :: Bool )
	db.transformed = val 
	return nothing 
end

# The results are indexed with integers and `get_ids` should 
# return a vector or iterator of all result ids:
"List of all `id :: Int` belonging to the stored results."
get_ids( db :: AbstractDB  ) = Int[]
get_ids( db :: ArrayDB ) = Base.eachindex( db.res )

# An `id` can then be used to retrieve a result from the database:
"Get result with `id` from database `db`."
function get_result( db :: AbstractDB{R,I}, id :: Int ) :: R where {R,I}
    return R()
end

function get_result( db :: ArrayDB, id :: Int)
	return db.res[id]
end

# When a new result is added to the database, `next_id` is called to get 
# its id:
"Return an id for the next result to be added to `db`."
function next_id( db :: AbstractDB ) :: Int 
	return -1
end

function next_id( db :: ArrayDB) :: Int
	return db.num_entries + 1
end

"Add result `res` to database `db`."
_add_result!(db :: AbstractDB{R,I}, res :: R) where{R,I}= nothing

function _add_result!( db :: ArrayDB{R,I}, res :: R ) where{R,I}
	push!(db.res, res)
	return nothing
end

# There is only one simple method to put an `AbstractSurrogateMeta` into a database:
"""
    stamp!(db, ids)

Put the saveable `ids` into the database `db`.
"""
function stamp!( db :: AbstractDB, ids :: NothingOrMeta) :: Nothing 
	return nothing 
end

function stamp!( db :: ArrayDB{R,I}, ids :: I) where{R,I}
	push!(db.iter_info, ids)
	return nothing
end


# ## Derived Methods

# ### Getters
# There are getters for the types ???
"Return type of results stored in database."
get_res_type( :: AbstractDB{R,I} ) where {R,I} = R
"Return type of `AbstractIterSaveable`s stored in database."
get_saveable_type( :: AbstractDB{R,I}) where {R,I} = I

# ??? and for the values:
"Return the evaluation site vector for result with `id` in database `db`."
get_value( db :: AbstractDB, id :: Int ) = get_value( get_result( db, id ) )
"Return the evaluation value vector for result with `id` in database `db`."
get_site( db :: AbstractDB, id :: Int ) = get_site( get_result( db, id) )

"Return a vector of all evaluation site vectors stored in database."
function get_sites( db :: AbstractDB)
	return [ get_site( get_result( db, id ) ) for id = get_ids( db ) ]
end

"Return a vector of all evaluation value vectors stored in database."
function get_values( db :: AbstractDB  )
	return [ get_value( get_result( db, id ) ) for id = get_ids( db ) ]
end

# For getting the number of results in a database we owerwrite
# `Base.length`.
"Number of entries in database."
Base.length( db :: AbstractDB ) :: Int = length(get_sites(db)) 

# There is a default implmentation to get the ids of results in 
# a database that don't have valid value vectors.
# I recommend to owerwrite this:
"Return `true` if the result with `id` in `db` has a valid evaluation vector."
function get_evaluated_flag( db, id ) :: Bool
    res = get_result(db, id)
    has_valid_value( res ) && return true 
    return false
end
   
"Return vector of ids of database `db` that are not evaluated yet."
function _missing_ids( db :: AbstractDB )
    missing_ids = Int[]
    for id = get_ids( db )
        if !get_evaluated_flag(db, id)
            push!(missing_ids, id)
        end
    end
    return _missing_ids
end

## specialization for `ArrayDB`
function _missing_ids(db :: ArrayDB)
	return db.unevaluated_ids
end

# ### Setters 

# `new_result!` is meant to construct a new result and return its data base index.
# It is implemented so that if `id` is provided as a positive integer, the new result 
# has that id. Else, `next_id` should be called internally.
"Add a new result to the database, return its id of type Int."
function new_result!( db :: AbstractDB{R,I}, x :: Vec, y = [], id :: Int = - 1 ) where{R,I}
    new_id = id < 0 ? next_id(db) : id
	new_result = init_res( R, new_id, x, y)
	_add_result!(db, new_result)
	if !has_valid_value(new_result) 
		set_evaluated_flag!( db, new_id, false )
	end
	db.num_entries += 1
	return new_id
end

# These setters are used in (un)transforming the database.
# They are based on the setters for `Result`s:
"Set site of result with `id` in database `db` to `x`."
function set_site!(db, id, x) :: Nothing
    set_site!( get_result(db,id), x )
end

"Set value of result with `id` in database `db` to `x`."
function set_value!(db, id, y) :: Nothing
    set_value!( get_result(db,id), y )
end

# If you overwrite `get_evaluated_flag` you might want to 
# overwrite `set_evaluated_flag!` too:
"Set the evaluation status for result with `id` to `state`."
set_evaluated_flag!( db :: AbstractDB, id :: Int, state = true) = nothing

function set_evaluated_flag!( db :: ArrayDB, id :: Int, state = true)
	if state == false 
		push!( db.unevaluated_ids, id)
	else
		pos = findfirst( i -> i == id, db.unevaluated_ids )
		if !isnothing(pos)
			deleteat!( db.unevaluated_ids, pos )
		end
	end	
	return nothing
end

# ### Miscellaneous

# To find a result by its values we have:
"""
    find_result(db, x, y)
Return id of a result in `db` that has site `x` and value `y` or return -1 
if there is no such result.
"""
function find_result( db :: AbstractDB, x :: Vec, y = nothing ) :: Int
    ignore_value = isnothing(y)
    for id ??? get_ids( db )
        if get_site( db, id ) == x && (ignore_value || get_value( db, id ) == y)
            return id
        end
    end
    return -1
end

# The above function is utilized in `ensure_contains_values!`:
"Return id of result in `db` with site `x` and values `y`. Create if necessary."
function ensure_contains_values!( db :: AbstractDB, x :: Vec, y ) :: Int
    x_pos = find_result(db, x, y)
    if x_pos < 0
        x_pos = new_result!(db, x, y)
    end
    return x_pos 
end

"Return id of result in `db` with site `x` and values `y`. Create if necessary."
function ensure_contains_res_with_site!( db :: AbstractDB, x :: Vec ) :: Int
    x_pos = find_result(db, x, nothing)
    if x_pos < 0
        new_y = empty_value(get_res_type(db))
        x_pos = new_result!(db, x, new_y)
    end
    return x_pos 
end


# The `eval_missing` method is important for the new two stage model 
# construction process and called after "preparing" the models for 
# updates but before calling the `update_model` methods:

"Evaluate all unevaluated results in `db` using objectives of `mop`."
function eval_missing!( db :: AbstractDB, mop :: AbstractMOP, scal :: AbstractVarScaler, func_indices) :: Nothing
    missing_ids = copy(_missing_ids(db))

    n_missing = length(missing_ids)
    @logmsg loglevel2 "Performing $(n_missing) evaluations into the database for \n\t\t$(func_indices...)."
    if n_missing > 0
        ## evaluate everything in one go to exploit parallelism
        eval_sites = untransform.( [ get_site( db, id ) for id in missing_ids ], scal )
        eval_values = _eval_to_vecs_at_indices_at_unscaled_sites(mop, func_indices, eval_sites)
    
        @assert length(eval_sites) == length(eval_values) == length(missing_ids) "Number of evaluation results does not match."
        for (i,id) in enumerate(missing_ids)
            set_value!( db, id, eval_values[i] )
        end
        for id = missing_ids
            set_evaluated_flag!(db, id, true)
        end
    end
    return nothing
end

#src NOTE the modifying methods require the `get_site` and `get_value`
#src to return mutable arrays for result. Can we ensure this?

# Internally the variables might be scaled.
# For conversion we offer the `transform!` and `untransform!` defaults:
"Scale the site of result with `id` in database `db` using bounds of `mop`."
function transform!( db :: AbstractDB, id :: Int, scal :: AbstractVarScaler )
    set_site!( db, id, transform( get_site( db, id ), scal ) ) 
    return nothing
end

"Unscale the site of result with `id` in database `db` using bounds of `mop`."
function untransform!( db :: AbstractDB, id :: Int, scal :: AbstractVarScaler )
    set_site!( db, id, untransfrom( get_site( db, id), scal ) )
    return nothing
end

# Both, variable scaling ~and objective sorting~, is combined in the 
# `(un)transform!` methods:

"Apply scaling and objectives sorting to each result in database `db`."
function transform!( db :: AbstractDB, scal :: AbstractVarScaler ) :: Nothing 
    if !is_transformed(db)
        for id in get_ids(db)
            transform!( db, id, scal)
        end
        set_transformed!(db, true)
    end
    nothing
end

"Undo scaling and objectives sorting to each result in database `db`."
function untransform!( db :: AbstractDB, scal :: AbstractVarScaler ) :: Nothing 
    if is_transformed(db)
        for id in get_ids(db)
            untransform!( db, id, scal)
        end
        set_transformed!(db, false)
    end
    nothing
end

# Finally, this little helper returns ids of database results that 
# conform to some variable bounds and is used in the model construction:
"Return indices of results in `db` that lie in a box with corners `lb` and `ub`."
function results_in_box_indices(db, lb, ub, exclude_indices = Int[] )
	return [ id for id = get_ids( db ) if
		id ??? exclude_indices && all(lb .<= get_site(db,id) .<= ub ) ]
end

####################################################
# Super DB 
####################################################
""" 
    SuperDB{T,Vector{<IterSaveable}}

A `SuperDB` is nothing but a directory of sub-databases (`sub_dbs`)
that store evaluation data for tuples of `AnyIndex` functions.
There is also a field `iter_data` that stores iteration data in a vector 
for later inspection.
"""
@with_kw struct SuperDB{ 
		T <: AbstractIterSaveable, 
		ST <: Union{AbstractDict, AbstractDictionary},#{FunctionIndexTuple,<:AbstractDB} 
	}

    "A dictionary mapping `AnyIndex` tuples to sub databases."
    sub_dbs :: ST

    "A vector storing iteration data for later inspection."
    iter_data :: Vector{T} = T[]
end

all_sub_db_indices( sdb :: SuperDB ) = keys(sdb.sub_dbs)
get_sub_db( sdb :: SuperDB, key_indices :: Tuple) = sdb.sub_dbs[key_indices]

function stamp!( sdb :: SuperDB, ids :: AbstractIterSaveable)
	push!(sdb.iter_data, ids)
	return nothing
end

get_saveable_type( sdb :: SuperDB{T,ST}) where{T,ST} = T

# derived 
all_sub_dbs( sdb :: SuperDB) = [ get_sub_db(sdb, ki ) for ki in all_sub_db_indices(sdb) ]
get_sub_db( sdb :: SuperDB, func_indices ) = get_sub_db( sdb, Tuple(func_indices) ) 

is_transformed(sdb :: SuperDB) = all( is_transformed(db) for db = all_sub_dbs(sdb) )

function transform!( sdb :: SuperDB, scal :: AbstractVarScaler  )
    for sub_db in all_sub_dbs( sdb )
        transform!(sub_db, scal)
    end
    return nothing
end

function untransform!( sdb :: SuperDB, scal :: AbstractVarScaler  )
    for sub_db in all_sub_dbs( sdb )
        transform!(sub_db, scal)
    end
    return nothing
end

"Evaluate all unevaluated results in `db` using objectives of `mop`."
function eval_missing!( sdb :: SuperDB, mop :: AbstractMOP, scal :: AbstractVarScaler )
    for func_indices in all_sub_db_indices(sdb)
        eval_missing!( get_sub_db(sdb, func_indices), mop, scal, func_indices )
    end
    return nothing
end

function put_eval_result_into_db!( sdb :: SuperDB, 
    eval_result :: Union{AbstractDict, AbstractDictionary}, x :: Vec 
)
    x_indices = Dict{NLIndexTuple, Int}()
    for func_indices in all_sub_db_indices(sdb)
        sub_db = get_sub_db( sdb, func_indices )
        vals = flatten_vecs( eval_result[f_ind] for f_ind=func_indices ) 
        ind = new_result!( sub_db, x, vals )
        x_indices[func_indices] = ind 
    end
    return x_indices
end