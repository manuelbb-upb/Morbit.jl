```@meta
EditURL = "<unknown>/../src/AbstractDBInterface.jl"
```

# Database Interface
This file describes the methods that should or can be implemented
for subtypes of `AbstractDB`.

First of all, we make it so that any `AbstractResult` is broadcasted wholly:

````julia
Broadcast.broadcastable( db :: AbstractDB ) = Ref( db );
nothing #hide
````

## Mandatory Methods

A database should be constructed using the `init_db` method:

````julia
"Constructor for empty database of type `T`."
function init_db( :: T, ::Type{ <: AbstractResult },
    :: Type{<:NothingOrSaveable}) :: T where T<:Type{<:AbstractDB}
    nothing
end
````

Internally the variables might be scaled (or *transformed*).
The `is_transformed` method gives an indication to whether or
not the site vectors of the stored results are transformed or not.
`set_transformed!` is used to set the flag.

````julia
"Bool indicating if the database data been transformed."
is_transformed( :: AbstractDB ) :: Bool = false
"Set the flag indicating whether the database data has been transformed or not."
set_transformed!( :: AbstractDB, :: Bool ) :: Nothing = nothing
````

The results are indexed with integers and `get_ids` should
return a vector or iterator of all result ids:

````julia
"List of all `id :: Int` belonging to the stored results."
get_ids( db :: AbstractDB  ) = Int[]
````

An `id` can then be used to retrieve a result from the database:

````julia
"Get result with `id` from database `db`."
function get_result( db :: AbstractDB{R,I}, id :: Int ) :: R where {R <: AbstractResult,I}
    return R()
end
````

When a new result is added to the database, `next_id` is called to get
its id:

````julia
"Return an id for the next result to be added to `db`."
function next_id( db :: AbstractDB ) :: Int
	return -1
end

"Add result `res` to database `db`."
_add_result!(db :: AbstractDB, res :: AbstractResult) = nothing
````

There is only one simple method to put an `AbstractIterSaveable` into a database:

````julia
"""
    stamp!(db, ids)

Put the saveable `ids` into the database `db`.
"""
function stamp!( db :: AbstractDB, ids :: NothingOrSaveable) :: Nothing
	return nothing
end
````

## Derived methods

### Getters
There are getters for the types …

````julia
"Return type of results stored in database."
get_res_type( :: AbstractDB{R,I} ) where {R,I} = R
"Return type of `AbstractIterSaveable`s stored in database."
get_saveable_type( :: AbstractDB{R,I}) where {R,I} = I
````

… and for the values:

````julia
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
````

For getting the number of results in a database we owerwrite
`Base.length`.

````julia
"Number of entries in database."
Base.length( db :: AbstractDB ) :: Int = length(get_sites(db))
````

There is a default implmentation to get the ids of results in
a database that don't have valid value vectors.
I recommend to owerwrite this:

````julia
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
````

### Setters

`new_result!` is meant to construct a new result and return its data base index.
It is implemented so that if `id` is provided as a positive integer, the new result
has that id. Else, `next_id` should be called internally.

````julia
"Add a new result to the database, return its id of type Int."
function new_result!( db :: AbstractDB{R,I}, x :: Vec, y :: Vec, id :: Int = - 1 ) where{R,I}
    new_id = id < 0 ? next_id(db) : id
	new_result = init_res( R, x, y, new_id )
	push!(db.res, new_result)
	if !has_valid_value(new_result)
		set_evaluated_flag!( db, new_id, false )
	end
	db.num_entries += 1
	return new_id
end
````

These setters are used in (un)transforming the database.
They are based on the setters for `AbstractResult`:

````julia
"Set site of result with `id` in database `db` to `x`."
function set_site!(db, id, x) :: Nothing
    set_site!( get_result(db,id), x )
end

"Set value of result with `id` in database `db` to `x`."
function set_value!(db, id, y) :: Nothing
    set_value!( get_result(db,id), y )
end
````

If you overwrite `get_evaluated_flag` you might want to
overwrite `set_evaluated_flag!` too:

````julia
"Set the evaluation status for result with `id` to `state`."
set_evaluated_flag!( db :: AbstractDB, id :: Int, state = true) = nothing
````

### Miscellaneous

To find a result by its values we have:

````julia
"""
    find_result(db, x, y)
Return id of a result in `db` that has site `x` and value `y` or return -1
if there is no such result.
"""
function find_result( db :: AbstractDB, x :: Vec, y :: Vec  ) :: Int
    for id ∈ get_ids( db )
        if get_site( db, id ) == x && get_value( db, id ) == y
            return id
        end
    end
    return -1
end
````

The above function is utilized in `ensure_contains_values!`:

````julia
"Return id of result in `db` with site `x` and values `y`. Create if necessary."
function ensure_contains_values!( db :: AbstractDB, x :: Vec, y :: Vec ) :: Int
    x_pos = find_result(db, x,y);
    if x_pos < 0
        x_pos = new_result!(db, x, y);
    end
    return x_pos
end
````

The `eval_missing` method is important for the new two stage model
construction process and called after "preparing" the models for
updates but before calling the `update_model` methods:

````julia
"Evaluate all unevaluated results in `db` using objectives of `mop`."
function eval_missing!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing

    missing_ids = _missing_ids(db)

    @logmsg loglevel2 "Performing $(length(missing_ids)) objective evaluations into the database."
    # evaluate everything in one go to exploit parallelism
    eval_sites = [ get_site( db, id ) for id in missing_ids ]
    eval_values = eval_all_objectives.(mop, eval_sites)

    @assert length(eval_sites) == length(eval_values) == length(missing_ids) "Number of evaluation results does not match."
    for (i,id) in enumerate(missing_ids)
        set_value!( db, id, eval_values[i] )
    end

    for id in missing_ids
        set_evaluated_flag!(db, id)
    end
    return nothing
end
````

Internally the variables might be scaled.
For conversion we offer the `scale!` and `unscale!` defaults:

````julia
"Scale the site of result with `id` in database `db` using bounds of `mop`."
function scale!( db :: AbstractDB, id :: Int, mop :: AbstractMOP ) :: Nothing
    set_site!( db, id, scale( get_site( db, id ), mop ) )
    return nothing
end

"Unscale the site of result with `id` in database `db` using bounds of `mop`."
function unscale!( db :: AbstractDB, id :: Int, mop :: AbstractMOP ) :: Nothing
    set_site!( db, id, unscale( get_site( db, id), mop ) )
    return nothing
end
````

Also, the value vectors might be resorted internally.
This is taken care of by the following methods:

````julia
"Apply internal objective sorting to result with `id` in `db`."
function apply_internal_sorting!( db :: AbstractDB, id :: Int, mop :: AbstractMOP ) :: Nothing
	set_value!( db, id, apply_internal_sorting( get_value(db, id), mop ) )
    return nothing
end

"Reverse internal sorting of objectives for the result with `id` in `db`."
function reverse_internal_sorting!( db :: AbstractDB, id :: Int, mop :: AbstractMOP ) :: Nothing
	set_value!( db, id, reverse_internal_sorting( get_value(db, id), mop ) )
    nothing
end
````

Both, variable scaling and objective sorting, is combined in the
`(un)transform!` methods:

````julia
"Apply scaling and objectives sorting to each result in database `db`."
function transform!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    if !is_transformed(db)
        for id in get_ids(db)
            scale!( db, id, mop)
            apply_internal_sorting!( db, id, mop )
        end
        set_transformed!(db, true)
    end
    nothing
end

"Undo scaling and objectives sorting to each result in database `db`."
function untransform!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    if is_transformed(db)
        for id in get_ids(db)
            unscale!( db, id, mop)
            reverse_internal_sorting!( db, id, mop )
        end
        set_transformed!(db, false)
    end
    nothing
end
````

We have a generic copy function that returns a new database containing
results from the old database:

````julia
"""
    copy_db( old_db, result_type, saveable_type )

Return a new database of same 'base' type but possibly with different result and
saveable type.
"""
function copy_db( old_db :: DBT; res_type = Nothing, saveable_type :: Type = Nothing ) where DBT <: AbstractDB
    try
        base_type = @eval $(DBT.name.name)
        _res_type = res_type <: Nothing ? get_res_type( old_db ) : res_type
        _saveable_type = saveable_type <: Nothing ? get_saveable_type( old_db ) : saveable_type
        new_db = init_db(base_type, _res_type, _saveable_type )
        for id = get_ids(old_db)
            res = get_result( old_db, id )
            new_result!(new_db, get_site(res), get_value(res) )
        end
        @logmsg loglevel2 "Copied database with new saveable type."
        return new_db
    catch e
        @error "Failed to copy database with new saveable type." exception=(e, catch_backtrace())
        return old_db
    end
end
````

Finally, this little helper returns ids of database results that
conform to some variable bounds and is used in the model construction:

````julia
"Return indices of results in `db` that lie in a box with corners `lb` and `ub`."
function results_in_box_indices(db, lb, ub, exclude_indices = Int[] )
	return [ id for id = get_ids( db ) if
		id ∉ exclude_indices && all(lb .<= get_site(db,id) .<= ub ) ]
end
````

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

