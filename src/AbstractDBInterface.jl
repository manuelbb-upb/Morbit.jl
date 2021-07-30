
############################################
# AbstractDB
Broadcast.broadcastable( db :: AbstractDB ) = Ref( db );

"Number of entries in database."
Base.length( db :: AbstractDB ) :: Int = length(get_sites(db)) 

"Constructor for empty database of type `T`."
init_db( :: Type{<:AbstractDB}, :: Type{<:AbstractFloat}, :: Union{Type{<:Nothing},Type{<:AbstractIterSaveable}} ) :: AbstractDB = nothing 

"Bool indicating if the database data been transformed."
is_transformed( :: AbstractDB ) :: Bool = false
"Set the flag indicating whether the database data has been transformed or not."
set_transformed!( :: AbstractDB, :: Bool ) :: Nothing = nothing

"List of all `id :: Int` belonging to the stored results."
Base.eachindex( db :: AbstractDB ) ::Vector{Int} = Int[]

"Get result with `id` from database `db`."
(get_result( db :: AbstractDB{F}, id :: Int ) :: AbstractResult ) where F = NoRes{F}()

function next_id( db :: AbstractDB ) :: Int 
	return -1
end

"Add a new result to the database, return its id of type Int."
function new_result!( db :: AbstractDB, x :: Vec, y :: Vec, id :: Int = - 1 ) :: Int
    return -1
end

function stamp!( db :: AbstractDB, ids :: AbstractIterSaveable ) :: Nothing 
	return nothing 
end

# Derived methods
get_precision( :: AbstractDB{F} ) where F = F

get_value( db :: AbstractDB, id :: Int ) = get_value( get_result( db, id ) )
get_site( db :: AbstractDB, id :: Int ) = get_site( get_result( db, id) )

function get_sites( db :: AbstractDB)
	return [ get_site( get_result( db, id ) ) for id = eachindex( db ) ]
end

function get_values( db :: AbstractDB  )
	return [ get_value( get_result( db, id ) ) for id = eachindex( db ) ]
end

function set_site!(db, id, x) :: Nothing
    set_site!( get_result(db,id), x )
end

function set_value!(db, id, y) :: Nothing
    set_value!( get_result(db,id), y )
end

function find_result( db :: AbstractDB, x :: Vec, y :: Vec  ) :: Int
    for id ∈ eachindex(db)
        if get_site( db, id ) == x && get_value( db, id ) == y
            return id
        end
    end
    return -1
end

function ensure_contains_values!( db :: AbstractDB, x :: Vec, y :: Vec ) :: Int
    x_pos = find_result(db, x,y);
    if x_pos < 0
        x_pos = new_result!(db, x, y);
    end
    return x_pos
end

# this should be overwritten to make it more efficient. 
# E.g., `new_result!` can store indices of results where `y` is empty
function eval_missing!( db :: AbstractDB{F}, mop :: AbstractMOP ) :: Nothing where F
    missing_ids = Int[]
    for id = eachindex(db)
        res = get_result( db, id )
        if !has_valid_value(res)
            push!(missing_ids, id)
        end
    end

    # evaluate everything in one go to exploit parallelism
    eval_sites = [ get_site( db, id ) for id in missing_ids ]
    eval_values = eval_and_sort_objectives.(mop, eval_sites)

    @assert length(eval_sites) == length(eval_values) == length(missing_ids) "Number of evaluation results does not match."
    for (i,id) in enumerate(missing_ids)
        set_value!( db, id, eval_values[i] )
    end
    
    return nothing
end

# NOTE the modifying methods require the `get_site` and `get_value`
# to return mutable arrays for result. Can we ensure this?

function scale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for id ∈ eachindex(db)
        scale!( get_site( db, id ), mop );
    end
    return nothing
end

function unscale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for id ∈ eachindex(db)
		unscale!( get_site( db, id), mop )
    end
    return nothing
end

function apply_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for id ∈ eachindex(db)
		apply_internal_sorting!( get_value(db, id), mop )
    end
    return nothing 
end

function reverse_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing 
    for id ∈ eachindex(db)
		reverse_internal_sorting!( get_value(db, id), mop )
    end
    nothing  
end

function transform!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing 
    if !is_transformed(db)
        scale!( db, mop);
        apply_internal_sorting!( db, mop );
        set_transformed!(db, true)
    end
    nothing
end
function untransform!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing 
    if is_transformed(db)
        unscale!( db, mop);
        reverse_internal_sorting!( db, mop );
        set_transformed!(db, false)
    end
    nothing
end

#=
function merge( db1 :: T, db2 :: T ) :: T where{T <: AbstractDB}
	new_db = T()
	if is_transformed(db1) == is_transformed(db2)
		for db in [ db1, db2 ]
			for id = Base.eachindex( db )
				new_result!( new_db, get_site( db, id ), get_value(db, id) )
			end
		end
		return new_db
	else
		error("Cannot merge two databases where only on is (un)transformed.")
	end
end
=#

"Return a new database of same 'base' type but with different saveable type."
function copy_db( old_db :: DBT, saveable_type :: Type ) where DBT <: AbstractDB 
    try
        base_type = @eval $(DBT.name.name)
        new_db = init_db(base_type, get_precision(old_db), saveable_type )
        for id = eachindex(old_db)
            res_id = get_result( old_db, id )
            new_result!(new_db, get_site(res_id), get_value(res_id) )
        end
        @logmsg loglevel2 "Copied database with new saveable type."
        return new_db
    catch
        @logmsg loglevel2 "Failed to copy database with new saveable type."
        return old_db 
    end
end