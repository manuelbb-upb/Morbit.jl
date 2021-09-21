@with_kw mutable struct ArrayDB{
	R <: AbstractResult, I <: NothingOrMeta} <: AbstractDB{R,I}
	
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

Base.length( db :: ArrayDB ) = db.num_entries

function init_db(:: Type{<:ArrayDB}, R :: Type{<:AbstractResult},
		I :: Type{<:NothingOrMeta})
	return ArrayDB{R, I}()
end

is_transformed( db :: ArrayDB ) = db.transformed

function set_transformed!( db :: ArrayDB, val :: Bool )
	db.transformed = val 
	return nothing 
end

get_ids( db :: ArrayDB ) = Base.eachindex( db.res )

function get_result( db :: ArrayDB, id :: Int)
	return db.res[id]
end

function next_id( db :: ArrayDB) :: Int
	return db.num_entries + 1
end

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

function _add_result!( db :: ArrayDB{R,I}, res :: R ) where{R,I}
	push!(db.res, res)
	return nothing
end

function _missing_ids(db :: ArrayDB)
	return db.unevaluated_ids
end

function stamp!( db :: ArrayDB{R,I}, ids :: I) where{R,I}
	push!(db.iter_info, ids)
	return nothing
end

###########################################
struct MockDB{R,I} <: AbstractDB{R,I} end
init_db( :: MockDB, R, I, args... ) = MockDB{R,I}()

####################################################
@with_kw struct SuperDB{ 
		T <: NothingOrSaveable, 
		ST <: AbstractDict{FunctionIndexTuple,<:AbstractDB} 
	} <: AbstractSuperDB
    sub_dbs :: ST
    iter_data :: Vector{T} = T[]
end

all_sub_db_indices( sdb :: SuperDB ) = keys(sdb.sub_dbs)
get_sub_db( sdb :: SuperDB, key_indices :: FunctionIndexTuple ) = sdb.sub_dbs[key_indices]

function stamp!( sdb :: SuperDB, ids :: NothingOrSaveable)
	push!(sdb.iter_data, ids)
	return nothing
end

get_saveable_type( sdb :: SuperDB{T,ST}) where{T,ST} = T