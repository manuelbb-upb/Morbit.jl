@with_kw mutable struct ArrayDB{
		F <:AbstractFloat, RT <: AbstractResult{F}, 
		IT <: Union{Nothing,AbstractIterSaveable} } <: AbstractDB{F}
	
	res :: Vector{RT} = RT[]

	# counter of entries 
	num_entries :: Int = 0

	transformed :: Bool = false
	
	iter_info :: Vector{IT} = IT[]

	unevaluated_ids :: Vector{Int} = []

	@assert length(res) == num_entries "Number of entries does not match `num_entries`."
end

Base.length( db :: ArrayDB ) = db.num_entries

function init_db(:: Type{<:ArrayDB}, F :: Type{<:AbstractFloat},
		IT :: Union{Type{<:AbstractIterSaveable},Type{<:Nothing}} )
	return ArrayDB{F, Result{F}, IT}()
end

is_transformed( db :: ArrayDB ) = db.transformed

function set_transformed!( db :: ArrayDB, val :: Bool )
	db.transformed = val 
	return nothing 
end

Base.eachindex( db :: ArrayDB ) ::Vector{Int} = Base.eachindex( db.res )

function get_result( db :: ArrayDB, id :: Int)
	return db.res[id]
end

function next_id( db :: ArrayDB) :: Int
	return db.num_entries + 1
end

function new_result!( db :: ArrayDB{F,RT,IT}, x :: Vec, y :: Vec, id :: Int = -1 ) where{F,RT,IT}
	new_id = id < 0 ? next_id(db) : id
	new_result = init_res( RT, x, y, new_id )
	append!(db.res, new_result)
	if !has_valid_value(new_result) 
		push!(db.unevaluated_ids, new_id)
	end
	db.num_entries += 1
	return new_id
end

function _missing_ids(db :: ArrayDB)
	return db.unevaluated_ids
end

function set_evaluated_flag!( db :: ArrayDB, id :: Int )
	if has_valid_value( get_result(db, id) )
		for (i,_id) in enumerate(db.unevaluated_ids)
			if _id == id
				deleteat!(db.unevaluated_ids,i)
				break
			end
		end
	end
	nothing
end


function stamp!( db :: ArrayDB, ids :: AbstractIterSaveable ) :: Nothing 
	push!(db.iter_info, ids)
	return nothing
end

###########################################
struct MockDB{F<:AbstractFloat} <: AbstractDB{F} end
init_db( :: MockDB, :: F, args... ) where F = MockDB{F}()