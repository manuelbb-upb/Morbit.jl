struct Result{
	X <: AbstractDictionary{VarInd,<:AbstractFloat},
	E <: AbstractDictionary{AnyIndex,<:AbstractFloat}
}
	in_dict :: X
	out_dict :: E 
	
	db_id :: Int
	todo :: Base.RefValue{<:Bool} 
end

Base.broadcastable( res :: Result ) = Ref(res)

_input_precision( :: Result{X,E} ) where {X,E} = _precision(X)
_eval_precision( :: Result{X,E} ) where {X,E} = _precision(E)

# replaces: get_site
_input_vector( r :: Result ) = collect(r.in_dict)
# replaces: get_value
_eval_vector( r :: Result ) = collect(r.out_dict)

# replaces: get_id
_id( r :: Result ) = r.db_id

function toggle_todo!(r :: Result) 
	r.todo[] ⊻= true
	return nothing
end

function init_no_val_result(
	in_dict, func_indices, db_id, 
	no_val_precision :: Type{<:AbstractFloat} = MIN_PRECISION, 
)
	return Result( 
		in_dict, 
		Dictionary(FillDictionary(
			func_indices, 
			Vector{no_val_precision}()
		)),
		db_id, 
		Ref(true)
	)
end

function has_valid_eval( r :: Result )
	for v in r.out_dict
		(isnan(v) || isempty(v)) && return false
	end
	return true
end

Base.@kwdef struct DataBase{
	R<:Result,
	M<:AbstractSurrogateMeta
}
	results :: Dictionary{Int, R}
	
	iter_indices :: Vector{Int}
	
	meta_data :: Union{Nothing, Dictionary{Int,Dictionary{InnerIndexTuple, M}}} = Nothing

	is_scaled :: Base.RefValue{<:Bool} = Ref(false)
end

Base.broadcastable( db :: DataBase ) = Ref(db)
_is_scaled(db :: DataBase ) = db.is_scaled[]
_next_id(db :: DataBase) = isempty(db.results) ? 1 : maximum(keys(db.results)) + 1

add_iter_index( db :: DataBase, i :: Int ) = push!( db.iter_indices, i )
add_iter_index( db :: DataBase ) = push!(db.iter_indices, last(db.iter_indices))