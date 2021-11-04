struct DummyFilter <: AbstractFilter end

init_empty_filter( :: Type{<:DummyFilter }, args...; kwargs ... ) = DummyFilter()

struct FilterVal{CType, FXType}
	c :: CType
	fx :: FXType
end

@with_kw struct MaxFilter{CType, FXType} <: AbstractFilter
	dict :: Dict{Int, FilterVal{CType,FXType}} = Dict()
	shift :: Float64 = 1e-3
end

get_all_filter_ids( filter :: MaxFilter ) = keys(filter.dict)
function get_values( filter :: MaxFilter, id :: Int ) 
	entry = filter.dict[id]
	return (entry.c, entry.fx)
end

get_shift( filter :: MaxFilter ) = filter.shift

remove_entry!(filter :: MaxFilter, id :: Int) = delete!( filter.dict, id )

function _add_entry!(filter :: MaxFilter, vals )
	all_ids = get_all_filter_ids(filter)
	id = length( all_ids ) == 0 ? 1 : maximum( all_ids )
	filter.dict[id] = FilterVal( vals... )
	return id
end

function compute_objective_val( filter :: Union{Type{<:MaxFilter},MaxFilter}, fx )
	return maximum(fx)
end

function init_empty_filter( filter_type :: Type{<:MaxFilter}, fx, l_E, l_I, c_E, c_I; shift = 1e-3, kwargs... )
	_θ, _f = compute_values( filter_type, fx, l_E, l_I, c_E, c_I )
	θ = _θ - shift * _θ
	f = _f .- shift * _θ
	return MaxFilter{ typeof(θ), typeof(f) }(; shift)
end

struct StrictFilter{T} <: AbstractFilter
	inner :: T
end

compute_objective_val( filter :: Union{Type{<:StrictFilter},StrictFilter}, fx ) = fx
function init_empty_filter( filter_type :: Type{<:StrictFilter}, args...; kwargs... )
	return StrictFilter( init_empty_filter(MaxFilter, args...;kwargs...) )
end


@forward StrictFilter.inner get_all_filter_ids, get_values, get_shift, remove_entry!, _add_entry!