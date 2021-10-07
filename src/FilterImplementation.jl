struct DummyFilter <: AbstractFilter end

init_empty_filter( :: Type{<:DummyFilter }, args...; kwargs ... ) = DummyFilter()

struct FilterVal{XType, CType, FXType}
	site :: XType
	c :: CType
	fx :: FXType
end

@with_kw struct MaxFilter{XType, CType, FXType} <: AbstractFilter
	dict :: Dict{Int, FilterVal{XType,CType,FXType}} = Dict()
	shift :: Float64 = 1e-3
end

get_all_filter_ids( filter :: MaxFilter ) = keys(filter.dict)
get_site( filter :: MaxFilter, id :: Int ) = filter.dict[id].site
function get_values( filter :: MaxFilter, id :: Int ) 
	entry = filter.dict[id]
	return (entry.c, entry.fx)
end

get_shift( filter :: MaxFilter ) = filter.shift

remove_entry!(filter :: MaxFilter, id :: Int) = delete!( filter.dict, id )

function _add_entry!(filter :: MaxFilter, site, vals )
	all_ids = get_all_filter_ids(filter)
	id = length( all_ids ) == 0 ? 1 : maximum( all_ids )
	filter.dict[id] = FilterVal( site, vals... )
	return id
end

function compute_objective_val( filter :: Union{Type{<:MaxFilter},MaxFilter}, fx )
	return maximum(fx)
end

function init_empty_filter( filter_type :: Type{<:MaxFilter}, x, fx, c_E, c_I; shift = 1e-3, kwargs... )
	_θ, _f = compute_values( filter_type, fx, c_E, c_I )
	θ = _θ - shift * _θ
	f = _f .- shift * _θ
	return MaxFilter{ typeof(x), typeof(θ), typeof(f) }(; shift)
end
