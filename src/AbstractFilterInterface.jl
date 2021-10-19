# mandatory:
get_all_filter_ids( :: AbstractFilter) = Int[]
#get_site( :: AbstractFilter, :: Int ) = MIN_PRECISION[]
get_values( :: AbstractFilter, :: Int ) = MIN_PRECISION.((NaN, NaN))
get_shift( :: AbstractFilter ) = 1e-3

remove_entry!( :: AbstractFilter, ::Int ) = nothing
_add_entry!( :: AbstractFilter, site, values ) :: Int = -1

function init_empty_filter( :: Type{<:AbstractFilter}, fx, c_E, c_I; shift = 1e-3, kwargs... ) :: AbstractFilter
    nothing
end

# defaults:
function compute_constraint_val( :: Union{Type{<:AbstractFilter}, AbstractFilter}, c_E, c_I )
	ineq_violation = isempty(c_I) ? 0 : maximum(c_I)
	eq_violation = isempty( c_E ) ? 0 : maximum(abs.(c_E))
	return max( 0, ineq_violation, eq_violation )
end

function compute_objective_val( :: Union{Type{<:AbstractFilter}, AbstractFilter}, fx )
	fx
end

# derived:
function compute_values( filter :: Union{Type{<:AbstractFilter}, AbstractFilter}, fx, c_E, c_I )
	return ( compute_constraint_val(filter, c_E, c_I), compute_objective_val(filter, fx) )
end

function add_entry!( filter:: AbstractFilter, site, values)
	θₖ, fₖ = values 
	γ_θ = get_shift(filter)
	θ = θₖ - γ_θ * θₖ
	f = fₖ .- γ_θ * θₖ
	return _add_entry!( filter, site, (θ,f) )
end

#=
function get_site_and_values( filter :: AbstractFilter, id :: Int)
	return (get_site(filter,id), get_values(filter,id))
end
=#

function is_acceptable( vals :: Tuple, filter :: AbstractFilter )
	θ,f = vals
	acceptable_flag = true
	for id = get_all_filter_ids( filter )
		θ_j , f_j = get_values( filter, id )	# actually, this is (1-γ_θ)*θ_j and (f_j - γ_θ*θ_j)
		if θ > θ_j && any( f .> f_j )			# f can be vector valued, by default we want all( f .<= f_j )
			acceptable_flag = false 
			break
		end
	end
	return acceptable_flag
end

function is_acceptable( vals :: Tuple, filter :: AbstractFilter, other_vals :: Tuple )
	γ_θ = get_shift(filter)
	θ,f = vals 
	θₖ, fₖ = other_vals 
	
	acceptable_flag = ( (θ <= (1-γ_θ) * θₖ) && all( f .<= fₖ .- γ_θ * θ_k ) )
	if acceptable_flag
		return is_acceptable( vals, filter )
	else
		return false
	end
end