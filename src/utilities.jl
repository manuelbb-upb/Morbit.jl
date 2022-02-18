# This file is included from the Main script `Morbit.jl`
# and mostly contains utility functions that do not fit anywhere else

"""
	register_func(func, func_name)

Registers the function `func` for subsequent use in a function 
expression.
Note, that using function expressions is really only advisable if 
the "base function" is truly expensive and surrogate modelling remedies 
the performance penalties from parsing strings and `@eval`ing expressions.
"""
function register_func(func, func_name :: Symbol)
	global registered_funcs
	@eval registered_funcs[$(Meta.quot(func_name))] = $func
	nothing
end

"Return the basis typename, i.e., remove all parameters from a typename."
function _typename end
_typename( T :: DataType ) = T.name.name
_typename( T :: UnionAll ) = _typename( T.body )

_ensure_vec( x :: Number ) = [x,]
_ensure_vec( x :: AbstractVector{<:Number} ) = x

# used instead of list comprehension
# works with vectors of vectors too:
flatten_vecs( x :: Number) = [x,]

function flatten_vecs(x)
	return [ e for e in Iterators.flatten(x) ]  # TODO check performance
end

function mat_from_row_vecs( row_vecs )
	return copy( transpose( hcat(row_vecs...) ) )
end

function build_super_db( groupings :: Vector{<:ModelGrouping}, x_scaled :: XT, eval_res ) where XT <: VecF
    n_vars = length(x_scaled)

    sub_dbs = Dict{NLIndexTuple, ArrayDB}()
    x_index_mapping = Dict{NLIndexTuple, Int}()
    for group in groupings 
        index_tuple = Tuple(group.indices)

        _group_vals = _flatten_mop_dict( eval_res, group.indices )
        group_vals = (Base.promote_eltype( _group_vals, MIN_PRECISION )).(_group_vals)

        res = Result(; x = SVector{n_vars}(x_scaled), y = MVector{length(group_vals)}(group_vals) )
        config_saveable_type = get_saveable_type( group.cfg, x_scaled, group_vals )
        
        sub_db = init_db( ArrayDB, typeof(res), config_saveable_type )
        set_transformed!(sub_db, true)
        x_index = ensure_contains_values!( sub_db, x_scaled, group_vals )
        
        sub_dbs[ index_tuple ] = sub_db
        x_index_mapping[ index_tuple ] = x_index
    end

    return sub_dbs, x_index_mapping
end
###################################################################
function ensure_precision( x :: X ) where X<:Real 
    _X = promote_type( X, MIN_PRECISION )
    return _X(x)
end

function ensure_precision( x :: AbstractVector{X} ) where X<:Real
    isempty(x) && return MIN_PRECISION[]
    _X = promote_type( X, MIN_PRECISION )
    return _X.(x)
end

function ensure_precision( x :: AbstractVector )
    isempty(x) && return MIN_PRECISION[]
    error("Cannot promote to a floating point type.")
end

# Scaling
function _scale!( x, lb, ub )
    for (i,var_bounds) ∈ enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            x[i] -= var_bounds[1]
            x[i] /= ( var_bounds[2] - var_bounds[1] )
        end
    end
    nothing
end

function _scale( x, lb, ub )
    w = ub .- lb
    _w = [ isinf(ω) ? 1 : ω for ω=w ]
    _lb = [ isinf(ℓ) ? 0 : ℓ for ℓ=lb ]
    return _scale_lb_w(x, _lb, _w)
end

function _unscale!( x_scaled, lb, ub )
    for (i,var_bounds) ∈ enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            # TODO: Make the component scaling memoized?
            x_scaled[i] *= (var_bounds[2] - var_bounds[1]) 
            x_scaled[i] += var_bounds[1]
        end
    end
    nothing
end

function _unscale( x_scaled, lb, ub )
    w = ub .- lb
    return [ isinf(w[i]) ? x_scaled[i] : x_scaled[i]*w[i] + lb[i] for i = eachindex(w) ]
end

function _unscale_lb_w( x_scaled, lb, w )
    return x_scaled .* w .+ lb 
end

function _scale_lb_w( x, lb, w )
    return ( x .- lb ) ./ w
end

function _project_into_box( z, lb, ub)
    return min.( max.( z, lb ), ub )
end

"Return the maximum or minimum stepsize ``σ`` such that ``x + σd`` conforms to the linear constraints."
function _intersect_bounds( x :: AbstractVector{R}, d, lb = [], ub = [], 
        A_eq = [], b_eq = [], A_ineq = [], b_ineq = []; 
        ret_mode = :pos, impossible_val = 0, _eps = -1 ) where R <: Real
	
    T = Base.promote_type(R, MIN_PRECISION )
    EPS = _eps < 0 ? eps( T ) : T(_eps)

	IMPOSSIBLE_VAL = R( impossible_val )
	# TODO instead of returning 0, we could error 
	
	if isempty( A_eq )
		# only inequality constraints
	
		d_zero_index = iszero.(d)
		d_not_zero_index = .!( d_zero_index )

		# lb <= x + σd - ε  ⇒  σ_i = (lb[i] - x[i] + ε) / d[i]
		σ_lb = if isempty(lb)
			T[]
		else
			( lb[ d_not_zero_index ] .- x[ d_not_zero_index ] .+ EPS) ./ d[ d_not_zero_index ]
		end

		# x + σ d + ε <= ub  ⇒  σ_i = (ub[i] - x[i] - ε) / d[i]
		σ_ub = if isempty(ub)
			T[]
		else
			( ub[ d_not_zero_index ] .- x[ d_not_zero_index ] .- EPS) ./ d[ d_not_zero_index ]
		end

		# linear inequality constraints
        # A * (x + d) + b .+ ε \le 0
        # A d = -Ax -b - ε
		σ_ineq = if isempty( A_ineq)
			T[] 
		else
			denom_ineq = A_ineq * d
			denom_ineq_not_zero_index = .!( iszero.( denom_ineq ) )
			if isempty(b_ineq)
				- (A_ineq[denom_ineq_not_zero_index, :] * x .- EPS) ./ denom_ineq[ denom_ineq_not_zero_index ]
			else
				- ( A_ineq[denom_ineq_not_zero_index, :] * x .+ b_ineq[denom_ineq_not_zero_index] .- EPS) ./ denom_ineq[ denom_ineq_not_zero_index ]
			end
		end

		σ = [ σ_lb; σ_ub; σ_ineq ]

		if isempty(σ)
			return ret_mode == :neg ? typemin(T) : typemax(T)
		else
			σ_non_neg_index = σ .>= 0
			σ_neg_index = .!(σ_non_neg_index)

			σ_non_neg = σ[ σ_non_neg_index ]
			σ_neg = σ[ σ_neg_index ]
			
			if ret_mode in [:pos, :absmax]
				σ_pos = isempty( σ_non_neg ) ? T(0) : minimum( σ_non_neg )
				ret_mode == :pos && return σ_pos
			end
			
			if ret_mode in [:neg, :absmax]
				σ_not_pos = isempty( σ_neg ) ? T(0) : maximum( σ_neg )
				ret_mode == :neg && return σ_not_pos
			end
			
            if ret_mode == :absmax
                if abs(σ_pos) >= abs(σ_not_pos)
                    return σ_pos
                else
                    return σ_not_pos
                end
            elseif ret_mode == :both 
                return σ_not_pos, σ_pos 
            end
		end
	
	else
		# there are equality constraints
		# they have to be all fullfilled
		N = size(A_eq, 1)
		n = length(d)
		_b = isempty(b_eq) ? zeros(T, N) : b_eq
		
		σ = missing
		for i = 1 : N
			if d[i] != 0
				σ_i = - (A_eq[i, :]'x + _b[i]) / d[i]
			else
				if A_eq[i,:]'x != 0
					return IMPOSSIBLE_VAL
				end
			end
			
			if ismissing(σ)
				σ = σ_i
			else
				if !(σ_i ≈ σ)
					return IMPOSSIBLE_VAL
				end
			end
		end
		
		if ismissing(σ)
			# only way this could happen:
			# d == 0 && x feasible w.r.t. eq const
			return IMPOSSIBLE_VAL
		end
		
		if N == 1 && abs(σ) < EPS
			# check if direction and single equality constraint are parallel?
			if d[2] / d[1] ≈ -A_eq[1,1] / A_eq[1,2]
				return _intersect_bounds(x, d, lb, ub, [], [], A_ineq, b_ineq )
			end
		end
			
		# check if x + σd is compatible with the other constraints
		x_trial = x + σ * d
		_b_ineq = isempty(b_ineq) ? zeros(T, n) : b_ineq
		
		lb_incompat = !isempty(lb) && any(x_trial .< lb .- EPS )
		ub_incompat = !isempty(ub) && any(x_trial .> ub .+ EPS )
		ineq_incompat = !isempty(A_ineq) && any( A_ineq * x_trial .+ _b_ineq .> EPS )
		if lb_incompat || ub_incompat || ineq_incompat			
			return IMPOSSIBLE_VAL
		else
			if ret_mode == :pos
				σ < 0 && return IMPOSSIBLE_VAL
			elseif ret_mode == :neg
				σ >= 0 && return IMPOSSIBLE_VAL
			end
			return σ				
		end					
	end
end


# TODO manually optimize this easy function
function intersect_box( x_scaled, d_scaled, lb_scaled, ub_scaled; return_vals = :absmax )
    return _intersect_bounds( x_scaled, d_scaled, lb_scaled, ub_scaled; ret_mode = return_vals)
end

"Return lower and upper bound vectors combining global and trust region constraints."
function _local_bounds( x, Δ, lb, ub )
    lb_eff = max.( lb, x .- Δ );
    ub_eff = min.( ub, x .+ Δ );
    return lb_eff, ub_eff 
end

"Local bounds vectors `lb_eff` and `ub_eff` using scaled variable constraints from `mop`."
function local_bounds( scal :: AbstractVarScaler, x :: Vec, Δ :: Union{Real, Vec} )
    lb, ub = full_bounds_internal(scal)
    return _local_bounds( x, Δ, lb, ub )
end

# use for finite (e.g. local) bounds only
function _rand_box_point(lb, ub, type :: Type{<:Real} = MIN_PRECISION)
    return lb .+ (ub .- lb) .* rand(type, length(lb))
end 

using Printf: @sprintf
function _prettify( vec, len :: Int = 5) :: AbstractString
    return string(
        "[",
        join( 
            [@sprintf("%.5f",vec[i]) for i = 1 : min(len, length(vec))], 
            ", "
        ),
        length(vec) > len ? ", …" : "",
        "]"
    )
end

#########################################################################

function zeros_like( x :: AbstractVector{R} ) where R<:Number 
	return zeros( R, length(x) )
end

function compute_constraint_val( filter :: AbstractFilter, iter_data :: AbstractIterate )
    return compute_constraint_val( filter,
    get_eq_const( iter_data ),
        get_ineq_const( iter_data ),
        get_nl_eq_const(iter_data),
        get_nl_ineq_const(iter_data)
    )
end

function _zero_for_constraints(θ :: R) where R<:Real
    T = Base.promote_type( R, MIN_PRECISION )
    return eps(T)*10
end

function constraint_violation_is_zero( θ )
    return abs( θ ) <= _zero_for_constraints( θ )
end

#########################################################################
# TODO remove 

function _intersect_bounds_jump( x :: AbstractVector{R}, d, lb = [], ub = [], 
        A_eq = [], b_eq = [], A_ineq = [], b_ineq = []; 
        ret_mode = :pos, impossible_val = 0, _eps = -1.0 ) where R <: Real

    _lb = !isempty(lb) 
    _ub = !isempty(ub)
    _eq = !(isempty(A_eq) || isempty(b_eq))  # TODO warn if only one is supplied
    _ineq = !(isempty(A_ineq) || isempty(b_ineq))  # TODO warn if only one is supplied

    if !(_lb || _ub || _eq || _empty_ineq)
        ret_mode == :neg && return -MIN_PRECISION(Inf)
        return MIN_PRECISION(INF)
    end

    lp = JuMP.Model( LP_OPTIMIZER )

    #JuMP.set_optimizer_attribute( lp, "polish", true );
    JuMP.set_silent(lp)

    if ret_mode == :pos 
        JuMP.@variable(lp, σ >= 0)
        JuMP.@objective(lp, Max, σ )
    elseif ret_mode == :neg
        JuMP.@variable(lp, σ <= 0)
        JuMP.@objective(lp, Min, σ )
    else
        JuMP.@variable(lp, σ)
        JuMP.@variable(lp, abs_σ)
        JuMP.@constraint(lp, -abs_σ <= σ  <= abs_σ)
        JuMP.@objective(lp, Max, abs_σ)
    end
    
    _lb && JuMP.@constraint(lp, lb .<= x .+ σ * d)
    _ub && JuMP.@constraint(lp, x .+ σ * d .<= ub)
    _eq && JuMP.@constraint(lp, A_eq*(x .+ σ * d) .== b_eq )
    _ineq && JuMP.@constraint(lp, A_ineq*(x .+ σ * d) .== b_ineq )

    JuMP.optimize!( lp )
    
    σ_opt = JuMP.value(σ)
    if isnan(σ_opt) σ_opt = impossible_val end 
    
    return σ_opt
end
#=
function find_compatible_radius( n, ac :: AbstractConfig )
    κ_Δ = filter_kappa_delta(ac)
    μ = filter_mu(ac)
    κ_μ = filter_kappa_mu(ac)

    Δ_max = get_delta_max( ac )
    norm_n = norm( n, Inf )

    # `κ_μ * Δ^μ` is monotonically increasing in `Δ`.
    # what is the smallest `Δ` with `min(1, κ_μ*Δ^μ) = 1`?
    _Δ = 1/(κ_μ ^ (1/μ))
    
    # first, assume that `min(1, κ_μ*Δ^μ) = κ_μ*Δ^μ`.
    # then, if we require ‖n‖ == κ_Δ κ_μ Δ^{1+μ} we get 
    Δ_1 = (norm_n / (κ_Δ * κ_μ))^(1/1+μ)

    if Δ_1 > _Δ
        # the solution is inconsistent with the assumption that `min…=κ_μ*Δ^μ`
        # so assume that `min… = 1`.
        Δ_2 = norm_n / κ_Δ
        if Δ_2 < _Δ 
            # again, solution is inconsistent
            Δ_ret = -1
        else 
            Δ_ret = Δ_2
        end
    else
        Δ_ret = Δ_1
    end 
    
    Δ_ret > 0 && Δ_ret <= Δ_max && return Δ_ret
    return -1
end
=#
