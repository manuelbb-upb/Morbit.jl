# This File is included from the Main script `Morbit.jl`
# It does not depend on user-defined concrete types, 
# but `AbstractMOP` and `AbstractConfig` should be defined.
#
# using LinearAlgebra: norm
#
# Contains:
# * variable scaling methods 
# * methods for stopping 
# * methods for prettier printing/logging
###################################################################

function _contains_index( m :: ModelGrouping, ind :: FunctionIndex )
    return ind in m.indices
end

function do_groupings( mop :: AbstractMOP, ac :: AbstractConfig )
    if !_combine_models_by_type(ac)
        return [ ModelGrouping( [ind,], get_cfg(_get(mop,ind)) ) for ind in get_function_indices(mop) ]
    end
    groupings = ModelGrouping[]
    for objf_ind1 in get_function_indices( mop )
        objf1 = _get( mop, objf_ind1 )

        # check if there already is a group that `objf1`
        # belongs to and set `group_index` to its position in `groupings`
        group_index = -1
        for (gi, group) in enumerate(groupings)
            if _contains_index( group, objf_ind1 )
                group_index = gi
                break
            end
        end
        # if there is no group with `objf1` in it, 
        # then create a new one and set `group_index` to new, last position
        if group_index < 0
            push!( groupings, ModelGrouping(FunctionIndex[objf_ind1,], model_cfg(objf1)) )
            group_index = length(groupings)
        end
        
        group = groupings[group_index]
        
        # now, for every remaining function index, check if we can add 
        # it to the group of `objf_ind1`
        for objf_ind2 in get_function_indices( mop )
            objf2 = _get( mop, objf_ind2 )
            if objf_ind1 != objf_ind2 && combinable( objf1, objf2 ) && !_contains_index(group, objf_ind2)
                push!( group.indices, objf_ind2 )
            end
        end
    end
    return groupings
end

function build_super_db( groupings :: Vector{<:ModelGrouping}, x_scaled :: XT, eval_res ) where XT <: VecF
    n_vars = length(x_scaled)

    sub_dbs = Dict{FunctionIndexTuple, ArrayDB}()
    x_index_mapping = Dict{FunctionIndexTuple, Int}()
    for group in groupings 
        index_tuple = Tuple(group.indices)

        _group_vals = flatten_mop_dict( eval_res, group.indices )
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
    _X = promote_type( X, MIN_PRECISION )
    return _X.(x)
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

function _intersect_bounds( x :: AbstractVector{R}, d, lb = [], ub = [], 
        A_eq = [], b_eq = [], A_ineq = [], b_ineq = []; 
        ret_mode = :pos, impossible_val = 0, _eps = -1.0 ) where R <: Real
	
    T = Base.promote_type(R, MIN_PRECISION )
    EPS = _eps <= 0 ? eps( T ) : T(_eps)

	IMPOSSIBLE_VAL = R( impossible_val )
	# TODO instead of returning 0, we could error 
	
	if isempty( A_eq )
		# only inequality constraints
	
		d_zero_index = iszero.(d)
		d_not_zero_index = .!( d_zero_index )

		# lb <= x + σd + ε  ⇒  σ_i = (lb[i] - x[i]) / d[i]
		σ_lb = if isempty(lb)
			T[]
		else
			( lb[ d_not_zero_index ] .- x[ d_not_zero_index ] .- EPS) ./ d[ d_not_zero_index ]
		end

		# x + σ d <= ub  ⇒  σ_i = (ub[i] - x[i]) / d[i]
		σ_ub = if isempty(ub)
			T[]
		else
			( ub[ d_not_zero_index ] .- x[ d_not_zero_index ] .- EPS) ./ d[ d_not_zero_index ]
		end

		# linear inequality constraints
        # A * (x + d) + b .- ε =̂ 0
        # A d = -Ax -b + ε
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
				return steplength(x, d, lb, ub, [], [], A_ineq, b_ineq )
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

function eval_linear_constraints_at_unscaled_site( x, mop )
    #A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints( scal, mop )
    A_eq, b_eq = get_eq_matrix_and_vector( mop )
    A_ineq, b_ineq = get_ineq_matrix_and_vector( mop )
    return (A_eq * x .+ b_eq, A_ineq * x + b_ineq)
end

function eval_linear_constraints_at_scaled_site( x_scaled, mop, scal )
    A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints( scal, mop )
    return (A_eq * x_scaled .+ b_eq, A_ineq * x_scaled + b_ineq)
end
######## Stopping

function _budget_okay( mop :: AbstractMOP, ac :: AbstractConfig ) :: Bool
    max_conf_evals = max_evals( ac )
    for objf ∈ list_of_objectives(mop)
        if num_evals(objf) >= min( max_evals(objf), max_conf_evals ) - 1
            return false;
        end
    end
    return true
end

function f_tol_rel_test( fx :: Vec, fx⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = f_tol_rel(ac)
    if isa(tol, Real)
        ret = norm( fx .- fx⁺, Inf ) <= tol * norm( fx, Inf )
    else
        ret = all( abs.( fx .- fx⁺ ) .<= tol .* fx )
    end
    ret && @logmsg loglevel1 "Relative (objective) stopping criterion fulfilled."
    ret
end

function x_tol_rel_test( x :: Vec, x⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = x_tol_rel(ac)
    if isa(tol, Real)
        ret = norm( x .- x⁺, Inf ) <= tol * norm( x, Inf )
    else
        ret = all( abs.( x .- x⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Relative (decision) stopping criterion fulfilled."
    ret
end

function f_tol_abs_test( fx :: Vec, fx⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = f_tol_abs(ac)
    if isa(tol, Real)
        ret = norm( fx .- fx⁺, Inf ) <= tol 
    else
        ret = all( abs.( fx .- fx⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Absolute (objective) stopping criterion fulfilled."
    ret
end

function x_tol_abs_test( x :: Vec, x⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = x_tol_abs(ac)
    if isa(tol, Real)
        ret =  norm( x .- x⁺, Inf ) <= tol 
    else
        ret = all( abs.( x .- x⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Absolute (decision) stopping criterion fulfilled."
    ret
end

function ω_Δ_rel_test( ω :: Real, Δ :: VecOrNum, ac :: AbstractConfig )
    ω_tol = omega_tol_rel( ac )
    Δ_tol = delta_tol_rel( ac )
    ret = ω <= ω_tol && all( Δ .<= Δ_tol )
    ret && @logmsg loglevel1 "Realtive criticality stopping criterion fulfilled."
    ret
end

function Δ_abs_test( Δ :: VecOrNum, ac :: AbstractConfig )
    tol = delta_tol_abs( ac )
    ret = all( Δ .<= tol )
    ret && @logmsg loglevel1 "Absolute radius stopping criterion fulfilled."
    ret
end

function ω_abs_test( ω :: Real, ac :: AbstractConfig )
    tol = omega_tol_abs( ac )
    ret = ω .<= tol
    ret && @logmsg loglevel1 "Absolute criticality stopping criterion fulfilled."
    ret
end

function _stop_info_str( ac :: AbstractConfig, mop :: Union{AbstractMOP,Nothing} = nothing )
    ret_str = "Stopping Criteria:\n"
    if isnothing(mop)
        ret_str *= "No. of objective evaluations ≥ $(max_evals(ac)).\n"
    else
        for f_ind ∈ get_function_indices(mop)
            func = _get(mop, f_ind)
            ret_str *= "• No. of. evaluations of $(f_ind) ≥ $(min( max_evals(func), max_evals(ac) )).\n"
        end
    end
    ret_str *= "• No. of iterations is ≥ $(max_iter(ac)).\n"
    ret_str *= @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g ⋅ ‖ fx ‖,\n", f_tol_rel(ac) )
    ret_str *= @sprintf("• ‖ x - x⁺ ‖ ≤ %g ⋅ ‖ x ‖,\n", x_tol_rel(ac) )
    ret_str *= @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g,\n", f_tol_abs(ac) )
    ret_str *= @sprintf("• ‖ x - x⁺ ‖ ≤ %g,\n", x_tol_abs(ac) )
    ret_str *= @sprintf("• ω ≤ %g AND Δ ≤ %g,\n", omega_tol_rel(ac), delta_tol_rel(ac))
    ret_str *= @sprintf("• Δ ≤ %g OR", delta_tol_abs(ac))
    ret_str *= @sprintf(" ω ≤ %g.", omega_tol_abs(ac))
end

function is_compatible( n, Δ, ac :: AbstractConfig )   
    κ_Δ = filter_kappa_delta(ac)
    μ = filter_mu(ac)
    κ_μ = filter_kappa_mu(ac)

    return norm( n, Inf ) <= κ_Δ * Δ * min( 1, κ_μ * Δ^μ )
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

function get_return_values(iter_data)
    ret_x = get_x(iter_data)
	ret_fx = get_fx( iter_data )
    return ret_x, ret_fx 
end

function _fin_info_str(iter_data :: AbstractIterate, 
        mop = nothing, stopcode = nothing, num_iterations = -1 )
    ret_x, ret_fx = get_return_values( iter_data )
    return """\n
        |--------------------------------------------
        | FINISHED ($stopcode)
        |--------------------------------------------
        | No. iterations:  $(num_iterations)
    """ * (isnothing(mop) ? "" :
        "    | No. evaluations: $(num_evals(mop))" ) *
    """ 
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """
end

using Printf: @sprintf
function _prettify( vec :: Vec, len :: Int = 5) :: AbstractString
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
