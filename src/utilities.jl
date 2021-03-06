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
    for (i,var_bounds) ??? enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            x[i] -= var_bounds[1]
            x[i] /= ( var_bounds[2] - var_bounds[1] )
        end
    end
    nothing
end

function _scale( x, lb, ub )
    w = ub .- lb
    _w = [ isinf(??) ? 1 : ?? for ??=w ]
    _lb = [ isinf(???) ? 0 : ??? for ???=lb ]
    return _scale_lb_w(x, _lb, _w)
end

function _unscale!( x_scaled, lb, ub )
    for (i,var_bounds) ??? enumerate(zip( lb, ub ))
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

function _intersect_bound_vec( 
    x :: AbstractVector{F}, b, dir, dir_nz = .!(iszero.(dir));
    sense = :lb
    ) where F
    
    isempty( b ) && return F[]

    d = dir[dir_nz]
    tmp = b[dir_nz] .- x[dir_nz]
    tmp_z = iszero.(tmp)
    tmp_nz = .!(tmp_z)
    ??_intersect = tmp[tmp_nz] ./ d[tmp_nz]
    
    _d = d[tmp_z]
    if isempty( _d )
        return ??_intersect
    end

    finf = F(Inf)
    fzero = F(0)
    ??_onbound = if sense == :lb
        F[ ?? > 0 ? finf : fzero for ?? = _d ] 
    else
        F[ ?? < 0 ? finf : fzero for ?? = _d ] 
    end
    return [??_intersect; ??_onbound]
end

"Return the maximum or minimum stepsize ``??`` such that ``x + ??d`` 
conforms to the linear constraints ``lb ??? x+??d ??? ub`` and ``A(x+??d) - b ??? 0``."
function _intersect_bounds( x :: AbstractVector{R}, d, lb = [], ub = [], 
        A_eq = [], b_eq = [], A_ineq = [], b_ineq = []; 
        ret_mode = :pos, impossible_val = 0, _eps = 0 ) where R <: Real
	
    # TODO can we pass precalculated `Ax` values for `A_eq` and `A_ineq`
    n_vars = length(x)
    T = Base.promote_type(R, MIN_PRECISION )
    EPS = _eps < 0 ? eps( T ) : T(_eps) # "safety buffer" for box constraints

	IMPOSSIBLE_VAL = R( impossible_val )
	# TODO instead of returning 0, we could error 
	
    if iszero(d)
        return T(Inf)
    end

	if isempty( A_eq )
		# only inequality constraints
	
		d_zero_index = iszero.(d)
		d_nz = .!( d_zero_index )

		# lb <= x + ??d - ??  ???  ??_i = (lb[i] - x[i] + ??) / d[i]
		??_lb = _intersect_bound_vec( x, lb, d, d_nz; sense = :lb)

		# x + ?? d + ?? <= ub  ???  ??_i = (ub[i] - x[i] - ??) / d[i]
		??_ub = _intersect_bound_vec( x, ub, d, d_nz; sense = :ub)

		# linear inequality constraint intersection
        ??_ineq = if isempty(A_ineq)
            T[] 
        else
            ineq_bound = isempty( b_ineq ) ? zeros( T, n_vars ) : b_ineq
		    _intersect_bound_vec( A_ineq*x, ineq_bound, A_ineq*d; sense = :ub ) 
        end
		?? = [ ??_lb; ??_ub; ??_ineq ]

		if isempty(??)
			return ret_mode == :neg ? typemin(T) : typemax(T)
		else
			??_non_neg_index = ?? .>= 0
			??_neg_index = .!??_non_neg_index #?? .< 0

			?????_array = ??[ ??_non_neg_index ]
			?????_array = ??[ ??_neg_index ]
			
			if ret_mode in [:pos, :absmax]
				??_pos = isempty( ?????_array ) ? T(0) : minimum( ?????_array )
				ret_mode == :pos && return ??_pos
			end
			
			if ret_mode in [:neg, :absmax]
		        ??_neg = isempty( ?????_array ) ? T(0) : maximum( ?????_array )
				ret_mode == :neg && return ??_neg
			end
            
            if ret_mode == :absmax
                if abs(??_pos) >= abs(??_neg)
                    return ??_pos
                else
                    return ??_neg
                end
            elseif ret_mode == :both 
                return ??_neg, ??_pos 
            end
		end
	
	else
		# there are equality constraints
		# they have to be all fullfilled and we loop through them one by one (rows of A_eq)
		N = size(A_eq, 1)
		n = length(d)
		_b = isempty(b_eq) ? zeros(T, N) : b_eq
		zero_tol = eps(T)

		?? = missing
		for i = 1 : N
            # a'(x+ ??d) - b = 0 ??? ?? a'd = -(a'x - b) ??? ?? = -(a'x -b)/a'd 
            ad = A_eq[i,:]'d
			if ad != 0  # abs(ad) > zero_tol ???? # TODO
				??_i = - (A_eq[i, :]'x - _b[i]) / ad
			else
                # check for primal feasibility of `x`:
				if abs(A_eq[i,:]'x .- _b[i]) > zero_tol
					return IMPOSSIBLE_VAL
				end
			end
			
			if ismissing(??)
				?? = ??_i
			else
				if !(??_i ??? ??)
					return IMPOSSIBLE_VAL
				end
			end
		end
		
		if ismissing(??)
			# only way this could happen:
			# ad == 0 for all i && x feasible w.r.t. eq const
            ?? = Inf
		end
		
		if isinf(??)
			return _intersect_bounds(x, d, lb, ub, [], [], A_ineq, b_ineq )
		end
			
		# check if x + ??d is compatible with the other constraints
		x_trial = x + ?? * d
		_b_ineq = isempty(b_ineq) ? zeros(T, n) : b_ineq
		
		lb_incompat = !isempty(lb) && any(x_trial .< lb .- EPS )
		ub_incompat = !isempty(ub) && any(x_trial .> ub .+ EPS )
		ineq_incompat = !isempty(A_ineq) && any( A_ineq * x_trial .- _b_ineq .+ EPS .> 0 )
		if lb_incompat || ub_incompat || ineq_incompat			
			return IMPOSSIBLE_VAL
		else
			if ret_mode == :pos
				?? < 0 && return IMPOSSIBLE_VAL
			elseif ret_mode == :neg
				?? >= 0 && return IMPOSSIBLE_VAL
			end
			return ??				
		end					
	end
end


# TODO manually optimize this easy function
function intersect_box( x_scaled, d_scaled, lb_scaled, ub_scaled; return_vals = :absmax )
    return _intersect_bounds( x_scaled, d_scaled, lb_scaled, ub_scaled; ret_mode = return_vals)
end

"Return lower and upper bound vectors combining global and trust region constraints."
function _local_bounds( x, ??, lb, ub )
    lb_eff = max.( lb, x .- ?? );
    ub_eff = min.( ub, x .+ ?? );
    return lb_eff, ub_eff 
end

"Local bounds vectors `lb_eff` and `ub_eff` using scaled variable constraints from `mop`."
function local_bounds( scal :: AbstractVarScaler, x :: Vec, ?? :: Union{Real, Vec} )
    lb, ub = full_bounds_internal(scal)
    return _local_bounds( x, ??, lb, ub )
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
        length(vec) > len ? ", ???" : "",
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

function _zero_for_constraints(?? :: R) where R<:Real
    T = Base.promote_type( R, MIN_PRECISION )
    return eps(T)*10
end

function constraint_violation_is_zero( ?? )
    return abs( ?? ) <= _zero_for_constraints( ?? )
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
        JuMP.@variable(lp, ?? >= 0)
        JuMP.@objective(lp, Max, ?? )
    elseif ret_mode == :neg
        JuMP.@variable(lp, ?? <= 0)
        JuMP.@objective(lp, Min, ?? )
    else
        JuMP.@variable(lp, ??)
        JuMP.@variable(lp, abs_??)
        JuMP.@constraint(lp, -abs_?? <= ??  <= abs_??)
        JuMP.@objective(lp, Max, abs_??)
    end
    
    _lb && JuMP.@constraint(lp, lb .<= x .+ ?? * d)
    _ub && JuMP.@constraint(lp, x .+ ?? * d .<= ub)
    _eq && JuMP.@constraint(lp, A_eq*(x .+ ?? * d) .== b_eq )
    _ineq && JuMP.@constraint(lp, A_ineq*(x .+ ?? * d) .== b_ineq )

    JuMP.optimize!( lp )
    
    ??_opt = JuMP.value(??)
    if isnan(??_opt) ??_opt = impossible_val end 
    
    return ??_opt
end
#=
function find_compatible_radius( n, ac :: AbstractConfig )
    ??_?? = filter_kappa_delta(ac)
    ?? = filter_mu(ac)
    ??_?? = filter_kappa_mu(ac)

    ??_max = delta_max( ac )
    norm_n = norm( n, Inf )

    # `??_?? * ??^??` is monotonically increasing in `??`.
    # what is the smallest `??` with `min(1, ??_??*??^??) = 1`?
    _?? = 1/(??_?? ^ (1/??))
    
    # first, assume that `min(1, ??_??*??^??) = ??_??*??^??`.
    # then, if we require ???n??? == ??_?? ??_?? ??^{1+??} we get 
    ??_1 = (norm_n / (??_?? * ??_??))^(1/1+??)

    if ??_1 > _??
        # the solution is inconsistent with the assumption that `min???=??_??*??^??`
        # so assume that `min??? = 1`.
        ??_2 = norm_n / ??_??
        if ??_2 < _?? 
            # again, solution is inconsistent
            ??_ret = -1
        else 
            ??_ret = ??_2
        end
    else
        ??_ret = ??_1
    end 
    
    ??_ret > 0 && ??_ret <= ??_max && return ??_ret
    return -1
end
=#

##########################################
"""
    nullify_last_row( R )

Returns matrices ``B`` and ``G`` with ``G???R = B``.
``G`` applies givens rotations to make ``R`` 
upper triangular.
``R`` is assumed to be an upper triangular matrix 
augmented by a single row.
"""
function nullify_last_row( R )
	m, n = size( R )
	#@assert LinearAlgebra.istriu( R[1:m-1, :] ) "The first rows of `R` must be upper triangular."
	G = Matrix(LinearAlgebra.I(m)) # orthogonal transformation matrix
	for j = 1 : min(m-1, n)
		## in each column, take the diagonal as pivot to turn last elem to zero
		g = LinearAlgebra.givens( R[j,j], R[m, j], j, m )[1]
		R = g*R
		G = g*G
	end
	return R, G
end