# include this file last :)


######## Stopping

function _budget_okay( mop :: AbstractMOP, ac :: AbstractConfig ) :: Bool
    max_conf_evals = max_evals( ac )
    for objf ∈ list_of_objectives(mop)
        !_budget_okay(objf,max_conf_evals) && return false
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
        | Stopped in iteration:  $(num_iterations)
    """ * (isnothing(mop) ? "" :
        "    | No. evaluations: $(num_evals(mop))" ) *
    """ 
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """
end

function is_compatible( n, Δ, ac :: AbstractConfig )   
    κ_Δ = filter_kappa_delta(ac)
    μ = filter_mu(ac)
    κ_μ = filter_kappa_mu(ac)

    return norm( n, Inf ) <= κ_Δ * Δ * min( 1, κ_μ * Δ^μ )
end

# this method is used in the algorithm and delegates the work 
function shrink_radius( ac :: AbstractConfig, Δ, steplength )
	return shrink_radius( Val( radius_update_method(ac) ), ac, Δ, steplength ) 
end

"Shrink radius according to `γ * Δ`."
function shrink_radius( ::Val{:standard}, ac, Δ, steplength )
	return Δ * _gamma_shrink( ac )
end

"Shrink radius according to `γ * ||s||`."
function shrink_radius( ::Val{:steplength}, ac, Δ, steplength )
	return steplength * _gamma_shrink( ac )
end

# this method is used in the algorithm and delegates the work 
function shrink_radius_much( ac :: AbstractConfig, Δ, steplength )
	return shrink_radius_much( Val( radius_update_method(ac) ), ac, Δ, steplength ) 
end

"Shrink radius much according to `γ * Δ`."
function shrink_radius_much( ::Val{:standard}, ac, Δ, steplength )
	return Δ * _gamma_shrink_much( ac )
end

"Shrink radius according to `γ * ||s||`."
function shrink_radius_much( ::Val{:steplength}, ac, Δ, steplength )
	return steplength * _gamma_shrink_much( ac )
end


# this method is used in the algorithm and delegates the work 
function grow_radius( ac :: AbstractConfig, Δ, steplength )
	return grow_radius( Val( radius_update_method(ac) ), ac, Δ, steplength ) 
end

"Grow radius according to `min( Δ_max, γ * Δ )`."
function grow_radius( ::Val{:standard}, ac, Δ, steplength )
	return min( get_delta_max(ac), _gamma_grow(ac) * Δ )
end 

"Grow radius according to `min( Δ_max, (γ + ||s||/Δ) * Δ )`"
function grow_radius( ::Val{:steplength}, ac, Δ, steplength )
	return min( get_delta_max(ac), ( _gamma_grow(ac) .+ steplength ./ Δ ) .* Δ )
end 

function do_radius_update(iter_data, _radius_update, algo_config, steplength)
	Δ = get_delta(iter_data)
	if _radius_update == LEAVE_UNCHANGED
		return Δ, Δ
	elseif _radius_update == GROW 
		return grow_radius( algo_config, Δ, steplength ), Δ
	elseif _radius_update == SHRINK
		return shrink_radius( algo_config, Δ, steplength ), Δ
	elseif _radius_update == SHRINK_MUCH
		return shrink_radius_much( algo_config, Δ, steplength ), Δ
	end
end

new_algo_config( :: AbstractConfig; kwargs... ) = DefaultConfig()
function new_algo_config( ac :: Union{Nothing, DefaultConfig}; kwargs... )
	if isempty( kwargs )
		return DefaultConfig()
	else
		float_types = [ eltype(v) for v in values(kwargs) if v isa AbstractFloat ]
		T = isempty(float_types) ? MIN_PRECISION : Base.promote_type( float_types... )
		return AlgorithmConfig{T}(; kwargs...)
	end
end

function new_algo_config( ac :: AlgorithmConfig; kwargs... )
	isempty( kwargs ) && return ac
	kw_keys = keys(kwargs)
	new_kw = Dict{Symbol,Any}()
	for fn in fieldnames(AlgorithmConfig)
		if fn in kw_keys
			new_kw[fn] = kwargs[fn]
		else
			new_kw[fn] = getfield( ac, fn )
		end
	end
	return AlgorithmConfig(; new_kw...)
end

function initialize_data( mop :: AbstractMOP, x0 :: Vec; 
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{SuperDB, Nothing} = nothing, kwargs... )
    
	if num_objectives(mop) == 0
		error("`mop` has no objectives!")
	end
	
	@logmsg loglevel1 """\n
    |--------------------------------------------
    | Initialization
    |--------------------------------------------"""	
	
	@logmsg loglevel1 "The evaluation counter of `mop` is reset."
	reset_evals!( mop )
	# initialize first iteration site
	@assert !isempty( x0 ) "Please provide a non-empty feasible starting point `x0`."

	# for backwards-compatibility with unconstrained problems:
	@assert num_vars(mop) > 0 "There are no variables associated with the mop."
		
	@assert length(x0) == num_vars( mop ) "Number of variables in `mop` does not match length of `x0`."

	ac = new_algo_config( algo_config; kwargs... )

	# make problem static 
	smop = MOPTyped(mop)

	# scale bound vars and ensure at least single-precision
	x = ensure_precision( x0 )
	# check box feasibility (unrelaxble)
	lb, ub = full_bounds(mop)
	if any( lb .> x ) || any( ub .< x )
		@warn "`x0` violates the box constraints. Projecting into global box domain."
		x = _project_into_box(x, lb, ub)
	end

	scal = get_var_scaler(x, mop :: AbstractMOP, ac :: AbstractConfig )
	x_scaled = transform(x, scal)
	
	# x and x_scaled need same precision, else it might to lead to 
    # inconsistencies when evaluating mop at scaled and unscaled sites;
	# needs to be done before first evaluation
	XT = Base.promote_eltype(x,x_scaled)
	x = XT.(x)
	x_scaled = XT.(x_scaled)	
	
	# initalize first objective vector, constraint vectors etc.
	@logmsg loglevel2 "Evaluating at start site."
	tmp_dict, objf_dict, eq_dict, ineq_dict = evaluate_at_unscaled_site( smop, x )

	# We check if the output dimensions for the inner functions are right
	# this is not strictly necessary, but we can give verbose errors:
	for (k,v) = pairs(tmp_dict)
		if num_outputs(k) != length(v)
			error("""
			Output dimension for $(k) and evaluation length mismatch $(length(v)).
			Has `n_out` been set correctly?.""")
		end
	end

	# build SuperDB `sdb` for functions with `NLIndex`
	groupings, groupings_dict = do_groupings( smop, ac )
	if isnothing( populated_db ) 
		sub_dbs, x_index_mapping = build_super_db( groupings, x_scaled, tmp_dict )
	else
		sdb = populated_db 
		transform!( sdb, scal )
		x_index_mapping = Dict{NLIndexTuple, Int}()
		for func_indices in all_sub_db_indices( sdb )
			db = get_sub_db( sdb, func_indices )
			vals = _flatten_mop_dict( tmp_dict, func_indices )
			x_index = ensure_contains_values!( db, x_scaled, vals )
			x_index_mapping[func_indices] = x_index
		end
		sub_dbs = sdb.sub_dbs
	end

	l_e, l_i = eval_linear_constraints_at_scaled_site( x_scaled, smop, scal )
	fx, c_e, c_i = _flatten_mop_dicts( objf_dict, eq_dict, ineq_dict )
	Δ_0 = eltype(x).(get_delta_0(ac))
	id = init_iterate( IterData, x, x_scaled, fx, l_e, l_i, c_e, c_i, Δ_0, x_index_mapping )
	
	# ToDo: Am I right to assume, that for linear constraints there is no need for a filter?
	# In that case, The normal step should achieve feasibility for the true feasible set (if possible)
	# and from there we can simply proceed to descent ?
	filterT = if num_nl_eq_constraints( mop ) + num_nl_ineq_constraints( mop ) > 0
		filter_type( ac )
	else
		DummyFilter
	end
	filter = init_empty_filter( filterT, fx, l_e, l_i, c_e, c_i; shift = filter_shift(ac) )

	init_stamp_content = get_saveable( IterSaveable, id; 
		rho = -Inf, omega = -Inf, steplength = -Inf, iter_counter = 0, it_stat = INITIALIZATION
	)
	sdb = SuperDB(; sub_dbs, iter_data = [init_stamp_content,] ) #, get_saveable_type(IterSaveable, id)[] )

	# initialize surrogate models
	sc = init_surrogates(smop, scal, id, ac, groupings, groupings_dict, sdb )

	return (smop, id, sdb, sc, ac, filter, scal)
end

function restoration(iter_data :: AbstractIterate, data_base,
		mop :: AbstractMOP, algo_config :: AbstractConfig, 
		filter :: AbstractFilter, scal :: AbstractVarScaler;
		r_guess_scaled = nothing, θ_k 
	)
	
	# TODO This is a very expensive task:
	# We are minimizing the constraint violation `θ`
	# by using the true constraint functions and NLopt.
	# Maybe we should instead do this with Morbit itself?

	x = get_x( iter_data )
	Xet = eltype(x)
	n_vars = length(x)	
	r0 = if isnothing( r_guess_scaled ) 
		zeros_like(x)
	else
		x - untransform( get_x_scaled(iter_data) .+ r_guess_scaled, scal )
	end
	
	A_eq, b_eq = get_eq_matrix_and_vector( mop )
	A_ineq, b_ineq = get_ineq_matrix_and_vector( mop )

	optim_objf = function( r, g )
		ξ = x .+ r
		c_e = eval_nl_eq_constraints_to_vec_at_unscaled_site(mop, ξ)
		c_i = eval_nl_ineq_constraints_to_vec_at_unscaled_site(mop, ξ)
		l_e = A_eq * ξ .+ b_eq
		l_i = A_ineq * ξ .+ b_ineq 
		return compute_constraint_val( filter, l_e, l_i, c_e, c_i)
	end

	lb, ub = full_bounds(mop)
	opt = NLopt.Opt(:LN_COBYLA, n_vars )
	opt.lower_bounds = collect(lb)	# vectors needed (not tuples)
	opt.upper_bounds = collect(ub)
	opt.min_objective = optim_objf
	opt.ftol_rel = 1e-3
	opt.stopval = _zero_for_constraints(θ_k)
	opt.maxeval = 500 * n_vars
	minθ, _rfin, ret = NLopt.optimize( opt, r0 )
	if ret in NLOPT_SUCCESS_CODES
		rfin = Xet.(_rfin)

		x_r = x .+ rfin
		x_r_scaled = transform(x_r, scal)

		tmp_dict, objf_dict, eq_dict, ineq_dict = evaluate_at_unscaled_site( mop, x_r )
		fx_r, c_e_r, c_i_r = _flatten_mop_dicts( objf_dict, eq_dict, ineq_dict )
		l_e_r, l_i_r = eval_linear_constraints_at_unscaled_site( x_r, mop )
		
		x_indices_r = put_eval_result_into_db!( data_base, tmp_dict, x_r_scaled )

		return (minθ, x_r, x_r_scaled, fx_r, c_e_r, c_i_r, l_e_r, l_i_r, x_indices_r)
	else
		return nothing
	end
end

function find_normal_step(iter_data :: I, data_base :: SuperDB, 
	mop :: AbstractMOP, sc :: SurrogateContainer, algo_config :: AbstractConfig, 
	filter :: AbstractFilter, scal :: AbstractVarScaler;
	iter_counter, last_it_stat :: ITER_TYPE, θ_k
) :: Tuple{Symbol, I} where I<:AbstractIterate
	
	@logmsg loglevel2 "Trying to find a normal step."
	x = get_x(iter_data)
	fx = get_fx(iter_data)

	_SWITCH_last_iter_restoration = (last_it_stat == RESTORATION)
	
	if _SWITCH_last_iter_restoration
		# if we performed a restoration step last, 
		# we allow for the radius to be increased, if necessary
		n, _Δ = compute_normal_step( mop, scal, iter_data, data_base, sc, algo_config; variable_radius = true )
	else
		n, _Δ = compute_normal_step( mop, scal, iter_data, data_base, sc, algo_config; variable_radius = false )
	end
	if _Δ > get_delta( iter_data )
		# if the radius has to be increased for a compatible normal step,
		# the surrogates will no longer be considered fully linear
		set_delta!( iter_data, _Δ )
		set_fully_linear!( sc, false )
	end

	# we now have to check if `n` is a *compatible* normal step
	# if that is the case, we can proceed in `iterate!`
	# if not, we have to perform an restoration iteration 
	#   * for nonlinearly constrained problems, this means calling `restoration`
	#     and we provide `n` as an intial guess for the restoration step `r`
	#   * for completely linearly constrained problems, `n` itself can 
	#     be used for restoration
	
	_SWITCH_perform_linear_restoration = false # this is used to avoid nonlinear restoration for linearly constrained problems
	_SWITCH_perform_restoration = false 
	_SWITCH_exit_with_infeasible = false 
	
	r_guess = zeros_like( x )	 # initial guess for restoration step

	_not_isnan_n = any(isnan.(n))
	if !is_compatible(n, get_delta(iter_data), algo_config )
		if _SWITCH_last_iter_restoration
			# last iteration already tried to restore feasibility 
			# but we still cannot find a suitable normal step ⇒ exit 
			_SWITCH_exit_with_infeasible = true 
		else
			if num_nl_constraints( mop ) == 0 
				if _not_isnan_n
					# we at least found a normal step, and can use it 
					# for restoration
					_SWITCH_perform_linear_restoration = true
				else 
					_SWITCH_exit_with_infeasible = true
				end
			else
				_SWITCH_perform_restoration = true
				if _not_isnan_n r_guess = n end 
			end
		end
	end

	if _SWITCH_perform_restoration
		@logmsg loglevel2 "Performing restoration for feasibility."
		add_entry!( filter, x, (θ_k, compute_objective_val(filter,fx)) )

		restoration_results = restoration(
			iter_data, data_base, mop, algo_config, filter, scal; 
			r_guess_scaled = r_guess, θ_k
		)

		if !isnothing(restoration_results)
			θ_r, x_r, x_r_scaled, fx_r, c_e_r, c_i_r, l_e_r, l_i_r, x_indices_r = restoration_results
			if is_acceptable( (θ_r, fx_r), filter )
				@logmsg loglevel2 "Found an acceptable restoration step with θ_r = $(θ_r). Next iteration."
				iter_data_r = init_iterate( I, x_r, x_r_scaled,
					fx_r, l_e_r, l_i_r, c_e_r, c_i_r, get_delta( iter_data ), x_indices_r 
				)

				return :restoration, iter_data_r 
			end
		end
		
		# no acceptable restoration step could be found ⇒ exit
		_SWITCH_exit_with_infeasible = true
	end

	if _SWITCH_exit_with_infeasible 
		@logmsg loglevel1 "Exiting because we could not find a suitable feasible iterate."
		return :exit, iter_data
	end

	# if we are here, we have a normal step and can add it to `x`
	x_n_scaled = get_x_scaled( iter_data ) .+ n
	x_n_unscaled = untransform( x_n_scaled, scal )

	@logmsg loglevel2 "The normal step is compatible, \n\t\tx_n = $(_prettify(x_n_unscaled))"
	
	# update all values and constraint_violations
	tmp_dict, objf_dict, eq_dict, ineq_dict = evaluate_at_unscaled_site( mop, x_n_unscaled )
	fx_n, c_e_n, c_i_n = _flatten_mop_dicts( objf_dict, eq_dict, ineq_dict )
	l_e_n, l_i_n = eval_linear_constraints_at_unscaled_site( x_n_unscaled, mop )
	
	x_indices_n = put_eval_result_into_db!( data_base, tmp_dict, x_n_scaled )

	iter_data_n = init_iterate( I, x_n_unscaled, x_n_scaled,
			fx_n, l_e_n, l_i_n, c_e_n, c_i_n, get_delta( iter_data ), x_indices_n )

	if _SWITCH_perform_linear_restoration
		return :restoration, iter_data_n
	end
	
	return :continue_iteration, iter_data_n
end

function criticality_routine(
	iter_data :: I, data_base :: SuperDB, 
	mop :: AbstractMOP, sc :: SurrogateContainer, algo_config :: AbstractConfig, 
	filter :: AbstractFilter, scal :: AbstractVarScaler;
	iter_counter, last_it_stat :: ITER_TYPE, _fully_linear_sc, ω
) where I<:AbstractIterate

	μ = _mu( algo_config )
	γ_c = _gamma_crit( algo_config )
	β = max( _beta( algo_config ), μ )
	
	@logmsg loglevel1 "Entered Criticallity Test."
	_SWITCH_do_loops = true
	if !_fully_linear_sc
		@logmsg loglevel1 "Ensuring that all models are fully linear."

		update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; 
			ensure_fully_linear = true )
		
		# are we still critical?
		ω, ω_data = get_criticality(mop, scal, iter_data, iter_data, data_base, sc, algo_config)
		
		_SWITCH_do_loops = if !fully_linear(sc)
			@logmsg loglevel2 "Could not make all models fully linear. Trying one last descent step."
			false
		else 
			all( get_delta( iter_data ) .> μ * ω)
		end
	end
	
	if _SWITCH_do_loops
		_SWITCH_exit_critical = false	
		num_critical_loops = 0

		Δ = get_delta( iter_data )
		Δ_0 = Δ

		while all(Δ .> μ * ω )
			@logmsg loglevel2 "Criticality loop $(num_critical_loops + 1)." 
			
			# check criticality loop stopping criteria
			if num_critical_loops >= max_critical_loops(algo_config)
				@logmsg loglevel1 "Maximum number ($(max_critical_loops(algo_config))) of critical loops reached. Exiting..."
				_SWITCH_exit_critical = true 
				break
			end
			if !_budget_okay(mop, algo_config)
				@logmsg loglevel1 "Computational budget exhausted. Exiting … "
				_SWITCH_exit_critical = true # returning with CRITICAL here only because I'm lazy 
				break
			end
			
			# shrink radius
			Δ = γ_c .* Δ				
			
			# make models fully linear on smaller trust region
			update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = true )

			# (re)calculate criticality
			ω, ω_data = get_criticality( mop, scal, iter_data, iter_data, data_base, sc, algo_config )
			num_critical_loops += 1

			if Δ_abs_test( Δ, algo_config ) || 
				ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
				_SWITCH_exit_critical = true
				break
			end
			if !fully_linear(sc)
				@logmsg loglevel2 "Could not make all models fully linear."
				_SWITCH_exit_critical = true 
				break
			end
		end#while

		# we have fully linear surrogates here AND Δ <= μ * ω so we would not 
		# enter the criticality test again if we were to restart the iteration
		
		@logmsg loglevel1 """
			Exiting after $(num_critical_loops) loops with 
			ω = $(ω) and Δ = $(Δ))."""

		set_delta!(iter_data, min( Δ_0, max( β * ω, Δ) ) )	
		if _SWITCH_exit_critical
			return :exit, iter_data, ω, ω_data
		end

	end#if _SWITCH_do_loops
	# if we are here, we deem the point not critical (enough)
	return :continue, iter_data, ω, ω_data
end

function iterate!( iter_data :: AbstractIterate, data_base :: SuperDB, 
		mop :: AbstractMOP, sc :: SurrogateContainer, algo_config :: AbstractConfig, 
		filter :: AbstractFilter = DummyFilter(), _scal :: AbstractVarScaler = nothing;
		iter_counter :: Int = 1, last_it_stat :: ITER_TYPE = ACCEPTABLE, logger = Logging.current_logger(), 
	)
	
	Logging.with_logger( logger ) do 

	x = get_x( iter_data )
	fx = get_fx( iter_data )

	# check (some) stopping conditions 
	# (rest is done at end of this function, when trial point is known)
	if iter_counter > max_iter(algo_config)
        @logmsg loglevel1 "Stopping. Maximum number of iterations reached."
        return MAX_ITER, EARLY_EXIT, _scal, iter_data
    end

    if !_budget_okay(mop, algo_config)
        @logmsg loglevel1 "Stopping. Computational budget is exhausted."
        return BUDGET_EXHAUSTED,EARLY_EXIT, _scal, iter_data
    end

    if Δ_abs_test( get_delta( iter_data ), algo_config )
        return TOLERANCE, EARLY_EXIT,_scal, iter_data
    end

	# read algorithm parameters from config 
	ν_success = _nu_success( algo_config )
	ν_accept = _nu_accept( algo_config )
	μ = _mu( algo_config )
	β = max( _beta( algo_config ), μ )
	ε_c = _eps_crit( algo_config )

	SAVEABLE_TYPE = get_saveable_type( data_base )

    @logmsg loglevel1 """\n
        |--------------------------------------------
        |Iteration $(iter_counter)
        |--------------------------------------------
        |  Current trust region radius is $(get_delta( iter_data )).
        |  Current number of function evals is $(num_evals(mop)).
        |  Iterate is $(_prettify(x))
        |  Values are $(_prettify(fx))
        |--------------------------------------------"""
	
	# obtain current variable scaler and transform data
	scal = new_var_scaler( get_x_scaled(iter_data), _scal, mop, sc, algo_config, iter_counter <= 1 ) 
		
	if _scal != scal
		@logmsg loglevel2 "Applying new scaling to database."
		if !isnothing(_scal)
			untransform!( data_base, _scal )
			transform!( data_base, scal )
		else
			db_scaler = combined_untransform_transform_scaler( _scal, scal )
			transform!( data_base, db_scaler )
		end
		iter_data = _init_iterate( IterData,
    		get_x(iter_data), transform(get_x(iter_data), scal) , get_fx(iter_data),
			get_eq_const(iter_data), get_ineq_const(iter_data),
    		get_nl_eq_const(iter_data), get_nl_ineq_const(iter_data), get_delta(iter_data),
			get_x_index_dict(iter_data)
		)
	end

    # update surrogate models
    if iter_counter > 1
        if last_it_stat == MODELIMPROVING 
            improve_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = false );
		else
            update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = false );
        end
    end

	# compute constraint violation
	θ_k = compute_constraint_val( filter, iter_data )
	@logmsg loglevel1 "Constraint Violation is $(θ_k)"
	
	if !( constraint_violation_is_zero(θ_k) )
		# if necessary, compute normal step:
		status, iter_data_n = find_normal_step(
			iter_data, data_base, mop, sc, algo_config, filter, scal; 
			iter_counter, last_it_stat, θ_k
		)
		if status == :exit
			return INFEASIBLE, EARLY_EXIT, scal, iter_data
		elseif status == :restoration
			stamp_content = get_saveable( SAVEABLE_TYPE, iter_data_n; 
				rho = -Inf, omega = -Inf, steplength = -Inf, 
				iter_counter, it_stat = RESTORATION
			)
			stamp!( data_base, stamp_content )
			return CONTINUE, RESTORATION, scal, iter_data_n
		end

		θ_n = compute_constraint_val( filter,iter_data_n )
		@logmsg loglevel3 "θ_n = $(θ_n)"
	else
		# in case, we did not need a normal step (n == 0)
		# keep the values as they are 
		θ_n = θ_k
		iter_data_n = iter_data
	end
		
    # calculate criticality value of x+n
    ω, ω_data = get_criticality(mop, scal, iter_data, iter_data_n, data_base, sc, algo_config )
    @logmsg loglevel1 "Criticality is ω = $(ω)."
    
	_θ_n_zero = constraint_violation_is_zero(θ_n)
	_θ_k_zero = constraint_violation_is_zero(θ_k)

	# if x_n is feasible and critical according to stopping conditions: return
    if _θ_n_zero && ( ω_Δ_rel_test(ω, get_delta( iter_data ), algo_config) || 
			ω_abs_test( ω, algo_config )
		)
        return CRITICAL, EARLY_EXIT, scal, iter_data_n
    end

   	#==============================
	Criticallity test
	==============================#
	_fully_linear_sc = fully_linear(sc)
	if _θ_k_zero && ω <= ε_c && (!_fully_linear_sc|| all(get_delta( iter_data ) .> μ * ω))
		# here it holds that iter_data == iter_data_n
		status, iter_data, ω, ω_data = criticality_routine(
			iter_data, data_base, mop, sc, algo_config, filter, scal;
			iter_counter, last_it_stat, _fully_linear_sc, ω)

		status == :exit && return CRITICAL, EARLY_EXIT, scal, iter_data
		iter_data_n = iter_data # just to make sure, that Δ is the same etc. (should not matter but for bugs)
	end

	#==============================
	Trial Point 
	==============================#
	# Calculation of trial point …
	ω, x_trial_scaled, mx_trial, _ = compute_descent_step(
		mop, scal, iter_data, iter_data_n, data_base, sc, algo_config, ω, ω_data 
	)

	x_scaled = get_x_scaled( iter_data )
	x_trial_unscaled = untransform( x_trial_scaled, scal )

	# … and evaluation of Objective and Surrogates:
	tmp_dict, objf_dict, eq_dict, ineq_dict = evaluate_at_unscaled_site( mop, x_trial_unscaled )
	fx_trial, c_e_trial, c_i_trial = _flatten_mop_dicts( objf_dict, eq_dict, ineq_dict )
	l_e_trial, l_i_trial = eval_linear_constraints_at_scaled_site( x_trial_scaled, mop, scal )
	# (put new values in data base)
	new_x_indices = put_eval_result_into_db!( data_base, tmp_dict, x_trial_scaled )
	
	mx = eval_container_objectives_at_scaled_site(sc, scal, x_scaled )
	mx_trial = eval_container_objectives_at_scaled_site(sc, scal, x_trial_scaled)
	
	θ_trial = compute_constraint_val( filter, l_e_trial, l_i_trial, c_e_trial, c_i_trial )
	fx_trial_filter_val = compute_objective_val(filter,fx_trial)
	
	#steplength = norm(untransform( x_scaled .- x_trial_scaled, scal ), Inf ) 
	steplength = norm( x_scaled .- x_trial_scaled, Inf ) 	# TODO transformed steplength????
	@logmsg loglevel2 """
	Testing step of length $steplength with trial point 
	x₊ = $(_prettify( x_trial_unscaled, 10)) ⇒
	| f(x)  | $(_prettify(fx))
	| f(x₊) | $(_prettify(fx_trial))
	| m(x)  | $(_prettify(mx))
	| m(x₊) | $(_prettify(mx_trial))
	The error betwenn f(x) and m(x) is $(sum(abs.(fx .- mx))).
	$(strict_acceptance_test(algo_config) ? "All" : "One") of the components must decrease."""

	#========================
	Acceptance Tests
	========================#	
	_SWITCH_is_acceptable_for_filter = is_acceptable(
		(θ_trial, fx_trial_filter_val), 
		filter, 
		(θ_k, compute_objective_val(filter,fx))
	)

	# we only need to compute ρ and ω(x) - ω(x₊) ≥ κ_ψ θ^ψ IF 
	# the trial point is acceptable for F ∪ {x}
	if _SWITCH_is_acceptable_for_filter
		if strict_acceptance_test( algo_config )
			model_denom = (mx .- mx_trial)
			_ρ = minimum( (fx .- fx_trial) ./ model_denom )
		else
			model_denom = (maximum(mx) - maximum(mx_trial))
			_ρ = (maximum(fx) - maximum( fx_trial ))/ model_denom
		end
		_SWITCH_good_decrease = all( 
			model_denom .>= filter_kappa_psi(algo_config) * θ_k^filter_psi(algo_config) 
		)
	else
		_ρ = NaN16
		_SWITCH_good_decrease = false
	end
	ρ = isnan(_ρ) ? -Inf : _ρ
	
	_iteration_classification = ACCEPTABLE
	_radius_update = LEAVE_UNCHANGED
	_SWITCH_accept_trial_point = true
		
	if _SWITCH_is_acceptable_for_filter
		if _SWITCH_good_decrease
			# the trial point is both acceptable for the filter
			# and the model decrease is large compared with the constraint violation
			# From here we can follow the logic for the UNCONSTRAINED TRM algorithm
			if ρ >= ν_success
				_SWITCH_accept_trial_point = true 
				_iteration_classification = SUCCESSFULL
				if get_delta(iter_data) < β * ω	
					_radius_update = GROW
				end
			else # ρ < ν_success
				if fully_linear( sc )
					if ρ >= ν_accept
						_SWITCH_accept_trial_point = true
						_iteration_classification = ACCEPTABLE
						_radius_update = SHRINK
					else
						_SWITCH_accept_trial_point = false 
						_iteration_classification = INACCEPTABLE
						_radius_update = SHRINK_MUCH
					end
				else
					# ρ < ν_success AND models not fully linear
					_SWITCH_accept_trial_point = false
					_iteration_classification = MODELIMPROVING
					_radius_update = LEAVE_UNCHANGED
				end
			end
		else
			# if the model decrease is small compared to constraint violation 
			_SWITCH_accept_trial_point = true
			_iteration_classification = FILTER_ADD
			_radius_update = ρ >= ν_success ? GROW : LEAVE_UNCHANGED
		end
	else
		# trial point not acceptable for filter
		_SWITCH_accept_trial_point = false
		_iteration_classification = FILTER_FAIL
		_radius_update = SHRINK_MUCH
	end

	#========================
	Updates
	========================#	

	if _iteration_classification == FILTER_ADD
		add_entry!( filter, x_trial_unscaled, (θ_trial, fx_trial_filter_val) )
	end

	Δ, Δ_old = do_radius_update(iter_data, _radius_update, algo_config, steplength)

	next_iterate = if _SWITCH_accept_trial_point
		init_iterate( IterData, 
			x_trial_unscaled, x_trial_scaled, fx_trial, l_e_trial, l_i_trial,
			c_e_trial, c_i_trial, Δ, new_x_indices )
	else
		set_delta!( iter_data, Δ)	
		iter_data	
	end

	@logmsg loglevel1 """
		ρ = $(ρ)
		θ_+ = $(θ_trial)
		The trial point was $(_SWITCH_accept_trial_point ? "" : "not ")accepted.
		The iteration is $(_iteration_classification).
		Moreover, the radius was updated as below:
		old radius : $(Δ_old)
		new radius : $(Δ)) ($(round(Δ/Δ_old * 100; digits=1)) %)"""

	stamp_content = get_saveable( SAVEABLE_TYPE, next_iterate; 
		rho = Float64(ρ), omega = Float64(ω), 
		steplength = Float64(steplength), iter_counter, it_stat = _iteration_classification 
	)
	stamp!( data_base, stamp_content )

	if ( x_tol_rel_test( x, x_trial_unscaled, algo_config  ) || 
			x_tol_abs_test( x, x_trial_unscaled, algo_config ) ||
			f_tol_rel_test( fx, fx_trial, algo_config  ) ||
			f_tol_abs_test( fx, fx_trial, algo_config ) 
		)
		return TOLERANCE, _iteration_classification, scal, next_iterate
	end

	return CONTINUE, _iteration_classification, scal, next_iterate

	end# with_logger
end

function optimize( mop :: AbstractMOP, x0 :: Vec;
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{SuperDB, Nothing} = nothing,
	verbosity :: Int = 0, kwargs... )

	logger = if verbosity >= 0 
		get_morbit_logger( LogLevel(-verbosity) )
	else
		Logging.current_logger()
	end

	Logging.with_logger( logger ) do 
		
		mop, iter_data, super_data_base, sc, ac, filter, scal = initialize_data(
			mop, x0; algo_config, populated_db, kwargs...)
		@logmsg loglevel1 _stop_info_str( ac, mop )

		@logmsg loglevel1 "Entering main optimization loop."
		
		ret_code = CONTINUE
		iter_counter = 1
		it_stat = ACCEPTABLE
		while ret_code == CONTINUE
			ret_code, it_stat, scal, iter_data = iterate!(
				iter_data, super_data_base, mop, sc, ac, filter, scal;
				iter_counter, last_it_stat = it_stat 
			)
			iter_counter += 1
		end# while

		ret_x, ret_fx = get_return_values( iter_data )
	
		@logmsg loglevel1 _fin_info_str(iter_data, mop, ret_code, iter_counter - 1 )

		if untransform_final_database( ac )
			untransform!( super_data_base, scal )
		end

		return ret_x, ret_fx, ret_code, super_data_base, iter_data, filter
	end
end# function optimize

