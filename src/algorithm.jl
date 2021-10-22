# include this file last :)

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

new_algo_config( :: AbstractConfig; kwargs... ) = DefaultConfig()
function new_algo_config( ac :: Union{Nothing, DefaultConfig}; kwargs... )
	if isempty( kwargs )
		return DefaultConfig()
	else
		return AlgoConfig(; kwargs...)
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

# we expect mop :: MixedMOP, but should work for static MOP if everything 
# is set up properly 
function initialize_data( mop :: AbstractMOP, x0 :: Vec; 
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractSuperDB, Nothing} = nothing, kwargs... )
    
	if num_objectives(mop) == 0
		error("`mop` has no objectives!")
	end
	
	@logmsg loglevel1 """\n
    |--------------------------------------------
    | Initialization
    |--------------------------------------------
	"""	
	
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

	# scale bound vars to unit hypercube
	# ensure at least single-precision
	x = ensure_precision( x0 )
	
	# initalize first objective vector, constraint vectors etc.
	@logmsg loglevel2 "Evaluating at start site."
	eval_res = eval_mop_at_unscaled_site( smop, x )

	scal = get_var_scaler(x, mop :: AbstractMOP, ac :: AbstractConfig )
	x_scaled = transform(x, scal)
	
	# build SuperDB `sdb`
	groupings = do_groupings( smop, ac )
	if isnothing( populated_db ) 
		sub_dbs, x_index_mapping = build_super_db( groupings, x_scaled, eval_res )
	else
		sdb = populated_db 
		transform!( sdb, scal )
		x_index_mapping = Dict{FunctionIndexTuple, Int}()
		for func_indices in all_sub_db_indices( sdb )
			db = get_sub_db( sdb, func_indices )
			vals = eval_result_to_vector( eval_res, func_indices )
			x_index = ensure_contains_values!( db, x_scaled, vals )
			x_index_mapping[func_indices] = x_index
		end
		sub_dbs = all_sub_dbs(sdb)
	end
	
	l_e, l_i = eval_linear_constraints_at_scaled_site( x_scaled, smop, scal )
	fx, c_e, c_i = eval_result_to_all_vectors( eval_res, smop )
	Δ_0 = eltype(x).(get_delta_0(ac))
	id = init_iter_data( IterData, x, fx, x_scaled, l_e, l_i, c_e, c_i, Δ_0, x_index_mapping )
	
	# ToDo: Am I right to assume, that for linear constraints there is no need for a filter?
	# In that case, The normal step should achieve feasibility for the true feasible set (if possible)
	# and from there we can simply proceed to descent ?
	filterT = if num_nl_eq_constraints( mop ) + num_nl_ineq_constraints( mop ) > 0
		filter_type( ac )
	else
		DummyFilter
	end
	filter = init_empty_filter( filterT, fx, l_e, l_i, c_e, c_i; shift = filter_shift(ac) )

    sdb = SuperDB(; sub_dbs, iter_data = typeof(get_saveable(IterSaveable,id))[] )

	# initialize surrogate models
	sc = init_surrogates(SurrogateContainer, smop, scal, id, ac, groupings, sdb )

	return (smop, id, sdb, sc, ac, filter, scal)
end

function restoration(iter_data :: AbstractIterData, data_base :: AbstractSuperDB,
		mop :: AbstractMOP, algo_config :: AbstractConfig, 
		filter :: AbstractFilter, scal :: AbstractVarScaler;
		r_guess_scaled = nothing
	)
	
	x = get_x( iter_data )
	n_vars = length(x)	
	r0 = isnothing( r_guess_scaled ) ? zeros_like(x) : untransform( r_guess_scaled, scal )
	
	θ_k = compute_constraint_val( filter, iter_data )

	# could also use
	# `eval_linear_constraints_at_scaled_site( x_scaled, smop, scal )`
	# A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints(scal, mop)
	A_eq, b_eq = get_eq_matrix_and_vector( mop )
	A_ineq, b_ineq = get_ineq_matrix_and_vector( mop )

	optim_objf = function( r, g )
		ξ = x .+ r
		c_e = eval_vec_mop_nl_eq_constraints_at_unscaled_site(mop, ξ)
		c_i = eval_vec_mop_nl_ineq_constraints_at_unscaled_site(mop, ξ)
		l_e = A_eq * ξ .+ b_eq
		l_i = A_ineq * ξ .+ b_ineq 
		return compute_constraint_val( filter, l_e, l_i, c_e, c_i)
	end

	opt = NLopt.Opt(:LN_COBYLA, n_vars )
	opt.min_objective = optim_objf
	opt.ftol_rel = 1e-3
	opt.stopval = _zero_for_constraints(θ_k)
	opt.maxeval = 500 * n_vars
	minθ, rfin, ret = NLopt.optimize( opt, r0 )

	r_scaled = transform( rfin, scal )
	x_r = get_x_scaled( iter_data ) .+ r_scaled
	mop_res_restoration = eval_mop_at_unscaled_site(mop, x_r )
	fx_r, c_e_r, c_i_r, = eval_result_to_all_vectors( mop_res_restoration, mop )
	l_e_r, l_i_r = eval_linear_constraints_at_unscaled_site( x_r, mop )

	x_indices_r = put_eval_result_into_db!( data_base, mop_res_restoration, x_r )

	return r_scaled, minθ, x_r, fx_r, c_e_r, c_i_r, l_e_r, l_i_r, x_indices_r
end

function iterate!( iter_data :: AbstractIterData, data_base :: AbstractSuperDB, 
	mop :: AbstractMOP, sc :: SurrogateContainer, algo_config :: AbstractConfig, 
	filter :: AbstractFilter = DummyFilter(), _scal :: AbstractVarScaler = nothing)

	# read iteration data
	x = get_x( iter_data )
	fx = get_fx( iter_data )
	Δ = get_delta( iter_data )

	# check (some) stopping conditions 
	# (rest is done at end of this function, when trial point is known)
	if get_num_iterations(iter_data) >= max_iter(algo_config)
        @logmsg loglevel1 "Stopping. Maximum number of iterations reached."
        return MAX_ITER, _scal, iter_data
    end

    if !_budget_okay(mop, algo_config)
        @logmsg loglevel1 "Stopping. Computational budget is exhausted."
        return BUDGET_EXHAUSTED, _scal, iter_data
    end

    if Δ_abs_test( Δ, algo_config )
        return TOLERANCE, _scal, iter_data
    end

	# read algorithm parameters from config 
	ν_success = _nu_success( algo_config )
	ν_accept = _nu_accept( algo_config )
	μ = _mu( algo_config )
	β = _beta( algo_config )
	ε_c = _eps_crit( algo_config )
	γ_c = _gamma_crit( algo_config )

	SAVEABLE_TYPE = get_saveable_type( data_base )

    # set iteration counter
    if it_stat(iter_data) != MODELIMPROVING || count_nonlinear_iterations( algo_config )
        inc_num_iterations!(iter_data)
        set_num_model_improvements!(iter_data, 0)
    else 
        inc_num_model_improvements!(iter_data)
    end
    
    @logmsg loglevel1 """\n
        |--------------------------------------------
        |Iteration $(get_num_iterations(iter_data)).$(get_num_model_improvements(iter_data))
        |--------------------------------------------
        |  Current trust region radius is $(Δ).
        |  Current number of function evals is $(num_evals(mop)).
        |  Iterate is $(_prettify(x))
        |  Values are $(_prettify(fx))
        |--------------------------------------------
    """
	
	# TODO adapt variable scaler
	if get_num_iterations(iter_data) <= 1 
		if isnothing(_scal)
			# should never happen, for backwards compatibility only
			scal = NoVarScaling( full_bounds(mop)... )
		else
			scal = _scal 
		end
	else
		scal = new_var_scaler( get_x_scaled(iter_data), _scal, mop, sc, algo_config ) 
	end
		
	if _scal != scal
		@logmsg loglevel2 "Applying new scaling to database."
		if !isnothing(_scal)
			untransform!( data_base, _scal )
			transform!( data_base, scal )
		else
			db_scaler = combined_untransform_transform_scaler( _scal, scal )
			transform!( data_base, db_scaler )
		end
		set_x_scaled!( iter_data, transform( x, scal ) )
	end

    # update surrogate models
    if get_num_iterations(iter_data) > 1
        if it_stat(iter_data) == MODELIMPROVING 
            improve_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = false );
        else
            update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = false );
        end
    end

	l_e = get_eq_const( iter_data )
	l_i = get_ineq_const( iter_data )
	c_e = get_nl_eq_const( iter_data )
	c_i = get_nl_ineq_const( iter_data )

	θ_k = compute_constraint_val( filter, l_e, l_i, c_e, c_i )
	
	if !( constraint_violation_is_zero(θ_k) )

		LAST_ITER_WAS_RESTORATION = it_stat(iter_data) == RESTORATION
		if LAST_ITER_WAS_RESTORATION
			n, _Δ = compute_normal_step( mop, scal, iter_data, data_base, sc, algo_config; variable_radius = true )
		else
			n, _Δ = compute_normal_step( mop, scal, iter_data, data_base, sc, algo_config; variable_radius = false )
			if _Δ > Δ
				# we increase the trust region radius in order to have 
				# a compatible normal step
				# the surrogates will no longer be considered fully linear
				Δ = _Δ
				set_delta!( iter_data, _Δ )
				set_fully_linear!( sc, false )
			end
		end
		
		# for debugging, uncomment to see what happens with an invalid step `n`
		# n = fill( NaN32, length(x) )

		_isnan_n = any( isnan.(n) )

		EXIT_FOR_NEXT_ITERATION = false	# this is used to avoid nonlinear restoration for linearly constrained problems
		PERFORM_RESTORATION = false 
		EXIT_WITH_INFEASIBLE = false
		
		r_guess = zeros_like(x)	 # initial guess for restoration step

		if _isnan_n 
			if LAST_ITER_WAS_RESTORATION
				EXIT_WITH_INFEASIBLE = true 
			else
				PERFORM_RESTORATION = true
			end
		else
			# `n` is at least a valid vector
			if !is_compatible( n, Δ, algo_config )
				if LAST_ITER_WAS_RESTORATION
					EXIT_WITH_INFEASIBLE = true
				elseif num_nl_constraints( mop ) == 0
					# if there are only linear constraints, we use 
					# `n` as the restoration step `r` and keep the radius
					EXIT_FOR_NEXT_ITERATION = true
				else 
					PERFORM_RESTORATION = true 
					r_guess = n
				end
			end
		end
		
		if PERFORM_RESTORATION
			@logmsg loglevel2 "Performing Restoration for feasibility."
			add_entry!( filter, x, (θ_k, fx) )

			# all decision space stuff is scaled:
			r, θ_r, x_r, fx_r, c_e_r, c_i_r, l_e_r, l_i_r, x_indices_r = restoration(
				iter_data, data_base, mop, algo_config, filter, scal; 
				r_guess_scaled = r_guess
			)

			if is_acceptable( (θ_r, fx_r), filter )
				@logmsg loglevel2 "Found an acceptable restoration step."
				iter_data_r = copy_iter_data( iter_data, untransform(x_r, scal), 
						fx_r, x_r, l_e_r, l_i_r, c_e_r, c_i_r, Δ, x_indices_r )

				
				# For debugging, uncomment the below to lines to see what 
				# happens if no good restoration step is found.
				# iter_data_r = iter_data 
				# θ_r = θ_k

				if !constraint_violation_is_zero(θ_r)
					# If θ_r ≈ 0, the next iteration does not need a normal step
					# n = 0 is automatically compatible, we can simply jump to the next 
					# iteration.
					# 
					# For θ_r > 0, we normally should check, if a 
					# compatible normal step can be found.
					# We leave this to the next iteration but set the iteration 
					# status so that we break if no normal step is found then:.
					it_stat!( iter_data_r, RESTORATION )
				end 
				return CONTINUE, scal, iter_data_r 
			end
			
			EXIT_WITH_INFEASIBLE = true
		end

		if EXIT_WITH_INFEASIBLE 
			@logmsg loglevel1 "Exiting because we could not find a suitable feasible iterate."
			return INFEASIBLE, scal, iter_data 
		end

		# add normal step and untransform 
		x_n_scaled = get_x_scaled( iter_data ) .+ n
		x_n_unscaled = untransform( x_n_scaled, scal )

		@logmsg loglevel2 "x_n = $(_prettify(x_n_unscaled))"
		
		# update all values and constraint_violations
		mop_res_normal = eval_mop_at_unscaled_site(mop, x_n_unscaled )
		fx_n, c_e_n, c_i_n = eval_result_to_all_vectors( mop_res_normal, mop )
		l_e_n, l_i_n = eval_linear_constraints_at_unscaled_site( x_n_unscaled, mop )
		
		x_indices_n = put_eval_result_into_db!( data_base, mop_res_normal, x_n_scaled )

		iter_data_n = copy_iter_data( iter_data, x_n_unscaled, 
				fx_n, x_n_scaled, l_e_n, l_i_n, c_e_n, c_i_n, Δ, x_indices_n )

		if EXIT_FOR_NEXT_ITERATION
			# we treat `n` as an restoration step (only for linearly constrained problems)
			it_stat!(iter_data_n, RESTORATION) 
			return CONTINUE, scal, iter_data_n
		end
	
		θ_n = compute_constraint_val( filter, l_e_n, l_i_n, c_e_n, c_i_n )
	else 
		# in case, we did not need a normal step (n == 0)
		# keep the values as they are 
		θ_n = θ_k
		
		# these values are needed for tangent step calculation:
		x_n_scaled = get_x_scaled(iter_data)

		# this object is used within criticality loop to improve surrogates
		# around the right point
		iter_data_n = iter_data
	end

    # calculate descent step and criticality value of x+n
    ω, ω_data = get_criticality(mop, scal, iter_data_n, data_base, sc, algo_config; x_n_scaled )
    @logmsg loglevel1 "Criticality is ω = $(ω)."
    
    # stop at the end of the this loop run?
	_θ_n_zero = constraint_violation_is_zero(θ_n)
    if _θ_n_zero && ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
        return CRITICAL, scal, iter_data_n
    end

    # Criticallity test
	_fully_linear_sc = fully_linear(sc)
	
	TR_CENTER_CHANGED = false

	if _θ_n_zero &&  ω <= ε_c && (!_fully_linear_sc|| all(Δ .> μ * ω))
		@logmsg loglevel1 "Entered Criticallity Test."
		DO_CRIT_LOOPS = true
        if !_fully_linear_sc
            @logmsg loglevel1 "Ensuring that all models are fully linear."
            update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = true );
            
            ω, ω_data = get_criticality(mop, scal, iter_data_n, data_base, sc, algo_config; x_n_scaled)
            if !fully_linear(sc)
				DO_CRIT_LOOPS = false
				@logmsg loglevel2 "Could not make all models fully linear. Trying one last descent step."
			end
		end
		
		num_critical_loops = 0
		if DO_CRIT_LOOPS			
			
			while all(Δ .> μ * ω )
				@logmsg loglevel2 "Criticality loop $(num_critical_loops + 1)." 
				
				if num_critical_loops >= max_critical_loops(algo_config)
					@logmsg loglevel1 "Maximum number ($(max_critical_loops(algo_config))) of critical loops reached. Exiting..."
					return CRITICAL
				end
				if !_budget_okay(mop, algo_config)
					@logmsg loglevel1 "Computational budget exhausted. Exiting…"
					return BUDGET_EXHAUSTED, scal, iter_data_n
				end
				
				# shrink radius
				Δ = γ_c .* Δ
				set_delta!(iter_data_n, Δ)
				
				# make models fully linear on smaller trust region
				update_surrogates!( sc, mop, scal, iter_data_n, data_base, algo_config; ensure_fully_linear = true )

				# (re)calculate criticality
				ω, ω_data = get_criticality( mop, scal, iter_data_n, data_base, sc, algo_config; x_n_scaled )
				num_critical_loops += 1

				if Δ_abs_test( Δ, algo_config ) || 
					ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
					return CRITICAL, scal, iter_data_n
				end
				if !fully_linear(sc)
					@logmsg loglevel2 "Could not make all models fully linear. Trying one last descent step."
					break
				end
			end

			TR_CENTER_CHANGED = true
		end
		# if we have not returned by now, the iteration continues 
		if num_critical_loops > 0
			@logmsg loglevel1 "Exiting after $(num_critical_loops) loops with ω = $(ω) and Δ = $(get_delta(iter_data))."
		end
	end
	
	if TR_CENTER_CHANGED 
		iter_data_fin = iter_data_n 
	else
		iter_data_fin = iter_data
	end

	ω, x_trial_scaled, mx_trial, steplength = compute_descent_step(
		mop, scal, iter_data_fin, data_base, sc, algo_config, ω, ω_data; x_n_scaled )

	x_trial_unscaled = untransform( x_trial_scaled, scal )

	model_result = eval_container_at_scaled_site(sc, scal, x_trial_scaled)
	mop_result = eval_mop_at_unscaled_site(mop, x_trial_unscaled)

	mx, mc_e, mc_i = eval_vec_container_at_scaled_site(sc, scal, x_n_scaled)
	mx_trial, mc_e_trial, mc_i_trial = eval_result_to_all_vectors(model_result, sc)
	fx_trial, c_e_trial, c_i_trial = eval_result_to_all_vectors(mop_result, sc)

	l_e_trial, l_i_trial = eval_linear_constraints_at_scaled_site( x_trial_scaled, mop, scal )

	_ρ =  if strict_acceptance_test( algo_config )
			minimum( (fx .- fx_trial) ./ (mx .- mx_trial) )
		else
			(maximum(fx) - maximum( fx_trial ))/(maximum(mx) - maximum(mx_trial))
		end
	ρ = isnan(_ρ) ? -Inf : _ρ
	
	@logmsg loglevel2 """\n
	Attempting descent of length $steplength with trial point 
	x₊ = $(_prettify( x_trial_unscaled, 10)) ⇒
	| f(x)  | $(_prettify(fx))
	| f(x₊) | $(_prettify(fx_trial))
	| m(x)  | $(_prettify(mx))
	| m(x₊) | $(_prettify(mx_trial))
	The error betwenn f(x) and m(x) is $(sum(abs.(fx .- mx))).
	$(strict_acceptance_test(algo_config) ? "All" : "One") of the components must decrease.
	Thus, ρ is $ρ.
	"""

	# put new values in data base 
	new_x_indices = put_eval_result_into_db!( data_base, mop_result, x_trial_scaled )

	Δ_old = Δ	# save for printing
	if ρ >= ν_success
		if Δ < β * ω 
			Δ = grow_radius( algo_config, Δ, steplength )
		end
		it_stat!( iter_data_fin, SUCCESSFULL )
	else
		if fully_linear(sc)
			if ρ < ν_accept
				Δ = shrink_radius_much( algo_config, Δ, steplength )
				it_stat!( iter_data_fin, INACCEPTABLE )
			else
				Δ = shrink_radius(algo_config, Δ, steplength)
				it_stat!( iter_data_fin, ACCEPTABLE )
			end
		else
			it_stat!( iter_data_fin, MODELIMPROVING )
		end
	end

	# before updating `iter_data`, retrieve a saveable
	# (order is important here to have current index and next index)
	stamp_content = get_saveable( SAVEABLE_TYPE, iter_data_fin; 
		ρ, ω, sc,
		stepsize = steplength )
	stamp!( data_base, stamp_content )

	# update iter data 
	set_delta!(iter_data_fin, Δ)
	if it_stat(iter_data_fin) in [SUCCESSFULL, ACCEPTABLE]
		set_x!(iter_data_fin, x_trial_unscaled)
		set_x_scaled!(iter_data_fin, x_trial_scaled)
		set_fx!(iter_data_fin, fx_trial)
		set_eq_const!(iter_data_fin, l_e_trial)
		set_ineq_const!(iter_data_fin, l_i_trial)
		set_nl_eq_const!(iter_data_fin, c_e_trial)
		set_nl_ineq_const!(iter_data_fin, c_i_trial)
		for func_indices in keys( new_x_indices )
			set_x_index!( iter_data_fin, func_indices, new_x_indices[func_indices] )
		end
	end

	_it_stat = it_stat( iter_data_fin )
	@logmsg loglevel1 """\n
		The trial point was $( 
			(_it_stat in [SUCCESSFULL, ACCEPTABLE] ) ? "" : "not ")accepted.
		The iteration is $(_it_stat).
		Moreover, the radius was updated as below:
		old radius : $(Δ_old)
		new radius : $(get_delta(iter_data_fin)) ($(round(get_delta(iter_data_fin)/Δ_old * 100; digits=1)) %)
	"""

	if ( x_tol_rel_test( x, x_trial_unscaled, algo_config  ) || 
			x_tol_abs_test( x, x_trial_unscaled, algo_config ) ||
			f_tol_rel_test( fx, fx_trial, algo_config  ) ||
			f_tol_abs_test( fx, fx_trial, algo_config ) 
		)
		return TOLERANCE, scal, iter_data_fin
	end

	return CONTINUE, scal, iter_data_fin
end

############################################
function optimize( mop :: AbstractMOP, x0 :: Vec;
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractSuperDB, Nothing} = nothing, kwargs... )
    
	mop, iter_data, super_data_base, sc, ac, filter, scal = initialize_data(
		mop, x0; algo_config, populated_db, kwargs...)
    @logmsg loglevel1 _stop_info_str( ac, mop )

    @logmsg loglevel1 "Entering main optimization loop."
	ret_code = CONTINUE
    while ret_code == CONTINUE
        ret_code, scal, iter_data = iterate!(iter_data, super_data_base, mop, sc, ac, filter, scal)
    end# while

	ret_x, ret_fx = get_return_values( iter_data )
   
    @logmsg loglevel1 _fin_info_str(iter_data, mop, ret_code)

	if untransform_final_database( ac )
		untransform!( super_data_base, scal )
	end

    return ret_x, ret_fx, ret_code, super_data_base, iter_data, filter
end# function optimize

