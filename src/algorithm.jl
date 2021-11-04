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

function do_radius_update(iter_data, _RADIUS_UPDATE, algo_config, steplength)
	Δ = get_delta(iter_data)
	if _RADIUS_UPDATE == LEAVE_UNCHANGED
		return Δ, Δ
	elseif _RADIUS_UPDATE == GROW 
		return grow_radius( algo_config, Δ, steplength ), Δ
	elseif _RADIUS_UPDATE == SHRINK
		return shrink_radius( algo_config, Δ, steplength ), Δ
	elseif _RADIUS_UPDATE == SHRINK_MUCH
		return shrink_radius_much( algo_config, Δ, steplength ), Δ
	end
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
	eval_res = eval_dict_mop_at_unscaled_site( smop, x )

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
			vals = flatten_mop_dict( eval_res, func_indices )
			x_index = ensure_contains_values!( db, x_scaled, vals )
			x_index_mapping[func_indices] = x_index
		end
		sub_dbs = all_sub_dbs(sdb)
	end
	
	l_e, l_i = eval_linear_constraints_at_scaled_site( x_scaled, smop, scal )
	fx, c_e, c_i = eval_result_to_all_vectors( eval_res, smop )
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
	sdb = SuperDB(; sub_dbs, iter_data = [init_stamp_content,]) # get_saveable_type(IterSaveable, id)[] )

	# initialize surrogate models
	sc = init_surrogates(SurrogateContainer, smop, scal, id, ac, groupings, sdb )

	return (smop, id, sdb, sc, ac, filter, scal)
end

function restoration(iter_data :: AbstractIterate, data_base :: AbstractSuperDB,
		mop :: AbstractMOP, algo_config :: AbstractConfig, 
		filter :: AbstractFilter, scal :: AbstractVarScaler;
		r_guess_scaled = nothing
	)
	
	# TODO This is a very expensive task:
	# We are minimizing the constraint violation `θ`
	# by using the true constraint functions and NLopt.
	# Maybe we should instead do this with Morbit itself?

	x = get_x( iter_data )
	Xet = eltype(x)
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
	minθ, _rfin, ret = NLopt.optimize( opt, r0 )
	rfin = Xet.(_rfin)

	r_scaled = transform( rfin, scal )
	x_r = get_x_scaled( iter_data ) .+ r_scaled
	
	mop_res_restoration = eval_dict_mop_at_unscaled_site(mop, x_r )
	fx_r, c_e_r, c_i_r, = eval_result_to_all_vectors( mop_res_restoration, mop )
	l_e_r, l_i_r = eval_linear_constraints_at_unscaled_site( x_r, mop )

	x_indices_r = put_eval_result_into_db!( data_base, mop_res_restoration, x_r )

	return r_scaled, minθ, x_r, fx_r, c_e_r, c_i_r, l_e_r, l_i_r, x_indices_r
end

function iterate!( iter_data :: AbstractIterate, data_base :: AbstractSuperDB, 
		mop :: AbstractMOP, sc :: SurrogateContainer, algo_config :: AbstractConfig, 
		filter :: AbstractFilter = DummyFilter(), _scal :: AbstractVarScaler = nothing;
		iter_counter :: Int = 1, last_it_stat :: ITER_TYPE = ACCEPTABLE, logger = Logging.current_logger(), 
	)
	
	Logging.with_logger( logger ) do 

	x = get_x( iter_data )
	fx = get_fx( iter_data )

	# check (some) stopping conditions 
	# (rest is done at end of this function, when trial point is known)
	if iter_counter >= max_iter(algo_config)
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
	γ_c = _gamma_crit( algo_config )

	SAVEABLE_TYPE = get_saveable_type( data_base )

    @logmsg loglevel1 """\n
        |--------------------------------------------
        |Iteration $(iter_counter)
        |--------------------------------------------
        |  Current trust region radius is $(get_delta( iter_data )).
        |  Current number of function evals is $(num_evals(mop)).
        |  Iterate is $(_prettify(x))
        |  Values are $(_prettify(fx))
        |--------------------------------------------
    """
	
	# TODO adapt variable scaler
	if iter_counter <= 1 
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
    if iter_counter > 1
        if last_it_stat == MODELIMPROVING 
            improve_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = false );
		elseif last_it_stat != CRIT_LOOP_EXIT
			# (if we exited in the last iteration after criticality loops 
			# then we already have good models)
            update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; ensure_fully_linear = false );
        end
    end

	l_e = get_eq_const( iter_data )
	l_i = get_ineq_const( iter_data )
	c_e = get_nl_eq_const( iter_data )
	c_i = get_nl_ineq_const( iter_data )

	θ_k = compute_constraint_val( filter, l_e, l_i, c_e, c_i )
	
	if !( constraint_violation_is_zero(θ_k) )
		@assert last_it_stat != CRIT_LOOP_EXIT "This should be impossible!"

		@show LAST_ITER_WAS_RESTORATION = (last_it_stat == RESTORATION)
		if LAST_ITER_WAS_RESTORATION
			n, _Δ = compute_normal_step( mop, scal, iter_data, data_base, sc, algo_config; variable_radius = true )
		else
			n, _Δ = compute_normal_step( mop, scal, iter_data, data_base, sc, algo_config; variable_radius = false )
			if _Δ > get_delta( iter_data )
				# we increase the trust region radius in order to have 
				# a compatible normal step
				# the surrogates will no longer be considered fully linear
				set_delta!( iter_data, _Δ )
				set_fully_linear!( sc, false )
			end
		end
		
		# for debugging, uncomment to see what happens with an invalid step `n`
		# n = fill( NaN32, length(x) )
		@show n
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
			if !is_compatible( n, get_delta( iter_data ), algo_config )
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
			add_entry!( filter, x, (θ_k, compute_objective_val(filter,fx)) )

			# all decision space stuff is scaled:
			r, θ_r, x_r, fx_r, c_e_r, c_i_r, l_e_r, l_i_r, x_indices_r = restoration(
				iter_data, data_base, mop, algo_config, filter, scal; 
				r_guess_scaled = r_guess
			)

			if is_acceptable( (θ_r, fx_r), filter )
				@logmsg loglevel2 "Found an acceptable restoration step."
				iter_data_r = init_iterate( IterData, untransform(x_r, scal), x_r,
						fx_r, l_e_r, l_i_r, c_e_r, c_i_r, get_delta( iter_data ), x_indices_r )

				
				# For debugging, uncomment the below to lines to see what 
				# happens if no good restoration step is found.
				# iter_data_r = iter_data 
				# θ_r = θ_k

				# If θ_r ≈ 0, the next iteration does not need a normal step
				# n = 0 is automatically compatible, we can simply jump to the next 
				# iteration.
				# 
				# For θ_r > 0, we normally should check, if a 
				# compatible normal step can be found.
				# We leave this to the next iteration but set the iteration 
				# status so that we break if no normal step is found then:
				return CONTINUE, RESTORATION, scal, iter_data_r 
			end
			
			EXIT_WITH_INFEASIBLE = true
		end

		if EXIT_WITH_INFEASIBLE 
			@logmsg loglevel1 "Exiting because we could not find a suitable feasible iterate."
			return INFEASIBLE, EARLY_EXIT, scal, iter_data 
		end

		# add normal step and untransform 
		x_n_scaled = get_x_scaled( iter_data ) .+ n
		x_n_unscaled = untransform( x_n_scaled, scal )

		@logmsg loglevel2 "x_n = $(_prettify(x_n_unscaled))"
		
		# update all values and constraint_violations
		mop_res_normal = eval_dict_mop_at_unscaled_site(mop, x_n_unscaled )
		fx_n, c_e_n, c_i_n = eval_result_to_all_vectors( mop_res_normal, mop )
		l_e_n, l_i_n = eval_linear_constraints_at_unscaled_site( x_n_unscaled, mop )
		
		x_indices_n = put_eval_result_into_db!( data_base, mop_res_normal, x_n_scaled )

		iter_data_n = init_iterate( IterData, x_n_unscaled, x_n_scaled,
				fx_n, l_e_n, l_i_n, c_e_n, c_i_n, get_delta( iter_data ), x_indices_n )

		if EXIT_FOR_NEXT_ITERATION
			# we treat `n` as an restoration step (only for linearly constrained problems)
			stamp_content = get_saveable( SAVEABLE_TYPE, iter_data_n; 
				rho = -Inf, omega = -Inf, steplength = -Inf, iter_counter, it_stat = RESTORATION
			)
			stamp!( data_base, stamp_content )
			return CONTINUE, RESTORATION, scal, iter_data_n
		end
	
		θ_n = compute_constraint_val( filter, l_e_n, l_i_n, c_e_n, c_i_n )
	else 
		# in case, we did not need a normal step (n == 0)
		# keep the values as they are 
		θ_n = θ_k
		
		# these values are needed for tangent step calculation:
		iter_data_n = iter_data
	end

    # calculate descent step and criticality value of x+n
    ω, ω_data = get_criticality(mop, scal, iter_data, iter_data_n, data_base, sc, algo_config )
	ω_data
    @logmsg loglevel1 "Criticality is ω = $(ω)."
    
    # stop at the end of the this loop run?
	_θ_n_zero = constraint_violation_is_zero(θ_n)
    if _θ_n_zero && ( ω_Δ_rel_test(ω, get_delta( iter_data ), algo_config) || 
			ω_abs_test( ω, algo_config )
		)
        return CRITICAL, EARLY_EXIT, scal, iter_data_n
    end

   	#==============================
	Criticallity test
	==============================#
	_fully_linear_sc = fully_linear(sc)
	if _θ_n_zero &&  ω <= ε_c && (!_fully_linear_sc|| all(get_delta( iter_data ) .> μ * ω))
		@logmsg loglevel1 "Entered Criticallity Test."
		
		DO_CRIT_LOOPS = true
        if !_fully_linear_sc
            @logmsg loglevel1 "Ensuring that all models are fully linear."

			# update with respect to unshifted center
            update_surrogates!( sc, mop, scal, iter_data, data_base, algo_config; 
				ensure_fully_linear = true )
            
			# are we still critical?
            ω, ω_data = get_criticality(mop, scal, iter_data, iter_data_n, data_base, sc, algo_config)
            
			if !fully_linear(sc)
				DO_CRIT_LOOPS = false
				@logmsg loglevel2 "Could not make all models fully linear. Trying one last descent step."
			else 
				DO_CRIT_LOOPS = all( get_delta( iter_data ) .> μ * ω)
			end

		end
		
		num_critical_loops = 0
		RETURN_CRITICAL = false
		if DO_CRIT_LOOPS			

			# if we are really near a critical point we compute the criticality from x+n hereafter
			# that's wey in the while loop I use `iter_data_n` instead of `iter_data`
			Δ = get_delta( iter_data_n )
			Δ_0 = Δ

			while all(Δ .> μ * ω )
				@logmsg loglevel2 "Criticality loop $(num_critical_loops + 1)." 
				
				if num_critical_loops >= max_critical_loops(algo_config)
					@logmsg loglevel1 "Maximum number ($(max_critical_loops(algo_config))) of critical loops reached. Exiting..."
					return CRITICAL, EARLY_EXIT, scal, iter_data_n
				end
				if !_budget_okay(mop, algo_config)
					@logmsg loglevel1 "Computational budget exhausted. Exiting…"
					return BUDGET_EXHAUSTED, EARLY_EXIT, scal, iter_data_n
				end
				
				# shrink radius
				Δ = γ_c .* Δ				
				
				# make models fully linear on smaller trust region
				update_surrogates!( sc, mop, scal, iter_data_n, data_base, algo_config; ensure_fully_linear = true )

				# (re)calculate criticality
				# it is fully intentional that the arguments read `…, iter_data_n, iter_data_n,…`
				ω, ω_data = get_criticality( mop, scal, iter_data_n, iter_data_n, data_base, sc, algo_config )
				num_critical_loops += 1

				if Δ_abs_test( Δ, algo_config ) || 
					ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
					RETURN_CRITICAL = true
					break
				end
				if !fully_linear(sc)
					@logmsg loglevel2 "Could not make all models fully linear."
					RETURN_CRITICAL = true 
					break
				end
			end

			# we have fully linear surrogates here AND Δ <= μ * ω so we would not 
			# enter the criticality test again if we were to restart the iteration
			
			@logmsg loglevel1 """
				Exiting after $(num_critical_loops) loops with 
				ω = $(ω) and Δ = $(Δ)).
			"""

			if RETURN_CRITICAL
				return CRITICAL, EARLY_EXIT, scal, iter_data_n
			end

			# if we are here, we deem the point x+n not critical (enough)
			# let's make x+n the next iterate 
			# iter_data_n already has the right x, fx, … values
			# (I guess, for convergence theory we should classify this by 
			# its own ITER_TYPE)
			set_delta!(iter_data_n, min( Δ_0, max( β * ω, Δ) ) )
			stamp_content = get_saveable( SAVEABLE_TYPE, iter_data_n; 
				rho = -Inf, omega = ω, steplength = -Inf, iter_counter, it_stat = CRIT_LOOP_EXIT
			)
			stamp!( data_base, stamp_content )
			return CONTINUE, CRIT_LOOP_EXIT, scal, iter_data_n
			# TODO somehow pass ω too so that it is not recomputed in the next iteration
			
		end
	end

	# Calculation of trial point and evaluation of Objective and Surrogates:
	ω, x_trial_scaled, mx_trial, _ = compute_descent_step(
		mop, scal, iter_data, iter_data_n, data_base, sc, algo_config, ω, ω_data 
	)

	x_scaled = get_x_scaled( iter_data )
	x_trial_unscaled = untransform( x_trial_scaled, scal )

	mop_result = eval_dict_mop_at_unscaled_site(mop, x_trial_unscaled)
	mx = eval_container_objectives_at_scaled_site(sc, scal, x_scaled )
	mx_trial = eval_container_objectives_at_scaled_site(sc, scal, x_trial_scaled)

	fx_trial, c_e_trial, c_i_trial = eval_result_to_all_vectors(mop_result, mop)

	l_e_trial, l_i_trial = eval_linear_constraints_at_scaled_site( x_trial_scaled, mop, scal )

	# put new values in data base 
	new_x_indices = put_eval_result_into_db!( data_base, mop_result, x_trial_scaled )

	#steplength = norm(untransform( x_scaled .- x_trial_scaled, scal ), Inf ) 
	steplength = norm( x_scaled .- x_trial_scaled, Inf ) 	# TODO transformed steplength????
	@logmsg loglevel2 """\n
	Attempting descent of length $steplength with trial point 
	x₊ = $(_prettify( x_trial_unscaled, 10)) ⇒
	| f(x)  | $(_prettify(fx))
	| f(x₊) | $(_prettify(fx_trial))
	| m(x)  | $(_prettify(mx))
	| m(x₊) | $(_prettify(mx_trial))
	The error betwenn f(x) and m(x) is $(sum(abs.(fx .- mx))).
	$(strict_acceptance_test(algo_config) ? "All" : "One") of the components must decrease.
	"""

	#========================
	Trust region updates
	========================#	
	θ_trial = compute_constraint_val( filter, l_e_trial, l_i_trial, c_e_trial, c_i_trial )
	IS_ACCEPTABLE_FOR_FILTER = is_acceptable( (θ_trial, compute_objective_val(filter,fx_trial)), 
		filter, (θ_k, compute_objective_val(filter,fx)) )

	# we only need to compute ρ and ω(x) - ω(x₊) ≥ κ_ψ θ^ψ IF 
	# the trial point is acceptable for F ∪ {x}
	if IS_ACCEPTABLE_FOR_FILTER
		if strict_acceptance_test( algo_config )
			model_denom = (mx .- mx_trial)
			_ρ = minimum( (fx .- fx_trial) ./ model_denom )
		else
			model_denom = (maximum(mx) - maximum(mx_trial))
			_ρ = (maximum(fx) - maximum( fx_trial ))/ model_denom
		end
		MODEL_DENOM_TEST = all( 
			model_denom .>= filter_kappa_psi(algo_config) * θ_k^filter_psi(algo_config) 
		)
	else
		_ρ = NaN16
		MODEL_DENOM_TEST = false
	end
	ρ = isnan(_ρ) ? -Inf : _ρ
	
	# we set the following variables here 
	# `IT_STAT` <--> iteration classification
	# `_RADIUS_UPDATE` <--> how to modify the trust region radius
	# `ACCEPT_TRIAL_POINT`
	# (theoretically it would suffice to set `IT_STAT`, the rest is redundant)
	# (but written out it is more easy to follow what happens)
	_RADIUS_UPDATE = LEAVE_UNCHANGED
	ACCEPT_TRIAL_POINT = true
	IT_STAT = ACCEPTABLE
		
	if IS_ACCEPTABLE_FOR_FILTER
		if MODEL_DENOM_TEST
			# From here we can follow the logic for the UNCONSTRAINED TRM algorithm
			if ρ >= ν_success
				ACCEPT_TRIAL_POINT = true 
				IT_STAT = SUCCESSFULL
				if get_delta(iter_data) < β * ω	
					_RADIUS_UPDATE = GROW
				end
			else # ρ < ν_success
				if fully_linear( sc )
					if ρ >= ν_accept
						ACCEPT_TRIAL_POINT = true
						IT_STAT = ACCEPTABLE
						_RADIUS_UPDATE = SHRINK
					else
						ACCEPT_TRIAL_POINT = false 
						IT_STAT = INACCEPTABLE
						_RADIUS_UPDATE = SHRINK_MUCH
					end
				else
					# ρ < ν_success AND models not fully linear
					ACCEPT_TRIAL_POINT = false
					IT_STAT = MODELIMPROVING
					_RADIUS_UPDATE = LEAVE_UNCHANGED
				end
			end
		else
			# if the model decrease is small compared to constraint violation 
			ACCEPT_TRIAL_POINT = true
			IT_STAT = FILTER_ADD
			_RADIUS_UPDATE = LEAVE_UNCHANGED
		end
	else
		# reject if trial point is not acceptable to filter
		ACCEPT_TRIAL_POINT = false
		IT_STAT = FILTER_FAIL
		_RADIUS_UPDATE = SHRINK_MUCH
	end

	#=
	Perform Updates 
	=#

	if IT_STAT == FILTER_ADD
		vals_trial = compute_values( filter, fx_trial, l_e_trial, l_i_trial, c_e_trial, c_i_trial )
		add_entry!( filter, x_trial_unscaled, vals_trial )
	end

	# before updating `iter_data`, retrieve a saveable
	# (order is important here to have current index and next index)
	stamp_content = get_saveable( SAVEABLE_TYPE, iter_data; 
		rho = Float64(ρ), omega = Float64(ω), steplength = Float64(steplength), iter_counter, it_stat = IT_STAT 
	)
	stamp!( data_base, stamp_content )

	Δ, Δ_old = do_radius_update(iter_data, _RADIUS_UPDATE, algo_config, steplength)

	next_iterate = if ACCEPT_TRIAL_POINT
		init_iterate( IterData, 
			x_trial_unscaled, x_trial_scaled, fx_trial, l_e_trial, l_i_trial,
			c_e_trial, c_i_trial, Δ, new_x_indices )
	else
		set_delta!( iter_data, Δ)	
		iter_data	
	end

	@logmsg loglevel1 """\n
		ρ = $(ρ)
		θ_+ = $(θ_trial)
		The trial point was $(ACCEPT_TRIAL_POINT ? "" : "not ")accepted.
		The iteration is $(IT_STAT).
		Moreover, the radius was updated as below:
		old radius : $(Δ_old)
		new radius : $(Δ)) ($(round(Δ/Δ_old * 100; digits=1)) %)
	"""

	if ( x_tol_rel_test( x, x_trial_unscaled, algo_config  ) || 
			x_tol_abs_test( x, x_trial_unscaled, algo_config ) ||
			f_tol_rel_test( fx, fx_trial, algo_config  ) ||
			f_tol_abs_test( fx, fx_trial, algo_config ) 
		)
		return TOLERANCE, IT_STAT, scal, next_iterate
	end

	return CONTINUE, IT_STAT, scal, next_iterate

	end# with_logger
end

############################################
function optimize( mop :: AbstractMOP, x0 :: Vec;
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractSuperDB, Nothing} = nothing,
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

