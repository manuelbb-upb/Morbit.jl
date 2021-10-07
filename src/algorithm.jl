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
	x = ensure_precision(scale( x0, smop ))
	
	# initalize first objective vector, constraint vectors etc.
	@logmsg loglevel2 "Evaluating at start site."
	eval_res = eval_mop_at_scaled_site( smop, x )
	
	# build SuperDB `sdb`
	groupings = do_groupings( smop, ac )
	if isnothing( populated_db ) 
		sub_dbs, x_index_mapping = build_super_db( groupings, x, eval_res )
	else
		sdb = populated_db 
		transform!( sdb, smop )
		x_index_mapping = Dict{FunctionIndexTuple, Int}()
		for func_indices in all_sub_db_indices( sdb )
			db = get_sub_db( sdb, func_indices )
			vals = eval_result_to_vector( eval_res, func_indices )
			x_index = ensure_contains_values!( db, x, vals )
			x_index_mapping[func_indices] = x_index
		end
		sub_dbs = all_sub_dbs(sdb)
	end
	
	fx, c_e, c_i = eval_result_to_all_vectors( eval_res, smop )
	Δ_0 = eltype(x).(get_delta_0(ac))
	id = init_iter_data( IterData, x, fx, c_e, c_i, Δ_0, x_index_mapping )
	
	filterT = if num_eq_constraints( mop ) + num_ineq_constraints( mop ) > 0
		filter_type( ac )
	else
		DummyFilter
	end
	filter = init_empty_filter( filterT, x, fx, c_e, c_i; shift = filter_shift(ac) )

    sdb = SuperDB(; sub_dbs, iter_data = typeof(get_saveable(IterSaveable,id))[] )

	# initialize surrogate models
	sc = init_surrogates(SurrogateContainer, smop, id, ac, groupings, sdb )

	return (smop, id, sdb, sc, ac, filter)
end


function iterate!( iter_data :: AbstractIterData, data_base :: AbstractSuperDB, mop :: AbstractMOP, 
	sc :: SurrogateContainer, algo_config :: AbstractConfig, filter :: AbstractFilter = DummyFilter() ) :: STOP_CODE

	# read iteration data
	x = get_x( iter_data )
	fx = get_fx( iter_data )
	Δ = get_delta( iter_data )

	# check (some) stopping conditions 
	# (rest is done at end of this function, when trial point is known)
	if get_num_iterations(iter_data) >= max_iter(algo_config)
        @logmsg loglevel1 "Stopping. Maximum number of iterations reached."
        return MAX_ITER
    end

    if !_budget_okay(mop, algo_config)
        @logmsg loglevel1 "Stopping. Computational budget is exhausted."
        return BUDGET_EXHAUSTED
    end

    if Δ_abs_test( Δ, algo_config )
        return TOLERANCE
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
        set_num_model_improvements!(iter_data, 0);
    else 
        inc_num_model_improvements!(iter_data)
    end
    
    @logmsg loglevel1 """\n
        |--------------------------------------------
        |Iteration $(get_num_iterations(iter_data)).$(get_num_model_improvements(iter_data))
        |--------------------------------------------
        |  Current trust region radius is $(Δ).
        |  Current number of function evals is $(num_evals(mop)).
        |  Iterate is $(_prettify(unscale(x, mop)))
        |  Values are $(_prettify(fx))
        |--------------------------------------------
    """

    # update surrogate models
    if get_num_iterations(iter_data) > 1
        if it_stat(iter_data) == MODELIMPROVING 
            improve_surrogates!( sc, mop, iter_data, data_base, algo_config; ensure_fully_linear = false );
        else
            update_surrogates!( sc, mop, iter_data, data_base, algo_config; ensure_fully_linear = false );
        end
    end

	# TODO: if problem is constrained by functions:
	# do "normal step here"

    # calculate descent step and criticality value
    ω, ω_data = get_criticality(mop, iter_data, data_base, sc, algo_config)
    @logmsg loglevel1 "Criticality is ω = $(ω)."
    
    # stop at the end of the this loop run?
    if ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
        return CRITICAL
    end

    # Criticallity test
	_fully_linear_sc = fully_linear(sc)
	if ω <= ε_c && (!_fully_linear_sc|| all(Δ .> μ * ω))
		@logmsg loglevel1 "Entered Criticallity Test."
		DO_CRIT_LOOPS = true
        if !_fully_linear_sc
            @logmsg loglevel1 "Ensuring that all models are fully linear."
            update_surrogates!( sc, mop, iter_data, data_base, algo_config; ensure_fully_linear = true );
            
            ω, ω_data = get_criticality(mop, iter_data, data_base, sc, algo_config )
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
					return BUDGET_EXHAUSTED
				end
				# shrink radius
				Δ = γ_c .* Δ
				set_delta!(iter_data, Δ)
				
				# make models fully linear on smaller trust region
				update_surrogates!( sc, mop, iter_data, data_base, algo_config; ensure_fully_linear = true )

				# (re)calculate criticality
				ω, ω_data = get_criticality( mop, iter_data, data_base, sc, algo_config )
				num_critical_loops += 1

				if Δ_abs_test( Δ, algo_config ) || 
					ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
					return CRITICAL
				end
				if !fully_linear(sc)
					@logmsg loglevel2 "Could not make all models fully linear. Trying one last descent step."
					break
				end
			end
		end
		if num_critical_loops > 0
			@logmsg loglevel1 "Exiting after $(num_critical_loops) loops with ω = $(ω) and Δ = $(get_delta(iter_data))."
		end
	end

    if ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
        return CRITICAL
    end

	n = compute_normal_step( mop, iter_data, data_base, sc, algo_config )
	ω, x₊, mx₊, steplength = compute_descent_step(mop, iter_data, data_base, sc, algo_config)

	model_result = eval_container_at_scaled_site(sc, x₊)
	mop_result = eval_mop_at_scaled_site(mop, x₊)

	mx, mc_e, mc_i = eval_vec_container_at_scaled_site(sc, x)
	mx₊, mc_e₊, mc_i₊ = eval_result_to_all_vectors(model_result, sc)
	fx₊, c_e₊, c_i₊ = eval_result_to_all_vectors(mop_result, sc)

	if strict_acceptance_test( algo_config )
		_ρ = minimum( (fx .- fx₊) ./ (mx .- mx₊) )
	else
		_ρ = (maximum(fx) - maximum( fx₊ ))/(maximum(mx) - maximum(mx₊))
	end
	ρ = isnan(_ρ) ? -Inf : _ρ
	
	@logmsg loglevel2 """\n
	Attempting descent of length $steplength with trial point 
	x₊ = $(_prettify(unscale(x₊, mop), 10)) ⇒
	| f(x)  | $(_prettify(fx))
	| f(x₊) | $(_prettify(fx₊))
	| m(x)  | $(_prettify(mx))
	| m(x₊) | $(_prettify(mx₊))
	The error betwenn f(x) and m(x) is $(sum(abs.(fx .- mx))).
	$(strict_acceptance_test(algo_config) ? "All" : "One") of the components must decrease.
	Thus, ρ is $ρ.
	"""

	# put new values in data base 
	new_x_indices = put_eval_result_into_db!( data_base, mop_result, x₊ )

	Δ_old = Δ	# save for printing
	if ρ >= ν_success
		if Δ < β * ω 
			Δ = grow_radius( algo_config, Δ, steplength )
		end
		it_stat!( iter_data, SUCCESSFULL )
	else
		if fully_linear(sc)
			if ρ < ν_accept
				Δ = shrink_radius_much( algo_config, Δ, steplength )
				it_stat!( iter_data, INACCEPTABLE )
			else
				Δ = shrink_radius(algo_config, Δ, steplength)
				it_stat!( iter_data, ACCEPTABLE )
			end
		else
			it_stat!( iter_data, MODELIMPROVING )
		end
	end

	# before updating `iter_data`, retrieve a saveable
	# (order is important here to have current index and next index)
	stamp_content = get_saveable( SAVEABLE_TYPE, iter_data; 
		ρ, ω, sc,
		stepsize = steplength )
	stamp!( data_base, stamp_content )

	# update iter data 
	set_delta!(iter_data, Δ)
	if it_stat(iter_data) in [SUCCESSFULL, ACCEPTABLE]
		set_x!(iter_data, x₊)
		set_fx!(iter_data, fx₊)
		set_eq_const!(iter_data, c_e₊)
		set_ineq_const!(iter_data, c_i₊)
		for func_indices in keys( new_x_indices )
			set_x_index!( iter_data, func_indices, new_x_indices[func_indices] )
		end
	end

	@logmsg loglevel1 """\n
		The trial point was $( (it_stat(iter_data) == SUCCESSFULL) || (it_stat(iter_data) == ACCEPTABLE ) ? "" : "not ")accepted.
		The iteration is $(it_stat(iter_data)).
		Moreover, the radius was updated as below:
		old radius : $(Δ_old)
		new radius : $(get_delta(iter_data)) ($(round(get_delta(iter_data)/Δ_old * 100; digits=1)) %)
	"""

	if ( x_tol_rel_test( x, x₊, algo_config  ) || x_tol_abs_test( x, x₊, algo_config ) ||
		f_tol_rel_test( fx, fx₊, algo_config  ) || f_tol_abs_test( fx, fx₊, algo_config ) )
		return TOLERANCE
	end

	return CONTINUE
end

############################################
function optimize( mop :: AbstractMOP, x0 :: Vec;
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractSuperDB, Nothing} = nothing, kwargs... )
    
	mop, iter_data, super_data_base, sc, ac, filter = initialize_data(mop, x0; algo_config, populated_db, kwargs...)
    @logmsg loglevel1 _stop_info_str( ac, mop )

    @logmsg loglevel1 "Entering main optimization loop."
	ret_code = CONTINUE
    while ret_code == CONTINUE
        ret_code = iterate!(iter_data, super_data_base, mop, sc, ac, filter)
    end# while

	ret_x, ret_fx = get_return_values( iter_data, mop )
   
    @logmsg loglevel1 _fin_info_str(super_data_base, iter_data, mop, ret_code)

    return ret_x, ret_fx, ret_code, super_data_base, iter_data, filter
end# function optimize

