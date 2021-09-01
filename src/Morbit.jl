
module Morbit

using DocStringExtensions

using Printf: @sprintf

# steepest descent and directed search
using LinearAlgebra: norm, pinv
import JuMP;
import OSQP;

# LagrangeModels and PS
import NLopt;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;

import ThreadSafeDicts: ThreadSafeDict;
using Memoize;

import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

import Logging: LogLevel, @logmsg
import Logging

export MixedMOP, RbfConfig, ExactConfig, TaylorConfig, TaylorApproximateConfig, TaylorCallbackConfig
export add_objective!, add_vector_objective!
export optimize
export SteepestDescentConfig, PascolettiSerafiniConfig

include("custom_logging.jl")

include("shorthands.jl");
include("Interfaces.jl");
include("diff_wrappers.jl");

# implementations (order should not matter)
include("VectorObjectiveFunction.jl");
include("MixedMOP.jl")
include("StaticMOP.jl")

include("ResultImplementation.jl")
include("DataBaseImplementation.jl")
include("IterDataImplementation.jl")
include("SurrogatesImplementation.jl");

include("ConfigImplementations.jl")

# utilities
include("adding_objectives.jl");
include("descent.jl")
#include("saving.jl")
include("utilities.jl")

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

# we expect mop :: MixedMOP, but should work for static MOP if everything 
# is set up properly 
function initialize_data( mop :: AbstractMOP, x0 :: Vec, fx0 :: Vec = MIN_PRECISION[]; 
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractDB, Nothing} = nothing )
    
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
	if num_vars(mop) == 0
		# will error if mop isa StaticMOP
		MOI.add_variables(mop, length(x0))
	end

	@assert length(x0) == num_vars( mop ) "Number of variables in `mop` does not match length of `x0`."

	# make problem static 
	smop = StaticMOP(mop)

	# scale bound vars to unit hypercube
	x_scaled = scale( x0, smop )
	# ensure at least single-precision
	XTe = Base.promote_eltype( x_scaled, MIN_PRECISION )
	x = XTe.(x_scaled)
	XT = typeof(x)

	# initalize first objective vector 
	if isempty( fx0 )
		# if no starting function value was provided, eval objectives
		fx_sorted = eval_all_objectives( smop, x )
	else
		fx_sorted = apply_internal_sorting( fx0, smop )
	end 

	YTe = Base.promote_eltype( fx_sorted, MIN_PRECISION )
	fx = YTe.(fx_sorted)
	YT = typeof( fx )

	if isnothing( algo_config )
		ac = DefaultConfig()
	else
		ac = algo_config
	end

	# initialize iter data obj
	Δ_0 = let Δ = get_delta_0( ac );
		T = promote_type( typeof(Δ), XTe )
		T.(Δ)
	end
	DT = typeof(Δ_0)

	id = init_iter_data( IterData, x, fx, Δ_0 )

	# initialize database
	if !isnothing(populated_db)
		# has a database been provided? if yes, prepare (scale vars, sort values)
		data_base = populated_db
		transform!( data_base, mop )
	else
		result_type = Result{ XT, YT }
		data_base = init_db( ArrayDB, result_type, Nothing )
		set_transformed!(data_base, true)
	end
	
	# make sure, x & fx are in database
	x_id = ensure_contains_values!(data_base, x, fx)
	set_x_index!(id, x_id)

	# make the problem static 
	# initialize surrogate models
	sc = init_surrogates(SurrogateContainer, smop, id, data_base, ac )

	# now that we have meta data available, we retrive the 
	# right saveable type and make a new database, that can handle `CT`
	CT = get_saveable_type( sc )
	saveable_type = IterSaveable{DT, CT}
	new_data_base = copy_db(data_base; saveable_type )
	return (smop, id, new_data_base, sc, ac)
end

function iterate!( iter_data :: AbstractIterData, data_base :: AbstractDB, mop :: AbstractMOP, 
	sc :: SurrogateContainer, algo_config :: AbstractConfig ) :: STOP_CODE

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
        |  Values are $(_prettify(reverse_internal_sorting(fx,mop)))
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
		if DO_CRIT_LOOPS
			num_critical_loops = 0
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
			if num_critical_loops > 0
			end
		end
		@logmsg loglevel1 "Exiting after $(num_critical_loops) loops with ω = $(ω) and Δ = $(get_delta(iter_data))."
	end
				
	ω, x₊, mx₊, steplength = compute_descent_step(mop, iter_data, data_base, sc, algo_config, ω, ω_data)

    if ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
        return CRITICAL
    end

	mx = eval_models(sc, x)
	fx₊ = eval_all_objectives(mop, x₊)
	
	if strict_acceptance_test( algo_config )
		_ρ = minimum( (fx .- fx₊) ./ (mx .- mx₊) )
	else
		_ρ = (maximum(fx) - maximum( fx₊ ))/(maximum(mx) - maximum(mx₊))
	end
	ρ = isnan(_ρ) ? -Inf : _ρ
	
	@logmsg loglevel2 """\n
	Attempting descent of length $steplength.
	| f(x)  | $(_prettify(fx))
	| f(x₊) | $(_prettify(fx₊))
	| m(x)  | $(_prettify(mx))
	| m(x₊) | $(_prettify(mx₊))
	The error betwenn f(x) and m(x) is $(sum(abs.(fx .- mx))).
	$(strict_acceptance_test(algo_config) ? "All" : "One") of the components must decrease.
	Thus, ρ is $ρ.
	"""

	# put new values in data base 
	trial_index = new_result!(data_base, x₊, fx₊)

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
		x_trial_index = trial_index,
		ρ, ω, sc,
		stepsize = steplength )
	stamp!( data_base, stamp_content )

	# update iter data 
	set_delta!(iter_data, Δ)
	if it_stat(iter_data) in [SUCCESSFULL, ACCEPTABLE]
		set_x!(iter_data, x₊)
		set_fx!(iter_data, fx₊)
		set_x_index!( iter_data, trial_index )
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
function optimize( mop :: AbstractMOP, x0 :: Vec, fx0 :: Vec = Float32[]; 
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractDB, Nothing} = nothing )
    
	mop, iter_data, data_base, sc, ac = initialize_data(mop, x0, fx0; algo_config, populated_db)
    @logmsg loglevel1 _stop_info_str( ac, mop )

    @logmsg loglevel1 "Entering main optimization loop."
	ret_code = CONTINUE
    while ret_code == CONTINUE
        ret_code = iterate!(iter_data, data_base, mop, sc, ac)
    end# while

	ret_x, ret_fx = get_return_values(data_base, iter_data, mop )
   
    @logmsg loglevel1 _fin_info_str(data_base, iter_data, mop, ret_code)

    return ret_x, ret_fx, ret_code, data_base
end# function optimize


end