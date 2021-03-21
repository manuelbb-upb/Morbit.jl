module Morbit

export morbit_formatter;
export MixedMOP, optimize, AlgoConfig, add_objective!, add_vector_objective!;
export ExactConfig, TaylorConfig, RbfConfig, LagrangeConfig;
export save_config, save_database, save_iter_data;
export load_config, load_database, load_iter_data;

# steepest descent
using LinearAlgebra: norm
import JuMP;
import OSQP;

# LagrangeModels and PS
import NLopt;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;
using Memoize: @memoize, memoize_cache;
import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

import Logging: LogLevel, @logmsg;
import Logging;

const loglevel1 = LogLevel(-1);
const loglevel2 = LogLevel(-2);
const loglevel3 = LogLevel(-3);
const loglevel4 = LogLevel(-4);

const printDict = Dict( 
    loglevel1 => (:blue, "Morbit"),
    loglevel2 => (:cyan, "Morbit "),
    loglevel3 => (:green, "Morbit  "),
    loglevel4 => (:green, "Morbit   ")
)

function morbit_formatter(level::LogLevel, _module, group, id, file, line)
    @nospecialize
    if level in keys(printDict)
        color, prefix = printDict[ level ]
        suffix::String = ""
        return color, prefix, ""
    else 
        return Logging.default_metafmt( level, _module, group, id, file, line )
    end
end

include("shorthands.jl");

include("Interfaces.jl");

include("diff_wrappers.jl");

# implementations
include("VectorObjectiveFunction.jl");
include("MixedMOP.jl");

include("results.jl");  
include("Surrogates.jl");

include("objectives.jl");

include("ConfigImplementations.jl")
include("descent.jl")

include("saving.jl")

function _budget_okay( mop :: AbstractMOP, ac :: AbstractConfig ) :: Bool
    for objf ∈ list_of_objectives(mop)
        if num_evals(objf) >= min( max_evals(objf), max_evals(ac) ) - 1
            return false;
        end
    end
    return true
end

"True if stepsize or radius too small."
function _rel_tol_test_decision_space( Δ :: Union{Real,RVec}, steplength :: Real, ac :: AbstractConfig) :: Bool 
    ret_val =  all(Δ .<= Δₗ(ac)) || all(steplength .< stepsize_min(ac )) || 
        all( Δ .<= Δ_crit(ac) ) && all( steplength .<= stepsize_crit(ac) );
    if ret_val
        @logmsg loglevel1 """\n
                Radius or stepsize too small.
                Δ = $(Δ), stepsize = $(steplength).
                Δ_min = $(Δₗ(ac)), Δ_crit = $(Δ_crit(ac)).
                stepsize_min = $(stepsize_min(ac)), stepsize_crit = $(stepsize_crit(ac)).
            """
    end
    return ret_val
end

function shrink_radius( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return Δ * γ_shrink(ac);
    elseif radius_update_method(ac) == :steplength
        return steplength * γ_shrink(ac);
    end
end
function shrink_radius_much( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return Δ * γ_shrink_much(ac);
    elseif radius_update_method(ac) == :steplength
        return steplength * γ_shrink_much(ac);
    end
end
function grow_radius( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return min( Δᵘ(ac), γ_grow(ac) * Δ )
    elseif radius_update_method(ac) == :steplength
        return min( Δᵘ(ac), ( γ_grow + steplength/Δ ) * Δ );
    end
end

using Printf: @sprintf
function _prettify( vec :: RVec, len :: Int = 5) :: AbstractString
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

############################################
function optimize( mop :: AbstractMOP, x⁰ :: RVec, 
    fx⁰ :: RVec = Real[]; algo_config :: AbstractConfig = EmptyConfig(), 
    populated_db :: Union{AbstractDB,Nothing} = nothing # TODO make passing of AbstractIterData possible
    )

    # parse fix configuration parameters
    ν_succ = ν_success( algo_config );
    ν_acc = ν_accept( algo_config );
    mu = μ( algo_config );
    beta = β( algo_config );
    eps_crit = ε_crit( algo_config );
    gamma_crit = γ_crit( algo_config );

    # TODO warn here 
    reset_evals!( mop );
    # for backwards-compatibility with unconstrained problems:
    if num_vars(mop) == 0
        MOI.add_variables(mop, length(x⁰))
    end

    # initialize first iteration site
    @assert !isempty( x⁰ );
    x = scale( x⁰, mop );
    
    # initalize first objective vector 
    if isempty( fx⁰ )
        # if no starting function value was provided, eval objectives
        fx = eval_all_objectives( mop, x );
    else
        fx = apply_internal_sorting( y, mop );
    end

    # initiliza database
    if !isnothing(populated_db)
        # has a database been provided? if yes, prepare
        data_base = populated_db;
        scale!( data_base, mop );
        apply_internal_sorting!( data_base, mop );
    else
        data_base = init_db(use_db(algo_config));
    end
    # make sure, x & fx are in database
    start_res = init_res(Res, x, fx );
    ensure_contains_result!(data_base, start_res);

    iter_data = init_iter_data(IterData, x, fx, Δ⁰(algo_config), data_base);
    xᵗ_index!(iter_data, get_id(start_res));

    # initialize surrogate models
    sc = init_surrogates( mop, iter_data, algo_config );
    sc.surrogates

    it_stat = SUCCESSFULL;
    n_improvements = n_iterations = 0
    MAX_ITER = max_iter(algo_config)
    steplength = Inf;   # set here for initial stopping test
    @logmsg loglevel1 "Entering main optimization loop."
    while n_iterations < MAX_ITER
        # read iter data to handy variables
        x = xᵗ(iter_data);
        fx = fxᵗ(iter_data);
        Δ = Δᵗ(iter_data);

        #@assert all( isapprox.( get_site( iter_data, xᵗ_index(iter_data ) ), x ) ); 
        #@assert get_value(iter_data,xᵗ_index(iter_data)) == fx
 
        # check other stopping conditions (could also be done in head of while-loop,
        # but looks a bit more tidy here
        if !_budget_okay(mop, algo_config)
            @logmsg loglevel1 "Stopping. Computational budget is exhausted."
            break;
        end
        
        # relative stopping (decision space)
        if _rel_tol_test_decision_space( Δ, steplength, algo_config)
            break;
        end

        # set iteration counter
        if it_stat != MODELIMPROVING || count_nonlinear_iterations(algo_config) 
            n_iterations += 1
            n_improvements = 0;
        else 
            n_improvements += 1;
        end
        
        @logmsg loglevel1 """\n
            |--------------------------------------------
            |Iteration $(n_iterations).$(n_improvements)
            |--------------------------------------------
            |  Current trust region radius is $(Δ).
            |  Current number of function evals is $(num_evals(mop)).
            |  Iterate is $(_prettify(unscale(x, mop)))
            |  Values are $(_prettify(reverse_internal_sorting(fx,mop)))
            |--------------------------------------------
        """

        # update surrogate models
        if n_iterations > 1
            if it_stat == MODELIMPROVING 
                improve_surrogates!( sc, mop, iter_data, algo_config; ensure_fully_linear = false );
            else
                update_surrogates!( sc, mop, iter_data, algo_config; ensure_fully_linear = false );
            end
        end

        # calculate descent step and criticality value
        ω, x₊, mx₊, steplength = compute_descent_step( algo_config, mop, iter_data, sc )
        @logmsg loglevel1 "Criticality is ω = $(ω)."

        # Criticallity test
        _fully_linear = fully_linear(sc)
        num_critical_loops = 0;

        if ω <= eps_crit && (!_fully_linear || all(Δ .> mu * ω))
            @logmsg loglevel1 "Entered Criticallity Test."
            if !_fully_linear
                @logmsg loglevel1 "Ensuring all models to be fully linear."
                update_surrogates!( sc, mop, iter_data, algo_config; ensure_fully_linear = true );
                
                ω, x₊, mx₊, steplength = compute_descent_step(algo_config,mop,iter_data,sc);
                if !fully_linear(sc)
                    @logmsg loglevel1 "Could not make all models fully linear. Trying one last descent step."
                    @goto MAIN;
                end
            end
            
            while all(Δᵗ(iter_data) .> mu * ω)
                @logmsg loglevel1 "Criticality loop $(num_critical_loops + 1)." 
                if num_critical_loops >= max_critical_loops(algo_config)
                    @logmsg loglevel1 "Maximum number ($(max_critical_loops(algo_config))) of critical loops reached. Exiting..."
                    @goto EXIT_MAIN
                end
                if !_budget_okay(mop, algo_config)
                    @logmsg loglevel1 "Computational budget exhausted. Exiting…"
                    @goto EXIT_MAIN
                end
                
                # shrink radius
                Δᵗ!( iter_data, Δᵗ(iter_data) .* gamma_crit );
                # make model linear 
                update_surrogates!( sc, mop, iter_data, algo_config; ensure_fully_linear = true );
                # (re)calculate criticality
                # TODO make backtracking optional and don't do here
                ω, x₊, mx₊, steplength = compute_descent_step(algo_config,mop,iter_data,sc);

                if _rel_tol_test_decision_space( Δᵗ(iter_data), steplength, algo_config)
                    @goto EXIT_MAIN 
                end

                if !fully_linear(sc)
                    @logmsg loglevel1 "Could not make all models fully linear. Trying one last descent step."
                    @goto MAIN;
                end

                num_critical_loops += 1;
            end
            "Exiting after $(num_critical_loops) loops with ω = $(ω) and Δ = $(Δᵗ(iter_data))."
            @goto MAIN
            @label EXIT_MAIN 
            break;
        end# Crit test if 

        @label MAIN # re-entry point after criticality test 

        mx = eval_models(sc, x);
        fx₊ = eval_all_objectives(mop, x₊);
        
        if strict_acceptance_test( algo_config )
            ρ = minimum( (fx .- fx₊) ./ (mx .- mx₊) )
        else
            ρ = (maximum(fx) - maximum( fx₊ ))/(maximum(mx)-maximum(mx₊))
        end
        
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
        #@assert all( mx .>= mx₊ )

        ρ = isnan(ρ) ? -Inf : ρ;
        Δ = Δᵗ(iter_data);  # if it was changed in criticality test
        if ρ >= ν_succ
            if Δ < beta * ω
                new_Δ = grow_radius(algo_config, Δ, steplength);
            else
                new_Δ = Δ;
            end
            it_stat = SUCCESSFULL
        else
            if fully_linear(sc)
                if ρ < ν_acc
                    new_Δ = shrink_radius_much(algo_config, Δ, steplength);
                    it_stat = INACCEPTABLE
                else
                    new_Δ = shrink_radius(algo_config, Δ, steplength);
                    it_stat = ACCEPTABLE
                end
            else
                it_stat = MODELIMPROVING
                new_Δ = Δ
            end
        end

        # accept x?
        old_x_index = xᵗ_index(iter_data); # for stamp!ing
        if it_stat == SUCCESSFULL || it_stat == ACCEPTABLE
            trial_point_index = set_next_iterate!(iter_data, x₊, fx₊, new_Δ);
            #@assert all(isapprox.(x₊,xᵗ(iter_data)))
        else
            trial_point_index = keep_current_iterate!(iter_data, x₊, fx₊, new_Δ);
            #@assert all( isapprox.(x, xᵗ(iter_data)))
        end
        @logmsg loglevel1 """\n
            The trial point was $((it_stat == SUCCESSFULL || it_stat == ACCEPTABLE) ? "" : "not ")accepted.
            The iteration is $(it_stat).
            Moreover, the radius was updated as below:
            old radius : $Δ
            new radius : $new_Δ ($(round(new_Δ/Δ * 100;digits=1)) %)
        """
        
        stamp!(data_base, Dict( 
                "iter_status" => it_stat,
                "xᵗ_index" => old_x_index,
                "xᵗ₊_index" => trial_point_index,
                "ρ" => ρ,
                "Δ" => Δ, 
                "ω" => ω,
                "num_critical_loops" => num_critical_loops,
                "stepsize" => steplength,
                "model_meta" => [ deepcopy(sw.meta) for sw ∈ sc.surrogates ]            
            )
        );
    end

    ret_x = unscale(xᵗ(iter_data),mop);
    ret_fx = reverse_internal_sorting(fxᵗ(iter_data),mop);
    @logmsg loglevel1 """\n
        |--------------------------------------------
        | FINISHED
        |--------------------------------------------
        | No. iterations:  $(n_iterations) 
        | No. evaluations: $(num_evals(mop))
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """

    # unscale sites and re-sort values to return to user
    if !isnothing( db(iter_data) )
        unscale!( db(iter_data) , mop);
        reverse_internal_sorting!( db(iter_data), mop);
    end

    return ret_x, ret_fx, iter_data

end# function optimize

end#module