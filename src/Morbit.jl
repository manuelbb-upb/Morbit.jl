module Morbit

# steepest descent
using LinearAlgebra: norm
import JuMP;
import OSQP;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;
using Memoize: @memoize, memoize_cache;
import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

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
        @info("""\n
                Radius or stepsize too small.
                Δ = $(Δ), stepsize = $(steplength).
                Δ_min = $(Δₗ(ac)), Δ_crit = $(Δ_crit(ac)).
                stepsize_min = $(stepsize_min(ac)), stepsize_crit = $(stepsize_crit(ac)).
            """
        );
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
    populated_db :: Union{AbstractDB,Nothing} = nothing )

    # parse fix configuration parameters
    ν_succ = ν_success( algo_config );
    ν_acc = ν_accept( algo_config );
    mu = μ( algo_config );
    beta = β( algo_config );
    eps_crit = ε_crit( algo_config );
    gamma_crit = γ_crit( algo_config );

    # TODO warn here 
    reset_evals!( mop );

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
        apply_internal_sorting( data_base, mop );
    else
        data_base = init_db(use_db(algo_config));
    end
    # make sure, x & fx are in database
    start_res = init_res(Res, x, fx );
    ensure_contains_result!(data_base, start_res);
    @show get_value(start_res)
    #stamp!(data_base);

    iter_data = init_iter_data(IterData, x, fx, Δ⁰(algo_config), data_base);
    xᵗ_index!(iter_data, get_id(start_res));

    # initialize surrogate models
    sc = init_surrogates( mop, iter_data );
    sc.surrogates

    IMPROVEMENT_STEP_FLAG = false;
    n_iterations = 0
    MAX_ITER = max_iter(algo_config)
    steplength = Inf;   # set here for initial stopping test
    while n_iterations < MAX_ITER
        # read iter data to handy variables
        x = xᵗ(iter_data);
        fx = fxᵗ(iter_data);
        Δ = Δᵗ(iter_data);

        @assert all( isapprox.( get_site( iter_data, xᵗ_index(iter_data ) ), x ) ); 
        @assert get_value(iter_data,xᵗ_index(iter_data)) == fx
 
        # check other stopping conditions (could also be done in head of while-loop,
        # but looks a bit more tidy here
        if !_budget_okay(mop, algo_config)
            @info "Stopping. Computational budget is exhausted."
            break;
        end
        
        # relative stopping (decision space)
        if _rel_tol_test_decision_space( Δ, steplength, algo_config)
            break;
        end

        # set iteration counter
        if !IMPROVEMENT_STEP_FLAG || count_nonlinear_iterations(algo_config) 
            n_iterations += 1
        end

        @info("""\n
            |--------------------------------------------
            |Iteration $(n_iterations).
            |--------------------------------------------
            |  Current trust region radius is $(Δ).
            |  Current number of function evals is $(num_evals(mop)).
            |  Iterate is $(_prettify(x))
            |  Values are $(_prettify(fx))
            |--------------------------------------------
        """);

        # update surrogate models
        if n_iterations > 1
            if IMPROVEMENT_STEP_FLAG 
                improve_surrogates!( sc, mop, iter_data; ensure_fully_linear = false );
            else
                update_surrogates!( sc, mop, iter_data; ensure_fully_linear = false );
            end
        end

        # calculate descent step and criticality value
        ω, x₊, mx₊, steplength = compute_descent_step( algo_config, mop, iter_data, sc )
        @info "Criticality is ω = $(ω)."

        # Criticallity test
        _fully_linear = fully_linear(sc)
        if ω <= eps_crit && (!_fully_linear || all(Δ .> mu * ω))
            @info "Entered Criticallity Test."
            if !_fully_linear
                @info "Ensuring all models to be fully linear."
                update_surrogates!( sc, mop, iter_data; ensure_fully_linear = true );
                
                ω, x₊, mx₊, steplength = compute_descent_step(algo_config,mop,iter_data,sc);
                if !fully_linear(sc)
                    @info "Could not make all models fully linear. Trying one last descent step."
                    @goto MAIN;
                end
            end
            num_critical_loops = 0;
            
            while all(Δᵗ(iter_data) .> mu * ω)
                @info "Criticality loop $(num_critical_loops + 1)." 
                if num_critical_loops >= max_critical_loops(algo_config)
                    @info "Maximum number ($(max_critical_loops(algo_config))) of critical loops reached. Exiting..."
                    @goto EXIT_MAIN
                end
                if !_budget_okay(mop, algo_config)
                    @info "Computational budget exhausted. Exiting…"
                    @goto EXIT_MAIN
                end
                
                # shrink radius
                Δᵗ!( iter_data, Δᵗ(iter_data) .* gamma_crit );
                # make model linear 
                update_surrogates!( sc, mop, iter_data; ensure_fully_linear = true );
                # (re)calculate criticality
                # TODO make backtracking optional and don't do here
                ω, x₊, mx₊, steplength = compute_descent_step(algo_config,mop,iter_data,sc);

                if _rel_tol_test_decision_space( Δᵗ(iter_data), steplength, algo_config)
                    @goto EXIT_MAIN 
                end

                if !fully_linear(sc)
                    @info "Could not make all models fully linear. Trying one last descent step."
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
        
        @info """\n
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

        ACCEPT_TRIAL_POINT = false
        ρ = isnan(ρ) ? -Inf : ρ;
        Δ = Δᵗ(iter_data);  # if it was changed in criticality test
        if ρ >= ν_succ
            if Δ < beta * ω
                new_Δ = grow_radius(algo_config, Δ, steplength);
            else
                new_Δ = Δ;
            end
            ACCEPT_TRIAL_POINT = true;
        else
            if fully_linear(sc)
                if ρ < ν_acc
                    new_Δ = shrink_radius_much(algo_config, Δ, steplength);
                else
                    new_Δ = shrink_radius(algo_config, Δ, steplength);
                    ACCEPT_TRIAL_POINT = true;
                end
            else
                IMPROVEMENT_STEP_FLAG = true;
                new_Δ = Δ
            end
        end

        if ACCEPT_TRIAL_POINT
            set_next_iterate!(iter_data, x₊, fx₊, new_Δ);
            @assert all(isapprox.(x₊,xᵗ(iter_data)))
        else
            keep_current_iterate!(iter_data, x₊, fx₊, new_Δ);
            @assert all( isapprox.(x, xᵗ(iter_data)))
        end
        @info """\n
            The step is $(ACCEPT_TRIAL_POINT ? (ρ >= ν_succ ? "very sucessfull!" : "acceptable.") : "unsucessfull…")
            Moreover, the radius was updated as below:
            old radius : $Δ
            new radius : $new_Δ ($(round(new_Δ/Δ * 100;digits=1)) %)
        """

        stamp!(data_base)
    end

    ret_x = unscale(xᵗ(iter_data),mop);
    ret_fx = reverse_internal_sorting(fxᵗ(iter_data),mop);
    @info("""\n
        |--------------------------------------------
        | FINISHED
        |--------------------------------------------
        | No. iterations:  $(n_iterations) 
        | No. evaluations: $(num_evals(mop))
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """);

    # unscale sites and re-sort values to return to user
    if !isnothing( db(iter_data) )
        unscale!( db(iter_data) , mop);
        reverse_internal_sorting!( db(iter_data), mop);
    end

    return ret_x, ret_fx, iter_data

end# function optimize

end#module