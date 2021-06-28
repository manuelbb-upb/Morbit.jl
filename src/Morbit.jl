module Morbit

export morbit_formatter;
export MixedMOP, optimize, AlgorithmConfig, add_objective!, add_vector_objective!;
export ExactConfig, TaylorConfig, RbfConfig, LagrangeConfig;
export save_config, save_database, save_iter_data;
export load_config, load_database, load_iter_data;

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
        # suffix ::String = ""
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

function f_tol_rel_test( fx :: RVec, fx⁺ :: RVec, ac :: AbstractConfig ) :: Bool
    tol = f_tol_rel(ac)
    if isa(tol, Real)
        ret = norm( fx .- fx⁺, Inf ) <= tol * norm( fx, Inf )
    else
        ret = all( abs.( fx .- fx⁺ ) .<= tol .* fx )
    end
    ret && @logmsg loglevel1 "Relative (objective) stopping criterion fulfilled."
    ret
end

function x_tol_rel_test( x :: RVec, x⁺ :: RVec, ac :: AbstractConfig ) :: Bool
    tol = x_tol_rel(ac)
    if isa(tol, Real)
        ret = norm( x .- x⁺, Inf ) <= tol * norm( x, Inf )
    else
        ret = all( abs.( x .- x⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Relative (decision) stopping criterion fulfilled."
    ret
end

function f_tol_abs_test( fx :: RVec, fx⁺ :: RVec, ac :: AbstractConfig ) :: Bool
    tol = f_tol_abs(ac)
    if isa(tol, Real)
        ret = norm( fx .- fx⁺, Inf ) <= tol 
    else
        ret = all( abs.( fx .- fx⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Absolute (objective) stopping criterion fulfilled."
    ret
end

function x_tol_abs_test( x :: RVec, x⁺ :: RVec, ac :: AbstractConfig ) :: Bool
    tol = x_tol_abs(ac)
    if isa(tol, Real)
        ret =  norm( x .- x⁺, Inf ) <= tol 
    else
        ret = all( abs.( x .- x⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Absolute (decision) stopping criterion fulfilled."
    ret
end

function ω_Δ_rel_test( ω :: Real, Δ :: Union{RVec, Real}, ac :: AbstractConfig )
    ω_tol = ω_tol_rel( ac )
    Δ_tol = Δ_tol_rel( ac )
    ret = ω <= ω_tol && all( Δ .<= Δ_tol )
    ret && @logmsg loglevel1 "Realtive criticality stopping criterion fulfilled."
    ret
end

function Δ_abs_test( Δ :: Union{RVec, Real}, ac :: AbstractConfig )
    tol = Δ_tol_abs( ac )
    ret = all( Δ .<= tol )
    ret && @logmsg loglevel1 "Absolute radius stopping criterion fulfilled."
    ret
end

function ω_abs_test( ω :: Real, ac :: AbstractConfig )
    tol = ω_tol_abs( ac )
    ret = ω .<= tol
    ret && @logmsg loglevel1 "Absolute criticality stopping criterion fulfilled."
    ret
end

Base.:(+)(s1 :: String, s2 :: String) = string( s1, s2 )
function _stop_info_str( ac :: AbstractConfig, mop :: Union{AbstractMOP,Nothing} = nothing )
    ret_str = "Stopping Criteria:\n"
    if isnothing(mop)
        ret_str += "No. of objective evaluations ≥ $(max_evals(ac)).\n"
    else
        for objf ∈ list_of_objectives(mop)
            ret_str += "• No. of. evaluations of objective(s) $(output_indices(objf,mop)) ≥ $(min( max_evals(objf), max_evals(ac) )).\n"
        end
    end
    ret_str += "• No. of iterations is ≥ $(max_iter(ac)).\n"
    ret_str += @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g ⋅ ‖ fx ‖,\n", f_tol_rel(ac) )
    ret_str += @sprintf("• ‖ x - x⁺ ‖ ≤ %g ⋅ ‖ x ‖,\n", x_tol_rel(ac) )
    ret_str += @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g,\n", f_tol_abs(ac) )
    ret_str += @sprintf("• ‖ x - x⁺ ‖ ≤ %g,\n", x_tol_abs(ac) )
    ret_str += @sprintf("• ω ≤ %g AND Δ ≤ %g,\n", ω_tol_rel(ac), Δ_tol_rel(ac))
    ret_str += @sprintf("• Δ ≤ %g OR", Δ_tol_abs(ac))
    ret_str += @sprintf(" ω ≤ %g.", ω_tol_abs(ac))
end

function _fin_info_str( iter_data :: AbstractIterData, mop :: Union{AbstractMOP, Nothing} = nothing )
    ret_x, ret_fx = get_return_values( iter_data, mop )
    return """\n
        |--------------------------------------------
        | FINISHED
        |--------------------------------------------
        | No. iterations:  $(num_iterations(iter_data)) 
    """ + (isnothing(mop) ? "" :
        "    | No. evaluations: $(num_evals(mop))" )+ 
    """ 
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """
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

"""
Perform initialization of the data passed to `optimize` function.
"""
function initialize_data( mop :: AbstractMOP, x0 :: Vec, fx0 :: Vec = Float16[]; 
    algo_config :: Union{AbstractConfig, Nothing} = nothing, 
    populated_db :: Union{AbstractDB, Nothing} = nothing )
    
    if num_objectives(mop) == 0
        error("`mop` has no objectives!")
    end
        
    @warn "The evaluation counter of `mop` is reset."
    reset_evals!( mop );
    # for backwards-compatibility with unconstrained problems:
    if num_vars(mop) == 0
        MOI.add_variables(mop, length(x0))
    end

    # initialize first iteration site
    @assert !isempty( x0 ) "Please provide a non-empty feasible starting point `x0`."
    
    x_scaled = scale( x0, mop );
    
    # initalize first objective vector 
    if isempty( fx0 )
        # if no starting function value was provided, eval objectives
        fx_sorted = eval_all_objectives( mop, x_scaled );
    else
        fx_sorted = apply_internal_sorting( fx0, mop );
    end 

    # ensure at least half-precision
    F = Base.promote_eltype( x_scaled, fx_sorted, Float16 )
    x = F.(x_scaled)
    fx = F.(fx_sorted)

    if isnothing( algo_config )
        ac = DefaultConfig{F}()
    else
        ac = algo_config
    end

    # initialize database
    if !isnothing(populated_db)
        # has a database been provided? if yes, prepare
        data_base = populated_db;
        transform!( data_base, mop );
    else
        data_base = init_db( use_db(ac) );
        set_transformed!(data_base, true)
    end

    # make sure, x & fx are in database
    start_res = init_res(Res, x, fx );
    ensure_contains_result!(data_base, start_res);

    iter_data = init_iter_data(IterData, x, fx, Δ⁰(ac), data_base);
    xᵗ_index!(iter_data, get_id(start_res));

    # initialize surrogate models
    sc = init_surrogates( mop, iter_data, ac );
    
    return (mop, ac, iter_data, sc)
end

function iterate!( iter_data :: AbstractIterData, mop :: AbstractMOP, 
    sc :: SurrogateContainer, algo_config :: AbstractConfig )
    
    # config params
    ν_succ = ν_success( algo_config );
    ν_acc = ν_accept( algo_config );
    mu = μ( algo_config );
    beta = β( algo_config );
    eps_crit = ε_crit( algo_config );
    gamma_crit = γ_crit( algo_config );
    count_nonlin = count_nonlinear_iterations( algo_config );

    # read iter data to handy variables
    x = xᵗ(iter_data);
    fx = fxᵗ(iter_data);
    Δ = Δᵗ(iter_data);

    # check some stopping conditions
    if num_iterations(iter_data) >= max_iter(algo_config)
        @logmsg loglevel1 "Stopping. Maximum number of iterations reached."
        return true;
    end

    if !_budget_okay(mop, algo_config)
        @logmsg loglevel1 "Stopping. Computational budget is exhausted."
        return true; # TODO stop codes
    end

    if Δ_abs_test( Δ, algo_config )
        return true;
    end
    
    # set iteration counter
    if it_stat(iter_data) != MODELIMPROVING || count_nonlin 
        inc_iterations!(iter_data)
        set_model_improvements!(iter_data, 0);
    else 
        inc_model_improvements!(iter_data)
    end
    
    @logmsg loglevel1 """\n
        |--------------------------------------------
        |Iteration $(num_iterations(iter_data)).$(num_model_improvements(iter_data))
        |--------------------------------------------
        |  Current trust region radius is $(Δ).
        |  Current number of function evals is $(num_evals(mop)).
        |  Iterate is $(_prettify(unscale(x, mop)))
        |  Values are $(_prettify(reverse_internal_sorting(fx,mop)))
        |--------------------------------------------
    """

    # update surrogate models
    if num_iterations(iter_data) > 1
        if it_stat == MODELIMPROVING 
            improve_surrogates!( sc, mop, iter_data, algo_config; ensure_fully_linear = false );
        else
            update_surrogates!( sc, mop, iter_data, algo_config; ensure_fully_linear = false );
        end
    end

    # calculate descent step and criticality value
    ω, x₊, mx₊, steplength = compute_descent_step( algo_config, mop, iter_data, sc )
    @logmsg loglevel1 "Criticality is ω = $(ω)."
    
    # stop at the end of the this loop run?
    if ω_Δ_rel_test(ω, Δ, algo_config) || ω_abs_test( ω, algo_config )
        return true;
    end

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

            if Δ_abs_test( Δᵗ(iter_data) , algo_config ) || 
                ω_Δ_rel_test(ω, Δᵗ(iter_data), algo_config) || ω_abs_test( ω, algo_config )
                
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
        return true;
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

    # update trust region radius
    ρ = isnan(ρ) ? -Inf : ρ;
    old_Δ = copy(Δᵗ(iter_data));  # if it was changed in criticality test
    if ρ >= ν_succ
        if Δ < beta * ω
            Δᵗ!(iter_data, grow_radius(algo_config, Δ, steplength) );
        end
        it_stat!(iter_data, SUCCESSFULL)
    else
        if fully_linear(sc)
            if ρ < ν_acc
                Δᵗ!(iter_data, shrink_radius_much(algo_config, Δ, steplength) );
                it_stat!(iter_data, INACCEPTABLE)
            else
                Δᵗ!(iter_data, shrink_radius(algo_config, Δ, steplength) );
                it_stat!(iter_data, ACCEPTABLE)
            end
        else
            it_stat!(iter_data, MODELIMPROVING)
        end
    end

    # accept x?
    old_x_index = xᵗ_index(iter_data); # for stamp!ing
    if it_stat(iter_data) == SUCCESSFULL || it_stat(iter_data) == ACCEPTABLE
        trial_point_index = set_next_iterate!(iter_data, x₊, fx₊);
    else
        trial_point_index = keep_current_iterate!(iter_data, x₊, fx₊);
    end
    @logmsg loglevel1 """\n
        The trial point was $( (it_stat(iter_data) == SUCCESSFULL) || (it_stat(iter_data) == ACCEPTABLE ) ? "" : "not ")accepted.
        The iteration is $(it_stat(iter_data)).
        Moreover, the radius was updated as below:
        old radius : $(old_Δ)
        new radius : $(Δᵗ(iter_data)) ($(round(Δᵗ(iter_data)/old_Δ * 100; digits=1)) %)
    """
    
    stamp!(db(iter_data), Dict( 
            "iter_status" => it_stat(iter_data),
            "xᵗ_index" => old_x_index,
            "xᵗ₊_index" => trial_point_index,
            "ρ" => ρ,
            "Δ" => Δᵗ(iter_data), 
            "ω" => ω,
            "num_critical_loops" => num_critical_loops,
            "stepsize" => steplength,
            "model_meta" => [ deepcopy(sw.meta) for sw ∈ sc.surrogates ]            
        )
    );
    
    if x_tol_rel_test( x, x₊, algo_config  ) ||
        x_tol_abs_test( x, x₊, algo_config ) ||
        f_tol_rel_test( fx, fx₊, algo_config  ) ||
        f_tol_abs_test( fx, fx₊, algo_config )
        return true;
    end
    return false
end

function get_return_values( iter_data :: AbstractIterData, mop :: AbstractMOP)
    ret_x = unscale(xᵗ(iter_data),mop);
    ret_fx = reverse_internal_sorting(fxᵗ(iter_data),mop);
    # TODO make tests for this: are returen values the same before and after `finalize_iter_data!` ?
    return (ret_x, ret_fx)
end

function finalize_iter_data!( iter_data :: AbstractIterData, mop :: AbstractMOP )
    if !isnothing( db(iter_data) )
        untransform!(db(iter_data), mop)
    end
    nothing
end

############################################
function optimize( mop :: AbstractMOP, x0 :: Vec, fx0 :: Vec = Float16[];
    algo_config :: Union{Nothing, AbstractConfig} = nothing, 
    populated_db :: Union{AbstractDB, Nothing} = nothing # TODO make passing of AbstractIterData possible
    )
    
    mop, ac, iter_data, sc = initialize_data( mop, x0, fx0; algo_config, populated_db )

    @logmsg loglevel1 _stop_info_str( ac, mop )

    @logmsg loglevel1 "Entering main optimization loop."
    while true
        abbort = iterate!(iter_data, mop, sc, ac)
        abbort && break;            
    end# while

    ret_x, ret_fx = get_return_values( iter_data, mop )
    # unscale sites and re-sort values to return to user
    finalize_iter_data!(iter_data, mop)

    # @assert all((ret_x, ret_fx) .== get_return_values( iter_data, mop ))
   
    @logmsg loglevel1 _fin_info_str(iter_data, mop)

    return ret_x, ret_fx, iter_data
end# function optimize

end#module