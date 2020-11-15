module Morbit

# import Solvers
using JuMP
using OSQP
using NLopt

using LinearAlgebra: norm, pinv

using Parameters: @with_kw, @unpack, @pack!, reconstruct   # use Parameters package for structs with default values and keyword initializers
import Base: isempty, Broadcast, broadcasted

export optimize!

# import MOP structures and utilities; make key features available outside this module
include("Surrogates.jl")

export MixedMOP, add_objective!, add_vector_objective!,
    ModelConfig, ExactConfig, RbfConfig, TaylorConfig, LagrangeConfig

export AlgoConfig, IterData

include("constraints.jl")
include("descent.jl")
include("plotting.jl")
include("saving.jl")

include("true_omega_calc.jl")

function optimize!( config_struct :: AlgoConfig, problem::MixedMOP, x₀::Vector{ R } where{R<:Real }, f_x₀ :: Vector{ R } where{R<:Real } = Float64[])

    # unpack parameters from settings object `config_struct`
    ## internal algorithm parameters:
    @unpack  n_vars, Δ₀, Δ_max, μ, β, ε_crit, ν_success, ν_accept, γ_crit,
        γ_grow, γ_shrink, γ_shrink_much = config_struct;
    ## stopping criteria:
    @unpack max_iter, max_evals, max_critical_loops, Δ_critical, Δ_min,
        stepsize_min, true_ω_stop = config_struct;
    ## descent related settings:
    @unpack radius_update, all_objectives_descent, descent_method, ideal_point,
        image_direction = config_struct;

    # SANITY CHECKS
    # set variables that depend on the length of input
    if n_vars == 0
        n_vars = length( x₀ );
        @pack! config_struct = n_vars;
    end

    # make sure that the constraint flag is set correctly
    if (isnothing(problem.lb) && !isnothing(problem.ub)) ||
        (!isnothing(problem.lb) && isnothing(problem.ub))
        @error("Problem must be fully box constrained or unconstrained.")
    elseif isnothing(problem.lb) && isnothing(problem.ub)
            problem.is_constrained = false
    else
        if all(isfinite.(problem.lb)) && all(isfinite.(problem.ub))
            problem.is_constrained = true
        else
            @error("Boundaries must be finite.")
        end
    end

    reset_evals!(problem)   # if warm start: reset evals to zero
    max_evals!(problem, max_evals)  # cap the maximum evaluation number
    set_non_exact_indices!(problem) # for decrease ratio calculation
    check_sorting!(problem) # to make availabe sorting via `apply_internal_sorting`

    # reference `problem` in `config_struct` to make it accesible from subroutines
    @pack! config_struct = problem;

    # reorder `ideal_point` and `image_direction` because the objective functions
    # could have different interal indices
    if !isempty(ideal_point)
        # Val(false) ⇒ Don't check if sorting index has been generated
        apply_internal_sorting!( problem, ideal_point, Val(false) )
    end
    if !isempty(image_direction)
        apply_internal_sorting!( problem, image_direction, Val(false) )
    end

    # INITILIZATION OF OPTIMIZATION ROUTINE
    ## initialize iteration site and trust region radius
    Δ = Δ₀;
    x = scale( problem, x₀ );    # scale to unit hypercube
    x_index = 1;

    @info """
    Starting at x₀ = $x $(problem.is_constrained ? "(scaled to unit hypercube [0,1]^n " : "")with Δ₀= $Δ.
    """

    ## has the first value vector been provided?
    if isempty( f_x₀ )
        # unscale x to original domain and evaluate all objectives (internal sorting)
        f_x = eval_all_objectives( problem, x )
    else
        if length(f_x₀) == problem.n_objfs
            f_x = apply_internal_sorting(problem, f_x₀, Val(false))
        else
            @error("f_x₀ has wrong length.")
        end
    end

    # Initialize `IterData` object to store iteration dependent data and
    # provide `sites_db` and `values_db` as databases for evaluations
    if isnothing( config_struct.iter_data )
        iter_data = IterData(
            x = x,
            f_x = f_x,
            x_index = 1,
            Δ = Δ,
            sites_db = [copy(x),],
            values_db = [copy(f_x),],
        );
    else
        # re-use old database entries to warm-start the optimization
        iter_data = IterData(
            x = x,
            f_x = f_x,
            Δ = Δ,
            sites_db = scale.(problem, config_struct.iter_data.sites_db),
            values_db = apply_internal_sorting.( problem,
                config_struct.iter_data.values_db, Val(false) ),
        );
        # look for x in old data
        let x_index = findfirst( [ X ≈ x for X in iter_data.sites_db ]  );   # TODO check atol
            if isnothing(x_index)
                push!(iter_data.sites_db , x )
                push!(iter_data.values_db, f_x)
                x_index = length(sites_db)
            end
            if !( iter_data.values_db[x_index] ≈ f_x )
                @error "f_x at starting point does not equal provided database value (at index $x_index)!"
            end
            iter_data.x_index = x_index
        end
    end
    @pack! config_struct = iter_data;
    @unpack sites_db, values_db = iter_data;

    # initialize surrogate model container and build the surrogates at first `x`
    @info("Initializing model container.")
    surrogates = init_surrogates( config_struct );

    # if desired (for debugging and testing)
    # initialize shadow surrogate container for true ω calculation
    DEBUG_STOP = isfinite( true_ω_stop )
    if DEBUG_STOP
        shadow_surrogates = create_shadow_surrogates( config_struct );
    else
        shadow_surrogates = SurrogateContainer();
    end

    # make availabe information arrays for introspection and analysis
    @unpack iterate_indices, stepsize_array, Δ_array, ω_array, ρ_array,
        num_crit_loops_array, model_meta = iter_data;
    push!(iterate_indices, iter_data.x_index);

    # enter optimization loop
    iter_index = 0.0;
    improvement_step = false; improvement_counter = 0;
    steplength = Δ;
    non_linear_indices = Int64[];
    
    TRUE_OMEGA_SMALL, true_ω = true_ω_small(Val(DEBUG_STOP), config_struct, shadow_surrogates) 

    while budget_okay( problem, improvement_step ) && loop_check(Δ, steplength, iter_index;
            Δ_min, Δ_critical, stepsize_min, max_iter, improvement_step) && !TRUE_OMEGA_SMALL

        if improvement_step
            improvement_counter += 1
            iter_index = div(iter_index,1) + (1 - exp( -improvement_counter - 1) )
        else
            improvement_counter = 0;
            iter_index = div(iter_index,1) + 1;
        end

        @info("""\n
        |--------------------------------------------
        |Iteration $iter_index.
        |--------------------------------------------
        |  Current trust region radius is $Δ.
        |  Current number of function evals is $(length(sites_db)).
        |  Current (possibly scaled) iterate has index $(iterate_indices[end])
        |  and coordinates $(x[1:min(5,end)])...
        |  Current values are $(f_x[1:min(5,end)])...
        |--------------------------------------------
        """)

        # model update
        if improvement_step
            @info("Improving models.")
            improve!(surrogates, non_linear_indices, config_struct )
        else
            @info("Updating models…")
            build_models!( surrogates, config_struct );
        end
        linear_flag, non_linear_indices = fully_linear(surrogates);

        # compute descent step
        @info("Computing descent step.")
        ω, d, x₊, m_x₊ = compute_descent_direction( config_struct, surrogates )
        steplength = norm( d, Inf);
        @info("\tCriticality measure ω is $ω.")

        # Criticallity Test
        num_critical_loops = 0;
        if ω <= ε_crit      # analog to small gradient
            Δ_old, ω_old = Δ, ω;    # for printing only
            @info("Entered criticallity test!")

            # check linearity on current trust region
            if !linear_flag
                @info "\t Making initial model linear."
                has_changed = make_linear!(surrogates, non_linear_indices, config_struct)
                if has_changed && fully_linear(surrogates)[1]
                    ω, d, x₊, m_x₊ = compute_descent_direction( config_struct, surrogates )
                end
            end

            # exit if model could not be made fully linear ( budget exhausted )
            steplength = norm( d, Inf );
            linear_flag, non_linear_indices = fully_linear(surrogates)
            while ( Δ > μ * ω &&
                linear_flag &&
                num_critical_loops < max_critical_loops &&
                budget_okay(problem, false) &&
                Δ_big_enough( Δ, steplength; Δ_min, Δ_critical, stepsize_min)
            )

                # shrink trust region radius
                Δ *= γ_crit
                @pack! iter_data = Δ;
                @info("\t\tCritical loop no. $num_critical_loops with Δ = $Δ > $μ * $ω = $(μ*ω)")
                build_models!( surrogates, config_struct )

                ω, d, x₊, m_x₊ = compute_descent_direction( config_struct, surrogates );
                steplength = norm( d, Inf );
                linear_flag, non_linear_indices = fully_linear(surrogates);
                num_critical_loops += 1
            end
            Δ = min( Δ_old, max( Δ, β * ω ));
            @info """
            Finished criticality test with
                • $num_critical_loops criticality loops (max. $max_critical_loops)
                • We now have $(length(sites_db)) objective evaluations.
                • Δ went from $Δ_old to $Δ.
                • ω went from $ω_old to $ω.
                • Last stepsize was $steplength.
                • Surrogates are $(linear_flag ? "" : "not") fully linear.
            """

            if ( num_critical_loops == max_critical_loops ||
                !Δ_big_enough( Δ, steplength; Δ_min, Δ_critical, stepsize_min) )
                @info """
                \tExit from main loop because maximum number of critical loops is or reached
                or stepsize is to small.
                """
                break; # breack from main loop
            end

        end

        # Apply step and evaluate at new sites
        @info("Attempting descent of length $steplength.")

        #x₊ = x .+ d;
        #m_x₊ = eval_models( surrogates, x₊ );
        m_x = eval_models( surrogates, x );
        f_x₊ = eval_all_objectives(problem, x₊)   # unscale and evaluate

        # retrieve expensive components only
        M_x, M_x₊, F_x₊, F_x = expensive_components.( [m_x, m_x₊, f_x₊, f_x], problem, Val(false))

        # acceptance test ratio
        if isempty(problem.non_exact_indices)
            ρ = 1.0
        else
            if all_objectives_descent
                ρ = minimum( (F_x .- F_x₊) ./ (F_x .- M_x₊) )
            else
                ρ = let max_F_x = maximum( F_x ); ( max_F_x - maximum( F_x₊ ) ) / ( max_F_x - maximum( M_x₊ ) ) end;
            end
        end

        @info """\n
        We have the following evaluations:
        | f(x)  | $(f_x[1:min(end,5)])
        | m(x)  | $(m_x[1:min(end,5)])
        | f(x₊) | $(f_x₊[1:min(end,5)])
        | m(x₊) | $(m_x₊[1:min(end,5)])
        The error betwenn f(x) and m(x) is $(sum(abs.(f_x .- m_x))).
        The expensive indices are $(problem.non_exact_indices).
        $(all_objectives_descent ? "All" : "One") of the components must decrease.
        Thus, ρ is $ρ.
        """

        # save new evaluation in database
        push!(sites_db, x₊)
        push!(values_db, f_x₊)

        # perform iteration step
        Δ_old = copy(Δ)         # for printing
        x_index_old = x_index   # for printing

        accept_x₊ = false;
        improvement_step = false;
        if !isnan(ρ) && ρ > ν_success
            if Δ < β * ω
                if radius_update == :standard
                    Δ = min( Δ_max, γ_grow * Δ );
                elseif radius_update == :steplength
                    Δ = min( Δ_max, (γ_grow + steplength/Δ ) * Δ );
                end
            end
            accept_x₊ = true
        else
            if linear_flag
                if isnan(ρ) || ρ < ν_accept
                    if radius_update == :standard
                        Δ *= γ_shrink_much;
                    elseif radius_update == :steplength
                        Δ = γ_shrink_much * steplength;
                    end
                else
                    if radius_update == :standard
                        Δ *= γ_shrink;
                    elseif radius_update == :steplength
                        Δ = γ_shrink * steplength;
                    end
                    accept_x₊ = true;
                end
            else
                improvement_step = true;
            end
        end

        if accept_x₊
            x = x₊;
            f_x = f_x₊;
            x_index = length(sites_db);
            iter_data.x_index = x_index;
        end

        @info """
        The step is $(accept_x₊ ? (ρ > ν_success ? "very sucessfull!" : "acceptable.") : "unsucessfull…")
        It follows that
          old iterate index: $x_index_old
          new iterate index: $x_index
          old radius : $Δ_old
          new radius : $Δ ($(round(Δ/Δ_old * 100;digits=1)) %)
        """

        # update iteration data for subsequent subroutines
        push!(iterate_indices, x_index);
        @pack! iter_data = x, f_x, Δ
        if accept_x₊
            # if x changed then the true criticality measure too
            TRUE_OMEGA_SMALL, true_ω = true_ω_small( Val(DEBUG_STOP), config_struct, shadow_surrogates );
        end

        # Update nice-to-know iter data information
        push!( iter_data.trial_point_indices, length(sites_db) )
        push!( Δ_array, Δ_old);
        push!( ω_array, ω );
        push!( stepsize_array, steplength );
        push!( num_crit_loops_array, num_critical_loops );
        push!( ρ_array, ρ );
        push!( model_meta, surrogates.model_meta ); # list of lists
    end
    @info """\n
    |--------------------------------------------
    | FINISHED
    |--------------------------------------------
    |  No. iterations:       $iter_index
    |  No. database entries: $(length(sites_db)).
    |  Final iterate has index $(iterate_indices[end]) and true coordinates
    |    $(unscale(problem,x)[1:min(5,end)])...
    |  Final values are $(f_x[1:min(5,end)])...
    |--------------------------------------------
    """
    @show unscale(problem,x);
    # Reverse scaling on all decision vectors
    unscale!.(problem, sites_db )
    # Reverse sorting on all value vectors
    reverse_internal_sorting!.( problem, values_db )
    if !isempty(ideal_point) reverse_internal_sorting!( problem, ideal_point) end
    if !isempty(image_direction) reverse_internal_sorting!( problem, image_direction ) end

    return sites_db[x_index], values_db[x_index], true_ω
end

# Stopping functions
@doc """
Return true if either `Δ` is big enough. Both conditions must be satisfied:
    • `Δ` is at least as large as `Δ_min`
    • `Δ >= Δ_critical` AND `stepsize >= stepsize_min`
"""
function Δ_big_enough( Δ :: Float64, stepsize :: Float64;
        Δ_min :: Float64, Δ_critical::Float64, stepsize_min :: Float64 )
    return  Δ >= Δ_min && !(Δ < Δ_critical && stepsize < stepsize_min )
end

function budget_okay( problem :: MixedMOP, improvement_step :: Bool )
    for objf ∈ problem.vector_of_objectives
        offset = (isa( objf.model_config, Union{LagrangeConfig, RbfConfig})
            && improvement_step) ? 2 : 1;
        if objf.n_evals + offset > objf.max_evals
            return false
        end
    end
    return true
end

function loop_check(Δ :: Float64, stepsize :: Float64, iter_index :: Float64;
        Δ_min :: Float64, Δ_critical :: Float64, stepsize_min :: Float64,
        max_iter :: Int64, improvement_step :: Bool )

    iterations_okay = iter_index < max_iter;
    !iterations_okay && @info """
    Maximum number of iterations reached.
        n_iterations = $iter_index,
        max_iter = $max_iter
    """

    #=
    budget_okay = (
        (!improvement_step && n_evals <= max_evals - 1) ||
        ( improvement_step && n_evals <= max_evals - 2 )
    )
    !budget_okay && @info """
    Budged exhausted:
       n_evals =   $n_evals,
       max_evals = $max_evals.
    """
    =#

    stepsize_okay = Δ_big_enough( Δ, stepsize; Δ_min, Δ_critical, stepsize_min );
    !stepsize_okay && @info """
    Stepsize or Δ too small.
    • Δ = $Δ and stepsize = $stepsize
    • Δ_min = $Δ_min
    • Δ_critical = $Δ_critical, stepsize_min = $stepsize_min.
    """

    return iterations_okay && stepsize_okay
end

end
