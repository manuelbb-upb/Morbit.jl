module Morbit

using LinearAlgebra: norm, I, qr, lu, cholesky, givens, tril, pinv
using LinearAlgebra: rank;
using ForwardDiff: jacobian, gradient
using JuMP
using OSQP
using NLopt
#using Ipopt
using Random: shuffle!
using Parameters: @with_kw, @unpack, @pack!, reconstruct   # use Parameters package for structs with default values and keyword initializers
import Base: isempty, Broadcast, broadcasted

export RBFModel, train!, improve!
export optimize!
export AlgoConfig, IterData, MOP, HeterogenousMOP

include("PointSampler.jl")
using .PointSampler: monte_carlo_th

include("data_structures.jl")
include("rbf.jl")
include("sampling.jl")
include("training.jl")
include("constraints.jl")
include("descent.jl")
include("plotting.jl")
include("objectives.jl")

function optimize!( config_struct :: AlgoConfig, problem::MOP, x₀::Array{Float64,1} )
    f_x_0 = problem.f(x_0)
    n_out = length( f_x_0 )
    mixed_mop = MixedMOP( lb = problem.lb, ub = problem.ub )
    add_objective!(mixed_mop, problem.f, :expensive, n_out )

    optimize!( config_struct, mixed_mop, x₀, f_x_0);
end

function optimize!( config_struct :: AlgoConfig, problem::Union{MOP, MixedMOP} )
    if !isempty( problem.x_0 )
        optimize!(config_struct, problem, problem.x_0)
    else
        error("Set vector x_0 for problem instance or provide it as 3rd argument.")
    end
end

function optimize!( config_struct :: AlgoConfig, problem::MixedMOP, x₀::Vector{Float64}, f_x₀ :: Vector{Float64} = Float64[])

    # unpack parameters from settings
    @unpack n_vars, Δ₀ = config_struct;
    @unpack max_iter, max_evals, max_critical_loops, Δ_critical, Δ_min, stepsize_min = config_struct;   # stopping criteria parameters
    @unpack μ, β, ε_crit, ν_success, ν_accept, γ_crit, γ_grow, γ_shrink, γ_shrink_much = config_struct; # algorithm parameters
    @unpack Δ_max, θ_enlarge_1, θ_enlarge_2, θ_pivot, θ_pivot_cholesky = config_struct                  # sampling parameters
    @unpack all_objectives_descent, descent_method = config_struct;                                     # descent step parameter
    @unpack rbf_kernel, rbf_shape_parameter, max_model_points = config_struct;                          # rbf model parameters
    @unpack ideal_point, image_direction = config_struct;                            # objective related parameters

    @pack! config_struct = problem   # so that subroutines can evaluate the objectives etc.

    # set n_vars dependent variables
    if n_vars == 0
        n_vars = length( x₀ );
        config_struct.n_vars = n_vars;
        if max_model_points == 1
            max_model_points = 2*config_struct.n_vars^2 + 1;
            config_struct.max_model_points = max_model_points;
        end
    end
    if θ_enlarge_2 == 0
        θ_enlarge_2 = max( 10, sqrt(n_vars) );  # as in ORBIT paper according to Wild
        config_struct.θ_enlarge_2 = θ_enlarge_2;
    end

    # reorder ideal_point and image_direction if indices are different
    if !isempty(ideal_point) ideal_point[:] = apply_internal_sorting( problem, ideal_point) end
    if !isempty(image_direction) image_direction[:] = apply_internal_sorting( problem, image_direction ) end

    problem.is_constrained && @info("CONSTRAINED PROBLEM")

    # setup stopping functions
    Δ_big_enough( Δ, stepsize ) = Δ >= Δ_min || ( stepsize >= stepsize_min && Δ >= Δ_critical )
    function loop_check(Δ, stepsize, iter_index)
        check_iter_index = iter_index < max_iter
        !check_iter_index && @info "\nMaximum number of iterations reached.\n"

        check_budget = ( (!improvement_step && length( sites_db  ) <= max_evals - 1) || ( improvement_step && length( sites_db ) <= max_evals - 2 ) )
        !check_budget && @info "\nBudget exhausted.\n"

        check_stepsize = Δ_big_enough( Δ, stepsize );
        !check_stepsize && @info "\nStepsize or Δ too small\n\t\t\tΔ = $Δ\n\t\t\tΔ_min = $Δ_min\n\t\t\tΔ_critical = $Δ_critical"

        return check_iter_index && check_budget && check_stepsize
    end

    # BEGINNING OF OPTIMIZATION ROUTINE

    # initialize iteration sites/values and databases
    Δ = Δ₀ #maximum( scaling_func( Δ₀ ) ); # TODO THINK ABOUT THIS
    x = scale(problem, x₀ );
    @info "Starting at x₀ = $x with Δ₀= $Δ (scaled to unit hypercube [0,1]^n)."

    config_struct.n_exp = problem.n_exp        # set number of expensive objectives    TODO remove from config_struct
    config_struct.n_cheap = problem.n_cheap;   # set number of cheap objectives

    if isempty( f_x₀ )
        f_x = eval_all_objectives( problem, x )
    else
        if length(f_x₀) == problem.n_exp + problem.n_cheap
            f_x = f_x₀
        else
            error("f_x₀ has wrong length.")
        end
    end

    if isnothing( config_struct.iter_data )
        iter_data = IterData( x = x, f_x = f_x, Δ = Δ, sites_db = [], values_db = [], update_extrema = config_struct.scale_values);   # make values available to subroutines
    else
        # re-use old database entries
        iter_data = IterData( x = x, f_x = f_x, Δ = Δ, sites_db = config_struct.iter_data.sites_db, values_db = config_struct.iter_data.values_db, update_extrema = config_struct.scale_values);
    end
    @pack! config_struct = iter_data;
    @unpack sites_db, values_db = iter_data;
    if !(x ∈ sites_db)
        push!(sites_db , x )
        push!(iter_data, f_x)
    end

    # initialize surrogate model
    @info("Initializing model.")
    rbf_model = build_model( config_struct, problem.is_constrained )

    # enter optimization loop
    exit_flag = false;
    iter_index = 0;
    improvement_step = false;
    stepsize = Δ;

    accept_x₊ = true;

    @unpack iterate_indices, model_info_array, stepsize_array, Δ_array, ω_array, ρ_array, num_crit_loops_array = iter_data;   # unpack data collection containers
    push!(iterate_indices, 1);
    while loop_check(Δ, stepsize, iter_index)

        iter_index += 1;

        @info("\n----------------------\nIteration $iter_index.")
        @info("\tCurrent trust region radius is $Δ.")
        @info("\tCurrent number of function evals is $(length(sites_db)).")
        @info("\tCurrent (scaled) iterate is $(x[1:min(5,end)])...")
        @info("\tCurrent values are $(f_x[1:min(5,end)]).")

        # model update

        if iter_index > 1   # model is initialized outside of while to store it in iter data; update not at end of while to save computation if no more iterations are comming
            if improvement_step
                @info("\tImproving model.")
                improve!(rbf_model, config_struct, problem.is_constrained);
            else
                @info("\tUpdating model")
                rbf_model = build_model( config_struct, problem.is_constrained ) # "update" model (actually constructs a new model instance)
            end
        end

        # compute descent step
        @info("\tComputing descent step.")
        ω, d, stepsize = compute_descent_direction( config_struct, rbf_model ) ;#Val(descent_method), rbf_model, f_cheap, x, f_x, Δ, problem.is_constrained, all_objectives_descent, ideal_point, image_direction, θ_enlarge_1)
        @info("\t\tCriticality measure ω is $ω.")

        # Criticallity Test
        num_critical_loops = 0;
        if ω <= ε_crit      # analog to small gradient
            @info("\tEntered criticallity test!")
            if !rbf_model.fully_linear
                n_improvements = make_linear!(rbf_model, config_struct, problem.is_constrained);
                if n_improvements > 0
                    ω, d, stepsize = compute_descent_direction( config_struct, rbf_model ) ;#compute_descent_direction(Val(descent_method), rbf_model, f_cheap, x, f_x, Δ, problem.is_constrained,  all_objectives_descent, ideal_point, image_direction, θ_enlarge_1)
                end
            end

            exit_flag = !( rbf_model.fully_linear && Δ_big_enough(Δ,stepsize) && num_critical_loops < max_critical_loops );    # exit if model is not fully linear ( budget exhausted in previous linearization )
            while Δ > μ * ω && !exit_flag
                if num_critical_loops < max_critical_loops && length(sites_db) < max_evals - 1
                    num_critical_loops += 1
                    Δ *= γ_crit
                    @pack! iter_data = Δ;
                    @info("\t\tCritical loop no. $num_critical_loops with Δ = $Δ > $μ * $ω = $(μ*ω)")
                    # NOTE matlab version able to set exit flag here when eval budget is exhausted -> then breaks
                    changed = make_linear!(rbf_model, config_struct, Val(true), problem.is_constrained);
                    if changed
                        ω, d, stepsize = compute_descent_direction( config_struct, rbf_model ) ;#compute_descent_direction( Val(descent_method), rbf_model, f_cheap, x, f_x, Δ, problem.is_constrained,  all_objectives_descent, ideal_point, image_direction, θ_enlarge_1)
                        exit_flag = !Δ_big_enough(Δ, stepsize)
                    else
                        @info "\t\t\tModel is still linear"
                    end
                else
                    exit_flag = true;
                end
            end

            if exit_flag
                @info("\tExit from main loop because maximum number of critical loops is or reached or computational budget is exhausted.")
                break;      # TODO: maybe change so that an descent attempt is made nevertheless
            end

            @info("\tTrust region radius now is $Δ, resulting in ω = $ω.")
        end

        # NOTE: MATLAB version tested if d != 0 ... but criticallity test should ensure that

        # apply step and evaluate at new sites
        @info("\tAttempting descent of length $stepsize.")

        x₊ = x + d;
        m_x = eval_surrogates( problem, rbf_model, x )
        m_x₊ = eval_surrogates( problem, rbf_model, x₊ )
        f_x₊ = eval_all_objectives(problem, x₊)

        if config_struct.scale_values
            M_x = scale(iter_data, m_x)
            M_x₊ = scale( iter_data, m_x₊)
            F_x₊ = scale( iter_data, f_x₊)
            F_x = scale( iter_data, f_x )
        else
            M_x, M_x₊, F_x₊, F_x = m_x, m_x₊, f_x₊, f_x
        end

        @info("\t\tm_x   = $M_x")
        @info("\t\tm_x_+ = $M_x₊")
        @info "\t\tf_x_+ = $F_x₊"

        # acceptance test ratio
        if all_objectives_descent
            ρ = minimum( (F_x .- F_x₊) ./ (F_x .- M_x₊) )
        else
            max_f_x = maximum( F_x );
            ρ = ( max_f_x - maximum( F_x₊ ) ) / ( max_f_x - maximum( M_x₊ ) )
        end

        @info("\tρ is $ρ")

        # save values to database
        push!(sites_db, x₊)
        #push!(values_db, f_x₊)
        push!(iter_data, f_x₊)

        # Update iter data components (I)
        push!( Δ_array, Δ);
        push!( ω_array, ω );
        push!( stepsize_array, norm( d, Inf) );
        push!( num_crit_loops_array, num_critical_loops );
        push!( ρ_array, ρ );
        if config_struct.n_exp > 0
            push!( model_info_array, deepcopy(rbf_model.model_info) );
        end

        # perform iteration step
        Δ_old = copy(Δ) # for printing
        accept_x₊ = false;
        improvement_step = false;
        if !isnan(ρ) && ρ > ν_success
            @info("\tVery successful descent step.")
            if Δ < β * ω
                Δ = min( Δ_max, γ_grow * Δ );
                @info("\t\tTrying to grow Δ from $Δ_old to $Δ.")
            end
            accept_x₊ = true
        else
            if rbf_model.fully_linear
                if isnan(ρ) || ρ < ν_accept
                    Δ *= γ_shrink_much;
                else
                    Δ *= γ_shrink;  # shrink Δ even if x is accepted, see diagram in paper
                    @info("\tAcceptable descent step.")
                    accept_x₊ = true;
                end
                @info("\t\tShrinking Δ from $Δ_old to $Δ.")
            else
                improvement_step = true;
            end
        end

        if accept_x₊
            x = x₊;
            f_x = f_x₊;
            push!(iterate_indices, length(sites_db));
        else
            push!(iterate_indices, iterate_indices[end])
            @info("\tUnsuccessful descent attempt.")
        end

        # update iteration data for subsequent subroutines
        @pack! iter_data = x, f_x, Δ
    end

    # Finished!
    @info("\n\n--------------------------")
    @info("Finished optimization after $iter_index iterations and $(length(sites_db)) evaluations.")
    @info("Final (unscaled) x  = $(unscale(problem, x)).")
    @info("Final fx = $f_x.")

    # Reverse scaling on all decision vectors
    sites_db[:] = unscale.(problem, sites_db )
    # Reverse sorting on all value vectors
    if !isempty(ideal_point) ideal_point[:] = reverse_internal_sorting( problem, ideal_point) end
    if !isempty(image_direction) image_direction[:] = reverse_internal_sorting( problem, image_direction ) end
    values_db[:] = reverse_internal_sorting( problem, values_db )
    return unscale(problem, x), f_x
end

end
