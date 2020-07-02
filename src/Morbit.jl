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
import Base: isempty

export RBFModel, train!, improve!
export optimize!
export AlgoConfig, IterData, MOP, HeterogenousMOP

include("data_structures.jl")
include("rbf.jl")
include("training.jl")
include("utilities.jl")
include("constraints.jl")
include("descent.jl")
include("plotting.jl")

ε_bounds = 0.0;

function optimize!( config_struct :: AlgoConfig, prob::MOP, x₀::Array{Float64,1} )
    heterogenous_mop = HeterogenousMOP(
    f_expensive = prob.f,
    f_cheap = x -> Array{Float64,1}(),      # assume all objectives to be expensive
    lb = prob.lb,
    ub = prob.ub,
    )
    optimize!( config_struct, heterogenous_mop, x₀);
end

function optimize!( config_struct :: AlgoConfig, prob::MixedMOP, x₀::Array{Float64,1} )

    f_expensive(x) = isempty(prob.expensive_indices) ? Array{Float64,1}() : [ f(x) for f ∈ prob.vector_of_funcs[ prob.expensive_indices ]  ];
    f_cheap(x) = isempty(prob.cheap_indices) ? Array{Float64,1}() : [ f(x) for f ∈ prob.vector_of_funcs[ prob.cheap_indices ]  ];

    heterogenous_mop = HeterogenousMOP(
    f_expensive = f_expensive,
    f_cheap = f_cheap,      # assume all objectives to be expensive
    lb = prob.lb,
    ub = prob.ub,
    )
    optimize!( config_struct, heterogenous_mop, x₀);

    sorting_indices = sortperm( [ prob.expensive_indices; prob.cheap_indices ] );
    config_struct.iter_data.values_db[:] = [ value[sorting_indices] for value ∈ config_struct.iter_data.values_db ];
    return true;
end

function optimize!( config_struct :: AlgoConfig, prob::Union{MOP, HeterogenousMOP, MixedMOP} )
    if !isempty( prob.x_0 )
        optimize(prob, prob.x_0, config_struct)
    else
        error("Set x_0 for problem instance.")
    end
end

function optimize!( config_struct :: AlgoConfig, prob::HeterogenousMOP, x₀::Array{Float64,1} )
    #global x, f_x, Δ, sites_db, values_db, rbf_model; # so that redefinitions within this function are understood to be affect global vars
    global ε_bounds;

    # unpack parameters from settings
    @unpack verbosity, n_vars, rbf_kernel, ε_bounds, rbf_shape_parameter, max_model_points, max_iter, max_evals,
    μ, β, ε_crit, max_critical_loops, ν_success, ν_accept, all_objectives_descent, descent_method,
    γ_crit, γ_grow, γ_shrink, γ_shrink_much, Δ₀, Δ_max, θ_enlarge_1, θ_enlarge_2, θ_pivot, θ_pivot_cholesky,
    Δ_critical, Δ_min, stepsize_min = config_struct;

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

    is_constrained, prob, f, f_expensive, f_cheap, scaling_func, unscaling_func = init_objectives( prob, config_struct )

    is_constrained && @info("CONSTRAINED PROBLEM")

    # setup stopping functions
    Δ_big_enough( Δ, stepsize ) = Δ >= Δ_min || ( stepsize >= stepsize_min && Δ >= Δ_critical )
    function loop_check(Δ, stepsize, iter_index)
        check_iter_index = iter_index < max_iter
        !check_iter_index && @info "\nMaximum number of iterations reached.\n"

        check_budget = ( (!improvement_step && length( sites_db  ) <= max_evals - 1) || ( improvement_step && length( sites_db ) <= max_evals - 2 ) )
        !check_budget && @info "\nBudget exhausted.\n"

        check_stepsize = Δ_big_enough( Δ, stepsize );
        !check_stepsize && @info "\nStepsize or Δ too small.\n"

        return check_iter_index && check_budget && check_stepsize
    end

    # BEGINNING OF OPTIMIZATION ROUTINE

    # initialize iteration sites/values and databases
    Δ = Δ₀ #maximum( scaling_func( Δ₀ ) ); # TODO THINK ABOUT THIS
    x = scaling_func( x₀ );
    @info "Starting at x₀ = $x with Δ₀= $Δ (scaled to unit hypercube [0,1]^n)."

    f_x_exp = f_expensive(x);
    f_x_cheap = f_cheap(x);
    n_objectives( y ) = ndims( y ) == 0 ? 1 : max( size( y )... );
    config_struct.n_exp = n_objectives( f_x_exp );    # set number of expensive objectives
    config_struct.n_cheap = n_objectives( f_x_cheap ); # set number of cheap objectives

    f_x = vcat( f_x_exp, f_x_cheap );

    sites_db = [x];    # "database" of all evaluation sites
    values_db = [f_x];

    # initialize surrogate model
    iter_data = IterData( x = x, f_x = f_x, Δ = Δ, sites_db = sites_db, values_db = values_db);   # make values available to subroutines
    @pack! config_struct = iter_data;
    @info("Initializing model.")
    rbf_model = build_model( config_struct, is_constrained )

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
                improve!(rbf_model, config_struct, is_constrained);
            else
                @info("\tUpdating model")
                rbf_model = build_model( config_struct, is_constrained ) # "update" model (actually constructs a new model instance)
            end
        end

        # compute descent step
        @info("\tComputing descent step.")
        @show f_x
        ω, d, stepsize = compute_descent_direction( Val(descent_method), rbf_model, f_cheap, x, f_x, Δ, is_constrained, all_objectives_descent)
        @info("\t\tCriticality measure ω is $ω.")

        # Criticallity Test
        num_critical_loops = 0;
        if ω <= ε_crit      # analog to small gradient
            @info("\tEntered criticallity test!")
            if !rbf_model.fully_linear
                n_improvements = make_linear!(rbf_model, config_struct, is_constrained);
                if n_improvements > 0
                    ω, d, stepsize = compute_descent_direction(Val(descent_method), rbf_model, f_cheap, x, f_x, Δ, is_constrained,  all_objectives_descent)
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
                    changed = make_linear!(rbf_model, config_struct, Val(true), is_constrained);
                    if changed
                        ω, d, stepsize = compute_descent_direction( Val(descent_method), rbf_model, f_cheap, x, f_x, Δ, is_constrained,  all_objectives_descent)
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

        f̃(x) = vcat( rbf_model.function_handle(x), f_cheap(x) );
        x₊ = x + d;
        m_x = f̃( x );    # for printing purposes only
        m_x₊ = f̃( x₊ );
        @info("\t\tm_x   = $m_x")
        @info("\t\tm_x_+ = $m_x₊")
        f_x₊ = f( x₊ )
        # acceptance test ratio
        if all_objectives_descent
            @show (f_x .- f_x₊) ./ (f_x .- m_x₊)
            ρ = minimum( (f_x .- f_x₊) ./ (f_x .- m_x₊)  )
        else
            max_f_x = maximum( f_x );
            ρ = ( max_f_x - maximum( f_x₊ ) ) / ( max_f_x - maximum( m_x₊ ) )
        end

        @info("\tρ is $ρ")

        # save values to database
        push!(sites_db, x₊)
        push!(values_db, f_x₊)

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
    @info("\n\n--------------------------")
    @info("Finished optimization after $iter_index iterations and $(length(sites_db)) evaluations.")
    @info("Final (unscaled) x  = $(unscaling_func(x)).")
    @info("Final fx = $f_x.")

    sites_db[:] = unscaling_func.( sites_db )
    return true #(unscaling_func(x) , f_x, unscaling_func.(sites_db), values_db, iterate_indices)
end

end
