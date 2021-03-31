# Activate benchmark environment to have all dependencies

using Pkg;
Pkg.activate(@__DIR__)

# Parse command line arguments
# 
# args will have fields 
# `outdir`; default = joinpath( ENV["HOME"], "MORBIT_BENCHMARKS" )
# `filename`; default = ""
# `runs-per-setting`; default = Threads.nthreads()
# `resume-from`; default = ""
include( "benchmark_helpers/utils.jl")
args = parse_commandline();

# Specific to our use case:
using Morbit;                   # our algorithm we test
using MultiObjectiveProblems;   # test problems (zdt, dtlz etc.) 
using HaltonSequences           # for deterministic starting points

# Optional : Set logging level
using Logging;
logger = ConsoleLogger(stdout, Logging.Warn)
global_logger(logger);

using Dates        # for printing below

# Or logging to file; `close(io)` when done!
#=
using Logging;
using Dates;
io = open( 
    joinpath(args["outdir"], 
    string( Dates.format(now(), "d_u_Y__H_M_S"), "_log.txt") ),
    "w+"
);
logger = ConsoleLogger(io, Logging.Warn); 
global_logger(logger);
=#

# Include helpers for DataFrames
include("benchmark_helpers/saving.jl")
include("benchmark_helpers/setting_helpers.jl");

# Preparations Done!!!

#%% ---------------------------------------------------------------------------- #

# Define "features" and their values to test for

# Small example where we simply test the performance of two model types for two problems

features = Dict(
    "method" => [:steepest_descent, :ps],
    "model" => ["cubic", "TP1", "LP1", "LP2"],
    "problem_str" => ["ZDT1", "ZDT2", "ZDT3", "DTLZ1", "DTLZ6"],
    "n_vars" => collect(2:15),
);
num_runs = args["runs-per-setting"];

# we perform `num_runs` runs for each possible combination of 
# feature values. Each run should have a different starting point "x0", but the problems
# could have different box constraints. Hence the feature "x0" is **dependent** on the 
# feature "problems"

# 1) define little helper to get start points from problem string
# note that its keyword arguments correspond to the feature(s)
# it depends on!
function get_test_problem(; kwargs... )
    n_vars = kwargs[:n_vars];
    prob_symb = Symbol(kwargs[:problem_str]);
    prob_type = @eval $prob_symb;
    if prob_type <: ZDT
        test_problem = prob_type( n_vars )
    elseif prob_type <: DTLZ
        k = min( n_vars - 1, 5 );
        test_problem = prob_type( 
            n_vars,
            n_vars - k + 1,
            k
        )
    end
    return test_problem
end

function x0_values_function(; kwargs... )
    n_vars = kwargs[:n_vars];
    test_problem = get_test_problem(; kwargs... );
    box = constraints( test_problem );
    return [ box.lb .+ ( box.ub .- box.lb ) .* site for site in 
        HaltonPoint( n_vars; length = num_runs )
    ];
end

# 2) define the actual dependent feature using a Dict and put it in a list
dependent_features = [
    Dict(
        "name" => "x0",
        "depends_on" => ["n_vars", "problem_str",],
        "values" => x0_values_function,
    ),
]

# create a table with all possible settings as rows
results = generate_all_settings( features, dependent_features );

# finally add observation columns by providing their names and types
observations = (; :n_evals => Int, :x => Vector{Float64}, :ω => Float64  );
add_observation_columns!(results, observations)

sort!(results, ["n_vars", "method", "model"]);

missing_rows = collect( 1 : size(results,1) );
# Optional: Resume previous simulation
if length( args["resume-from"] ) > 0 && isfile( args["resume-from"] )
    old_results = load_previous_results( args["resume-from"] );
    _, missing_rows = fill_from_partial_results!( results, old_results, features, dependent_features, observations );
end

println("All in all we have to perform $(length(missing_rows)) of $(size(results,1)) tests.")
#%% ---------------------------------------------------------------------------- #

# Now setup the actual tests.
# To keep things clean we use several getter functions that take 
# as keyword arguments the features defined above.

# reading the large data files in parallel does often not work
# hence the lock
lagrange_lock = ReentrantLock();

function get_model_config(; kwargs...)
    global lagrange_lock;
    model_str = kwargs[:model];
    n_vars = kwargs[:n_vars];

    if model_str == "cubic"
        cfg = RbfConfig(
            kernel = :cubic,
            shape_parameter = 1.0,
            max_model_points = n_vars <= 10 ? (n_vars +1 )*(n_vars + 2)/2 : 2*n_vars + 1,
            θ_enlarge_1 = 2.0,
            θ_pivot = 1/4
        )
    elseif model_str == "TP1"
        cfg = TaylorConfig(
            degree = 1,
            gradients = :fdm
        )
    elseif model_str == "LP1"
        cfg = LagrangeConfig(
            degree = 1,
            optimized_sampling = true, 
            Λ = 1.5,
        )
    elseif model_str == "LP2"
        cfg = LagrangeConfig(
            degree = 2,
            optimized_sampling = (n_vars <= 5),
            Λ = 1.5,
            save_path = joinpath(ENV["HOME"], "LagrangeSites", "n_vars_$(n_vars)_deg_2_Λ_15_.jld2"),
            io_lock = lagrange_lock,
        )
    end
    cfg.max_evals = n_vars * 1000;
    return cfg;
end

function get_mop(test_problem; kwargs...)
    model_cfg = get_model_config(; kwargs... );

    box = constraints( test_problem );
    objectives = get_objectives( test_problem );
    
    mop = MixedMOP( box.lb, box.ub );
    add_objective!( mop, objectives[1], ExactConfig(gradients=:fdm) );
    for objf in objectives[2:end]
        add_objective!( mop, objf, model_cfg );
    end
    return mop;
end

function get_algo_config( test_problem; kwargs...)
    model = kwargs[:model]
    n_vars = kwargs[:n_vars]
    
    AlgoConfig(;
        max_critical_loops = model == "TP1" ? 0 : 3,
        ε_crit = .01,
        max_iter = 100,
        Δ_0 = .1,
        Δ_max = .5,
        x_tol_rel = 1e-3,
        f_tol_rel = 1e-3,
        descent_method = kwargs[:method],
        strict_acceptance_test = true,
        strict_backtracking = true,
        # some pascoletti_serafini settings
        reference_point = get_ideal_point( test_problem ) .- 1,
        ps_algo = :GN_ISRES,
        max_ps_problem_evals = 50 * (n_vars + 1),
        max_ps_polish_evals = 100 * (n_vars + 1),
        ps_polish_algo = :LD_MMA,
    )
end

# here we take arbitrary keyword arguments so that we can pass 
# whole rows from the result dataframe
function perform_test(; kwargs... )
    model = kwargs[:model];
    problem_str = kwargs[:problem_str];
    x0 = kwargs[:x0];
    
    test_problem = get_test_problem(;kwargs...);
    ω_func = get_omega_function( test_problem );

    ac = get_algo_config(test_problem ;kwargs...);

    mop = get_mop(test_problem; kwargs... );
    
    x, _ = optimize( mop, x0; algo_config = ac)
    ω = ω_func(x);

    n_evals = maximum( o.n_evals for o ∈ mop.vector_of_objectives[2:end] );
    
    return (; :n_evals => n_evals, :x => x, :ω => ω);
end

# obtain functions to save results dataframe 
save_func = get_save_function(args; caller_path = @__FILE__);
save_lock = ReentrantLock()     # optional, for saving when MultiThreading 
save_counter = 0;
# RUN the actual tests; note the `@view` which is important to actually
# be able to store the results;

# %% start the fun
results_lock = ReentrantLock();
Threads.@threads for result_table_row in eachrow( @view(results[ missing_rows, : ]) )
    global save_counter;
    println("$(Dates.format(now(), "HH:MM:SS:sss")) : $(result_table_row)")
    try
        row_results = perform_test(; result_table_row... )

        lock(results_lock) do
            for observation in keys(observations)
                result_table_row[ observation ] = row_results[ observation ]
            end
        end

        # Option: save results every T-th iteration
        lock( save_lock ) do 
            save_counter += 1;
            if save_counter % args["save-every"] == 1
                save_func( results; save_csv = false )
            end
        end
    catch e
        @error "Error for \n$(result_table_row)." exception=(e, catch_backtrace())
    end
end

#%% ---------------------------------------------------------------------------- #
# Save final results
save_func(results;save_csv = false);