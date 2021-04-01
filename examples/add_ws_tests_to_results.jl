# Retrospectively add a comparison of with WS method 
# to a table of test runs

using Pkg;
Pkg.activate( @__DIR__ );

using DataFrames;
using FileIO, JLD2
using Dates
import NLopt
using MultiObjectiveProblems
using Dates

#%% load helpers 
include("benchmark_helpers/setting_helpers.jl");
include("plot_helpers/loading_saving.jl")

#%% load results
res_file = joinpath(
    ENV["HOME"], "MORBIT_BENCHMARKS", 
    #"results_XIHJrRkp_4480x8_30_Mar_2021__02_22_26.jld2"
    "results_32sZVhKT_6720x8_31_Mar_2021__23_47_11.jld2"
);
prev_res = load_results( res_file )

prev_problems = unique( prev_res.problem_str )
prev_n_vars = unique( prev_res.n_vars )

features = Dict(
    "problem_str" => prev_problems,
    "n_vars" => prev_n_vars
)

function _get_x0(; problem_str, n_vars )
    global prev_res;
    return unique( 
        prev_res[ 
            (prev_res.problem_str .== problem_str) .&
            (prev_res.n_vars .== n_vars ), :x0]
    )
end

dependent_features = [
    Dict(
        "name" => "x0",
        "depends_on" => ["n_vars", "problem_str",],
        "values" => _get_x0,
    ),
]

# create a table with all possible settings as rows
results = generate_all_settings( features, dependent_features );

# finally add observation columns by providing their names and types
observations = (; :n_evals => Int, :x => Vector{Float64}, :ω => Float64  );
add_observation_columns!(results, observations)

#%%

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

function perform_nlopt_run(; kwargs...)
    x0 = kwargs[:x0]
    n_vars = kwargs[:n_vars];

    test_prob = get_test_problem(;kwargs...)
    objfs = get_objectives(test_prob);
    box_constraints = constraints(test_prob);
    LB = box_constraints.lb; UB = box_constraints.ub;
    ω_func = get_omega_function( test_prob );

    opt = NLopt.Opt( :LN_COBYLA, length(x0) )
    opt.lower_bounds = LB;
    opt.upper_bounds = UB;
    opt.maxeval = 1000*n_vars;
    opt.xtol_rel = 1e-2;
    opt.ftol_rel = 1e-2;
    opt.min_objective = function( x, g )
        return sum(f(x) for f ∈ objfs)
    end
    _, x, _ = NLopt.optimize( opt, x0 )

    ω = ω_func(x);

    return (; :n_evals => opt.numevals, :x => x, :ω => ω);
end

# %% start the fun

# obtain functions to save results dataframe 

save_lock = ReentrantLock()     # optional, for saving when MultiThreading 
save_counter = 0;
# RUN the actual tests; note the `@view` which is important to actually
# be able to store the results;

results_lock = ReentrantLock();
Threads.@threads for result_table_row in eachrow( @view(results[:,:]) )
    global save_counter;
    println("$(Dates.format(now(), "HH:MM:SS:sss")) : $(result_table_row)")
    try
        row_results = perform_nlopt_run(; result_table_row... )

        lock(results_lock) do
            for observation in keys(observations)
                result_table_row[ observation ] = row_results[ observation ]
            end
        end

        # Option: save results every T-th iteration
        #=
        lock( save_lock ) do 
            save_counter += 1;
            if save_counter % args["save-every"] == 1
                save_func( results; save_csv = false )
            end
        end
        =#
    catch e
        @error "Error for \n$(result_table_row)." exception=(e, catch_backtrace())
    end
end
#%%
res_out = string(splitext(res_file)[1], "_WS", ".jld2")
save(res_out, Dict("results" => results))