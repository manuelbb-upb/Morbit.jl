using Pkg;
Pkg.activate( @__DIR__ );

using AbstractPlotting, CairoMakie
using AlgebraOfGraphics
using Statistics
using Colors;
using DataFrames;
using FileIO, JLD2

using MultiObjectiveProblems
using Morbit 
import NLopt

include("plot_helpers/loading_saving.jl")

#%% Settings

n_vars = 10;
num_runs = 10;
problem = ZDT2
max_evals = 100

# retrieve test problem
test_prob = problem(n_vars);
f1, f2 = get_objectives(test_prob);
box_constraints = constraints(test_prob);
LB = box_constraints.lb; UB = box_constraints.ub;

# steepest descent config
ac_sd = AlgoConfig(
    max_iter = 100,
    descent_method = :steepest_descent,
)

# generate starting points 
X0 = Vector.(HaltonPoint(n_vars; length = num_runs));

#%% setup test functions
function _morbit_run( ac, x0 )
    global test_prob, LB, UB, f1, f2, max_evals;

    cfg = RbfConfig(
        kernel = :cubic,
        max_evals = max_evals,
    );
    mop = MixedMOP( LB, UB )
    add_objective!(mop, f1, cfg )
    add_objective!(mop, f2, cfg )

    x, fx, _ = optimize(mop, x0; algo_config = ac )
    return x, fx 
end

function _nlopt_run( x0 )
    global LB, UB, f1, f2, max_evals;
    opt = NLopt.Opt( :LN_COBYLA, length(x0) )
    opt.lower_bounds = LB;
    opt.upper_bounds = UB;
    opt.maxeval = max_evals;
    opt.min_objective = function( x, g )
        return f1(x) + f2(x)
    end
    _, x, _ = NLopt.optimize( opt, x0 )
    return x, [f1(x); f2(x)]
end

#%% First Morbit run (steepest_descent)
X_sd = []
FX_sd = []
for x0 ∈ X0
    res = _morbit_run(ac_sd, x0) 
    push!( X_sd, res[1])
    push!( FX_sd, res[2])
end

#%% NLopt run 
FX_nlopt = []
for x0 ∈ X0
    res = _nlopt_run(x0)
    push!(FX_nlopt, res[2])
end

#%%
FX_ps = []
for (i,x0) ∈ enumerate(X0)
    dir = [0 ; 5 * i];
    # setup Morbit configuration
    ac_ps = AlgoConfig(
        max_iter = 100,
        descent_method = :ps,
        reference_point = get_ideal_point( test_prob ) .- dir
    );
    res = _morbit_run(ac_ps, x0) 
    push!( FX_ps, res[2])
end

#%% Plotting 

# First plot the Pareto Front 
pf = get_pareto_front( test_prob )
pfp = get_points( pf, 200 );
fig, ax, _ = lines(Tuple.(pfp); linewidth = 1.5, 
    label = "Front",
    figure = (resolution = (500, 450),),
) 

F0 = [[f1(x);f2(x)] for x ∈ X0]
# add results from Morbit
scatter!(Tuple.(F0); color = upb_blue, marker =:circle, label = "f(x₀)" )
scatter!(Tuple.(FX_sd); color = upb_lightblue, marker = :diamond, label = "SD")
scatter!(Tuple.(FX_ps); color = upb_cyan, marker = :cross, label = "PS")
scatter!(Tuple.(FX_nlopt); color = upb_orange, marker = :rect, label = "WS")

ax.title[] = "Solution distribution ($(string(problem)), $(n_vars) variables)"
ax.xlabel[] = "f₁"
ax.ylabel[] = "f₂"
leg = fig[1, 2] = Legend(fig[1,1], ax;
    framevisible = false, labelsize = 15 )
fig

plot_file = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
 "compare_ws_morbit_$(string(problem))_$(n_vars).png"
)

saveplot(plot_file, fig)