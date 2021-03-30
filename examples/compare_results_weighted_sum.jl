using Pkg;
Pkg.activate( @__DIR__ );

using AbstractPlotting, CairoMakie
using AlgebraOfGraphics
using Statistics
using Colors;
using DataFrames;
using FileIO, JLD2

using MultiObjectiveProblems
import NLopt

# disable inline plotting
#AbstractPlotting.inline!(true);

#%% load helpers 
include("plot_helpers/upb_colors.jl");
include("plot_helpers/loading_saving.jl")

#%% settings 
res_file = joinpath(
    ENV["HOME"], "MORBIT_BENCHMARKS", 
    "results_WoJEMTIw_3600x8_29_Mar_2021__18_13_49.jld2"
);

results = load_results( res_file )

#%%
n_vars = 15;
problem = ZDT2
test_prob = problem(n_vars);
method = "ps"
model = "TP1"

box_constraints = constraints(test_prob);
f1, f2 = get_objectives(test_prob);
LB = box_constraints.lb; UB = box_constraints.ub;

X0 = unique(results[ results.n_vars .== n_vars ,:x0])
max_evals = Int( ceil( mean(
    results[ 
        (results.n_vars .== n_vars) .& (results.problem_str .== string(problem)) .& 
        (results.model .== model) .& (results.method .== method ),
        :n_evals
    ])))
    
Xmorbit = unique(results[ 
    (results.n_vars .== n_vars) .& (results.model .== model) .& 
    (results.method .== method) .& (results.problem_str .== string(problem)) ,:x]
)
#%%
Xf = []
for x₀ ∈ X0
    @show x₀
    opt = NLopt.Opt( :LN_COBYLA, n_vars )
    opt.lower_bounds = LB;
    opt.upper_bounds = UB;
    opt.maxeval = max_evals;
    opt.min_objective = function( x, g )
        return f1(x) + f2(x)
    end
    _, xf, _ = NLopt.optimize( opt, x₀ )
    @show opt.numevals
    push!(Xf, xf)
end

#%%

pf = get_pareto_front( test_prob )
pfp = get_points( pf, 200 );
fig, ax, _ = lines(Tuple.(pfp))

F0 = [ (f1(x), f2(x)) for x ∈ X0];
Fmorbit = [ (f1(x), f2(x)) for x ∈ Xmorbit];
Ff = [ (f1(x), f2(x)) for x ∈ Xf];

scatter!(Tuple.(F0); color = :green )
scatter!(Tuple.(Ff))
scatter!( Tuple.(Fmorbit); color = :red, marker = :diamond )
fig
