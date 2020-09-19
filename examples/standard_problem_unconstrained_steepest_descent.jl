using Morbit

# Standard Problem #
# no boundaries
# multiobjective steepest descent

lb = -3 .* ones(2)
ub = 3 .* ones(2)

x_0 = lb .+ (ub .- lb ) .* rand(2);

f1(x) = (x[1] - 1)^2 + (x[2] - 1)^2;
f2(x) = (x[1] + 1)^2 + (x[2] + 1)^2;

opt_settings = AlgoConfig(
    #max_iter = typemax(Int64),
    max_iter = 10,
    Δ₀ = .1,
    max_critical_loops = 10,
    ε_crit = 0.0000001,
    all_objectives_descent = false,
    sampling_algorithm = :monte_carlo,
    descent_method = :steepest,
    #ideal_point = [0,0]
);    # use default settings

problem_instance = MixedMOP()# lb = lb, ub = ub);

add_objective!(problem_instance, f1, :cheap)
add_objective!(problem_instance, f2, :expensive)

optimize!(opt_settings, problem_instance, x_0);

# Uncomment below to plot

using Plots

# true data for comparison
f(x) = [f1(x); f2(x)];
points_x = collect(-1:0.05:1);
pset = ParetoSet(points_x, points_x)
pfront = ParetoFrontier(f, pset);

plot(
    plot_decision_space(opt_settings, pset),
    plot_objective_space(opt_settings, pfront),
    plotstepsizes(opt_settings),
    plotfunctionvalues(opt_settings),
)
