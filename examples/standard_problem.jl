using Morbit

# Standard Problem #
# Two Parabolas

lb = -3 .* ones(2)
ub = 3 .* ones(2)

x_0 = lb .+ (ub .- lb ) .* rand(2);

# this x_0 provokes funny behavior in :direct_search
 x_0 = [1.9893381732081306, 2.3364546273987514]
# :cubic and :orthogonal does not work
# :cubic and :monte_carlo does
# Using a TaylorModel (degree = 1, :autodiff) produces a beautiful curve

f1(x) = (x[1] - 1)^2 + (x[2] - 1)^2;
f2(x) = (x[1] + 1)^2 + (x[2] + 1)^2;

opt_settings = AlgoConfig(
    ε_crit = 1e-12,
    max_iter = 11,
    Δ₀ = .1,
    Δ_max = 0.1,
    all_objectives_descent = true,
    descent_method = :steepest,
    ideal_point = [0,0]
);

problem_instance = MixedMOP(lb = lb, ub = ub);

rbf_conf = RbfConfig(
    kernel = :multiquadric,
    shape_parameter = 1,
    sampling_algorithm = :orthogonal
)

taylor_conf = TaylorConfig(
    gradients = :autodiff,
    #hessians = :fdm,
    degree = 2
)

lagrange_conf = LagrangeConfig(
    degree = 2,
    Λ = 2.0
)

add_objective!(problem_instance, f1, :cheap)
add_objective!(problem_instance, f2, lagrange_conf)

optimize!(opt_settings, problem_instance, x_0);

# Uncomment below to plot
#=
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
=#
