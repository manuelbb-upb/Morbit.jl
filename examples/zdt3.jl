using Morbit

# ZDT 3

n_vars = 30;
lb = zeros(n_vars)
ub = ones(n_vars);

h(f1, g) = 1.0 - sqrt(f1 / g) - (f1 / g) * sin(10 * pi * f1)
g(x) = 1 + 9 / (n_vars - 1) * sum(x[2:end])

f1(x) = x[1]
f2(x) = g(x) * h(f1(x), g(x))

x_0 = rand(n_vars);
#x_0 = ones(n_vars);

# pareto data for comparison
pset = nothing
pfront = ParetoFrontier(n_objfs = 2, objective_arrays = [[], []])
# pareto frontier
F = [
    [1e-10, 0.0] .+ v
    for
    v ∈ [
        [0, 0.0830015349],
        [0.1822287280, 0.2577623634],
        [0.4093136748, 0.4538821041],
        [0.6183967944, 0.6525117038],
        [0.8233317983, 0.8518328654],
    ]
]
for v ∈ F
    y1 = range(v[1], v[2]; length = 20)
    y2 = 1 .- sqrt.(y1) .- y1 .* sin.(10 * pi .* y1)
    push!(pfront.objective_arrays[1], y1...)
    push!(pfront.objective_arrays[2], y2...)
end

opt_settings = AlgoConfig(
    max_iter = 10,
    ε_crit = 1e-9,
    Δ₀ = .2,
    θ_enlarge_1 = 4.0,
    Δ_max = 0.5,
    θ_enlarge_2 = 10.0,
    max_critical_loops = 2,
    descent_method = :direct_search,
    rbf_kernel = :multiquadric,
    rbf_shape_parameter = cs -> 1 / (cs.iter_data.Δ),
    sampling_algorithm = :orthogonal
)

problem_instance = MixedMOP(lb = lb, ub = ub)

add_objective!(problem_instance, f1, :cheap)
add_objective!(problem_instance, f2, :expensive)

optimize!(opt_settings, problem_instance, x_0);

using Plots

plot(
    plot_decision_space(opt_settings, pset),
    plot_objective_space(opt_settings, pfront),
    plotstepsizes(opt_settings),
    plotfunctionvalues(opt_settings),
)
