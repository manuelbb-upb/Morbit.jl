using Morbit
#%%
n_vars = 10;

lb = zeros(n_vars)
ub = ones(n_vars);

h(f1, g) = 1.0 - sqrt(f1 / g)
g(x) = 1 + 9 / (n_vars - 1) * sum(x[2:end])

f1(x) = x[1]
f2(x) = g(x) * h(f1(x), g(x))

x_0 = rand(n_vars);
#%%

opt_settings = AlgoConfig(
    max_iter = n_vars * 5,
    ε_crit = 1e-5,
    Δ₀ = .1,
    descent_method = :steepest,
    true_ω_stop = 1e-2,
)

rbf_conf = RbfConfig(
#    max_evals = 5 * n_vars ,
    kernel = :cubic,
    #θ_enlarge_2 = 0.0,
    θ_enlarge_1 = 2.0,
    θ_pivot = 1/8,
    sampling_algorithm = :orthogonal
)

lagrange_conf = LagrangeConfig(
    #max_evals = 10 * n_vars,
    degree = 1,
    Λ = 100,
)

problem_instance = MixedMOP(lb = lb, ub = ub)

add_objective!(problem_instance, f1, :cheap)
add_objective!(problem_instance, f2, lagrange_conf)

optimize!(opt_settings, problem_instance, x_0);

#%%
# pareto data for comparison
pset = nothing
X = range(0,1;length=300);
pfront = ParetoFrontier(n_objfs = 2, objective_arrays = [
    X, 
    h.(X,1)    
    ]
);

#%%
using Plots

plot(
    plot_decision_space(opt_settings, pset),
    plot_objective_space(opt_settings, pfront),
    plotstepsizes(opt_settings),
    plotfunctionvalues(opt_settings),
)
