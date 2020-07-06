using Morbit

# Standard Problem #####
#=
lb = [-4.0, -4.0];
ub = [4.0, 4.0];
x_0 = lb .+ (ub .- lb ) .* rand(2);
x_0 = [π, 2.5];

f1(x) = (x[1]-1)^2 + (x[2]-1)^2 ;
f2(x) = (x[1]+1)^2 + (x[2]+1)^2 ;

# true data for comparison
f(x) = [f1(x);f2(x)];
points_x = collect(-1:0.05:1);
pset = ParetoSet( points_x, points_x )
pfront = ParetoFrontier(f, pset);
=#
#####

# ZDT 3
n_vars = 30;
lb = zeros(n_vars)
ub = ones(n_vars);

h(f1, g) = 1.0 - sqrt( f1/g ) - (f1/g)*sin(10*pi*f1)
g(x) = 1 + 9/(n_vars-1) * sum( x[2:end] )

f1(x) = x[1]
f2(x) = g(x) * h( f1(x), g(x) )

x_0 = rand(n_vars);
x_0 = ones(n_vars);

# pareto data for comparison
pset = nothing
pfront = ParetoFrontier(n_objfs = 2, objective_arrays = [[],[]])
# pareto frontier
F = [ [1e-10, 0.0] .+ v for v ∈ [
    [0, 0.0830015349],
    [0.1822287280, 0.2577623634],
    [0.4093136748, 0.4538821041],
    [0.6183967944, 0.6525117038],
    [0.8233317983, 0.8518328654]
]]
for v ∈ F
    y1 = range( v[1], v[2]; length = 20 );
    y2 = 1 .- sqrt.(y1) .- y1 .* sin.( 10*pi .* y1)
    push!( pfront.objective_arrays[1], y1... )
    push!( pfront.objective_arrays[2], y2... )
end

problem_instance = MixedMOP( lb = lb, ub = ub )

add_objective!( problem_instance, f1, :cheap)
add_objective!( problem_instance, f2, :expensive )

#=
opt_settings = AlgoConfig(
    rbf_kernel = :multiquadric,
    max_iter = 15,
    max_model_points = 400,
    max_critical_loops = 5,
    Δ_max = 0.4,
    Δ₀ = 0.4,
    θ_enlarge_1 = 2,
    θ_enlarge_2 = 4,
    rbf_shape_parameter = Δ -> min(10/Δ, 1e8),
    all_objectives_descent = true,
    Δ_critical = 0.0
);
=#

opt_settings = AlgoConfig(
    descent_method = :direct_search,
    rbf_kernel = :multiquadric,
    rbf_shape_parameter = Δ -> 10/Δ,
    ε_crit = 1e-15,
    max_evals = 150,
    #max_model_points = 150,
    ν_accept = 0.0,
    ν_success = 0.2,
    Δ₀ = .1,
    Δ_max = 1,
    θ_enlarge_1 = 10,
    θ_enlarge_2 = 3,
    all_objectives_descent = false,
    γ_shrink = 1,
    γ_crit = 0.85,
    γ_shrink_much = 0.75,
    #ideal_point = [0;0]
)

for i = 1
    global x_0, problem_instance, opt_settings

    opt_settings = AlgoConfig(
        descent_method = :direct_search,
        rbf_kernel = :multiquadric,
        rbf_shape_parameter = Δ -> 10;
        ε_crit = 1e-15,
        max_evals = 50,
        #max_model_points = 150,
        ν_accept = 0.0,
        ν_success = 0.2,
        Δ₀ = .1,
        Δ_max = 1,
        θ_enlarge_1 = 10,
        θ_enlarge_2 = 3,
        all_objectives_descent = false,
        γ_shrink = 1,
        γ_crit = 0.85,
        γ_shrink_much = 0.75,
        #ideal_point = [0;0]
    )

    optimize!(opt_settings, problem_instance, x_0 );
    x_0 = opt_settings.iter_data.x
end

using Plots

plot(
    plot_decision_space(opt_settings, pset),
    plot_objective_space(opt_settings, pfront),
    plotstepsizes(opt_settings)
    )
