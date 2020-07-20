using Morbit
using Plots: plot

# Standard Problem #####
lb = [-4.0, -4.0];
ub = [4.0, 4.0];
x_0 = lb .+ (ub .- lb ) .* rand(2);
x_0 = [π, 2.5];

f1(x) = (x[1]-1)^2 + (x[2] -1)^2
f2(x) = (x[1] + 1)^2 + (x[2]+1)^2 ;
∇f1(x) = [ 2*(x[1]-1); 2*(x[2]-1) ];
∇f2(x) = [ 2*(x[1]+1); 2*(x[2]+1) ];

opt_settings = AlgoConfig(
    rbf_kernel = :cubic,
    max_iter = 20,
    descent_method = :steepest,
    rbf_shape_parameter = Δ -> 1/Δ,
    all_objectives_descent = true,
    ν_success = 0.25,
    Δ₀ = .1,
);

problem_instance = MixedMOP( lb = lb, ub = ub )

add_objective!( problem_instance, f1, ∇f1)
add_objective!( problem_instance, f2, ∇f2)

optimize!(opt_settings, problem_instance, x_0 );

plot(
    plotdecisionspace(opt_settings),
    plotobjectivespace(opt_settings)
)
