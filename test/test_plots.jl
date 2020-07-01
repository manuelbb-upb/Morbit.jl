using Morbit

lb = [-4.0, -4.0];
ub = [4.0, 4.0];
x_0 = lb .+ (ub .- lb ) .* rand(2);
x_0 = [π, 2.5];


problem_instance = MixedMOP( lb = lb, ub = ub )

f1(x) = (x[1]-1)^2 + (x[2]-1)^2 ;
f2(x) = (x[1]+1)^2 + (x[2]+1)^2 ;

add_objective!( problem_instance, f1, :expensive)
add_objective!( problem_instance, f2, :cheap )

problem_instance = HeterogenousMOP( f_expensive = f1, f_cheap = f2, lb = lb, ub = ub)


opt_settings = AlgoConfig(
    rbf_kernel = "multiquadric",
    max_iter = 10,
    rbf_shape_parameter = Δ -> 10/Δ,
    all_objectives_descent = true
);

optimize!(opt_settings, problem_instance, x_0 );

using Plots
gr()

f(x) = [f1(x);f2(x)];

points_x = collect(-1:0.05:1);
pset = ParetoSet( points_x, points_x )
pfront = ParetoFrontier(f, pset);
plot(
    plot_decision_space(opt_settings, pset),
    plot_objective_space(opt_settings, pfront)
)
