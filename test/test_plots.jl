using Morbit

lb = [-4.0, -4.0];
ub = [4.0, 4.0];

f1(x) = (x[1]-1)^2 + (x[2]-1)^2 ;
f2(x) = (x[1]+1)^2 + (x[2]+1)^2 ;
x_0 = lb .+ (ub .- lb ) .* rand(2);
x_0 = [π, 2.5];

f(x) = [f1(x);f2(x)];

problem_instance = HeterogenousMOP( f_expensive = f1, f_cheap = f2, lb = lb, ub = ub)


opt_settings = AlgoConfig(
    rbf_kernel = "cubic",
    max_iter = 20,
    rbf_shape_parameter = Δ -> 1/Δ,
    all_objectives_descent = true
);

optimize!(opt_settings, problem_instance, x_0 );

using Plots

pset = range(-1, 1, length=30); pset = [ pset, pset ];
pfront = [ map( f1, eachrow( hcat(pset...) )  ), map( f2, eachrow( hcat(pset...))) ]
plot(
    plotdecisionspace(opt_settings, pset),
    plotobjectivespace(opt_settings, pfront)
)
