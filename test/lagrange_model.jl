using Morbit
using Test

LB = [-2.0, -2.0];
UB = [2.0, 2.0];
x0 = LB .+ (UB .- LB ) .* rand(2);
g1(x) = sum( (x .- 1.0).^2 );
g2(x) = sum( (x .+ 1.0).^2 );
mop = MixedMOP(lb = LB, ub = UB);

# test optimized sampling
mop1 = deepcopy(mop);
add_objective!(mop1, g1, :cheap);
add_objective!(mop1, g2, LagrangeConfig(Λ=10, optimized_sampling = true, degree=2));
opt_settings = AlgoConfig(
    Δ₀ = 0.2,
    max_iter = 10,
);
   
x,fx = optimize!( opt_settings, mop1, x0 );
@test x[1] ≈ x[2] atol=1e-2

# test unoptimized samling
mop1 = deepcopy(mop);
add_objective!(mop1, g1, :cheap);
add_objective!(mop1, g2, LagrangeConfig(Λ=10, optimized_sampling = false, degree=2));
opt_settings = AlgoConfig(
    Δ₀ = 0.2,
    max_iter = 10,
);
   
x,fx = optimize!( opt_settings, mop1, x0 );
@test x[1] ≈ x[2] atol=1e-2


# test optimized samling with possibly non fully linear models
mop1 = deepcopy(mop);
add_objective!(mop1, g1, :cheap);
add_objective!(mop1, g2, LagrangeConfig(Λ=10, optimized_sampling = true, allow_not_linear = true, degree=2));
opt_settings = AlgoConfig(
    Δ₀ = 0.1,
    Δ_max = .3,
    max_iter = 15,
);
   
x,fx = optimize!( opt_settings, mop1, x0 );
@test x[1] ≈ x[2] atol=1e-2