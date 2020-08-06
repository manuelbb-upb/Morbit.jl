using Morbit
using Test

@testset "1In1OutProblem_SteepestDescent" begin

    x0 = [5.0];
    f1(x) = x[1]^2;

    # I) unconstrained
    opt_settings = AlgoConfig(
        max_iter = 10
    )
    ## 1) treat objective as cheap
    mop = MixedMOP();
    add_objective!( mop, f1, :cheap )
    x,fx = optimize!(opt_settings, mop, x0)

    @test x ≈ [ 0.0 ]

    ## 2) treat objective as expensive
    mop = MixedMOP();
    add_objective!( mop, f1, :expensive )
    x,fx = optimize!(opt_settings, mop, x0)

    @test x ≈ [ 0.0 ] atol = 0.01

    #II) constrained

    ## 1) treat objective as cheap
    opt_settings = AlgoConfig(
        max_iter = 10
    )
    mop = MixedMOP( lb = [-8.0], ub = [8.0] );
    add_objective!( mop, f1, :cheap)
    x,fx = optimize!(opt_settings, mop, x0)

    @test x ≈ [ 0.0 ] atol=0.5

    ## 2) treat objective as expensive
    opt_settings = AlgoConfig(
        max_iter = 20
    )
    mop = MixedMOP( lb = [-8.0], ub = [8.0] );
    add_objective!( mop, f1, :expensive)
    x,fx = optimize!(opt_settings, mop, x0)

    @test x ≈ [ 0.0 ] atol=1
end


@testset "2In2OutProblem_Steepest_Descent" begin
    x0 = [ π , π^2 ]
    ub = 10 .* ones(2);
    lb = -ub;

    g1(x) = sum( (x .- 1.0).^2 );
    g2(x) = sum( (x .+ 1.0).^2 );

    # unconstrained, cheap
    opt_settings = AlgoConfig(
        max_iter = 20
    )
    mop = MixedMOP();
    add_objective!(mop, g1, :cheap)
    add_objective!(mop, g2, :cheap)
    x,fx = optimize!( opt_settings, mop, x0 )

    @test x[1] ≈ x[2] atol = .01

    # unconstrained, expensive
    opt_settings = AlgoConfig(
        max_iter = 20
    )
    mop = MixedMOP();
    add_objective!(mop, g1, :expensive)
    add_objective!(mop, g2, :expensive)
    x,fx = optimize!( opt_settings, mop, x0 )

    @test x[1] ≈ x[2] atol = .01

    # unconstrained, heterogenous
    opt_settings = AlgoConfig(
        max_iter = 20
    )
    mop = MixedMOP();
    add_objective!(mop, g1, :expensive)
    add_objective!(mop, g2, :cheap)
    x,fx = optimize!( opt_settings, mop, x0 )

    @test x[1] ≈ x[2] atol = .05

    # constrained, cheap
    opt_settings = AlgoConfig(
        Δ₀ = 0.1,
        max_iter = 50
    )
    mop = MixedMOP(lb = lb, ub = ub);
    add_objective!(mop, g1, :cheap)
    add_objective!(mop, g2, :cheap)
    x,fx = optimize!( opt_settings, mop, x0 )

    @test x[1] ≈ x[2] atol = .01

    # constrained, expensive
    opt_settings = AlgoConfig(
        Δ₀ = 0.2,
        max_iter = 150,
        rbf_kernel = :exp,
        rbf_shape_parameter = Δ -> 10/Δ,
        θ_pivot = 1e-3,
        Δ_min = 1e-12,
        Δ_critical = 1e-10,
        stepsize_min = 1e-14,
        all_objectives_descent = true,
        max_model_points = 10,
        sampling_algorithm = :monte_carlo
    )
    mop = MixedMOP(lb = lb, ub = ub);
    add_objective!(mop, g1, :expensive)
    add_objective!(mop, g2, :expensive)
    x,fx = optimize!( opt_settings, mop, x0 )

    @test x[1] ≈ x[2] atol = .1

    # constrained, heterogenous
    opt_settings = AlgoConfig(
        Δ₀ = 0.2,
        max_iter = 150,
        rbf_kernel = :thin_plate_spline,
        sampling_algorithm = :monte_carlo,
        Δ_min = 1e-8,
        all_objectives_descent = true
    )
    mop = MixedMOP(lb = lb, ub = ub);
    add_objective!(mop, g1, :expensive)
    add_objective!(mop, g2, :cheap)
    x,fx = optimize!( opt_settings, mop, x0 )

    @test x[1] ≈ x[2] atol = .05
end

using Plots

# true data for comparison
f(x) = [g1(x);g2(x)];
points_x = collect(-1:0.05:1);
pset = ParetoSet( points_x, points_x )
pfront = ParetoFrontier(f, pset);

plot(
    plot_decision_space(opt_settings, pset),
    plot_objective_space(opt_settings, pfront),
    plotstepsizes(opt_settings),
    plotfunctionvalues(opt_settings),
)
