using Morbit
using Test

# Two standard (easy) test problems with known Pareto Data
# Many combinations of constaints, heterogenity, descent method and sampling algorithm are used

#using Logging
#global_logger(SimpleLogger(stdout, Logging.Debug))
@testset "1In1OutProblem_SteepestDescent" begin

    x0 = [5.0];
    f1(x) = x[1]^2;

    # I) unconstrained
    opt_settings = AlgoConfig(
        max_iter = 15
    )
    ## 1) treat objective as cheap
    mop = MixedMOP();
    add_objective!( mop, f1, :cheap )
    x,fx = optimize!(opt_settings, mop, x0)

    @test x ≈ [ 0.0 ] atol=1e-2

    ## 2) treat objective as expensive
    opt_settings = AlgoConfig(
        max_iter = 15
    )
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
    @testset "unbound_cheap_cheap" begin
        opt_settings = AlgoConfig(
            max_iter = 30
        )
        mop = MixedMOP();
        add_objective!(mop, g1, :cheap)
        add_objective!(mop, g2, :cheap)
        x,fx = optimize!( opt_settings, mop, x0 )

        @test x[1] ≈ x[2] atol = .01
    end

    # unconstrained, expensive
    @testset "unbound_exp_exp" begin
        opt_settings = AlgoConfig(
            max_iter = 30
        )
        mop = MixedMOP();
        add_objective!(mop, g1, :expensive)
        add_objective!(mop, g2, :expensive)
        x,fx = optimize!( opt_settings, mop, x0 )

        @test x[1] ≈ x[2] atol = .05
    end

    # unconstrained, heterogenous
    @testset "unbound_exp_cheap" begin
        opt_settings = AlgoConfig(
            max_iter = 30
        )
        mop = MixedMOP();
        add_objective!(mop, g1, :expensive)
        add_objective!(mop, g2, :cheap)
        x,fx = optimize!( opt_settings, mop, x0 )

        @test x[1] ≈ x[2] atol = .05
    end

    # constrained, cheap
    @testset "bound_cheap_cheap" begin
        opt_settings = AlgoConfig(
            Δ₀ = 0.1,
            max_iter = 40
        )
        mop = MixedMOP(lb = lb, ub = ub);
        add_objective!(mop, g1, :cheap)
        add_objective!(mop, g2, :cheap)
        x,fx = optimize!( opt_settings, mop, x0 )

        @test x[1] ≈ x[2] atol = .05
    end

    # constrained, expensive
    @testset "bound_exp_exp" begin
        opt_settings = AlgoConfig(
            Δ₀ = 0.2,
            max_iter = 55,
            rbf_kernel = :multiquadric,
            rbf_shape_parameter = cs -> let Δ = cs.iter_data.Δ; return 1/(10*Δ) end,
            max_model_points = 6,
            use_max_points = true,
            sampling_algorithm = :monte_carlo
        )
        mop = MixedMOP(lb = lb, ub = ub);
        add_objective!(mop, g1, :expensive)
        add_objective!(mop, g2, :expensive)
        x,fx = optimize!( opt_settings, mop, x0 )

        @test x[1] ≈ x[2] atol = .2
    end

    # constrained, heterogenous
    @testset "bound_exp_cheap" begin
        opt_settings = AlgoConfig(
            Δ₀ = 0.2,
            max_iter = 40,
            rbf_kernel = :multiquadric,
            rbf_shape_parameter = cs -> let Δ = cs.iter_data.Δ; return 1/(10*Δ) end,
            sampling_algorithm = :monte_carlo,
            all_objectives_descent = false,
            use_max_points = true
        )
        mop = MixedMOP(lb = lb, ub = ub);
        add_objective!(mop, g1, :expensive)
        add_objective!(mop, g2, :cheap)
        x,fx = optimize!( opt_settings, mop, x0 )

        @test x[1] ≈ x[2] atol = .1
    end
end


@testset "direct_search" begin
    # 1D1D
    x0 = [5.0];
    f1(x) = x[1]^2;

    for lb_ub ∈ [ ([],[]), ([-5.0], [5.0]), ([-10.0], [10.0])]
        lb, ub = lb_ub
        for type ∈ [:cheap, :expensive ]
            for ideal_point ∈ [[], [0.0]]
                opt_settings = AlgoConfig(
                    max_iter = 20,
                    descent_method = :direct_search,
                    ideal_point = ideal_point,
                )
                mop = MixedMOP(lb = lb, ub = ub);
                add_objective!(mop, f1, :type)
                x, fx = optimize!(opt_settings, mop, x0)
                @test x[end] ≈ 0.0 atol = .1
            end
        end
    end

    # 2D2D
    x0 = [ π , π^2 ]
    ub = 10 .* ones(2);
    lb = -ub;

    g1(x) = sum( (x .- 1.0).^2 );
    g2(x) = sum( (x .+ 1.0).^2 );
    for lb_ub ∈ [ ( [],[] ), ([-10.0,-10.0], [10.0, 10.0])]
        lb,ub = lb_ub
        for type_tuple ∈ [ (:cheap, :cheap), (:expensive, :expensive), (:expensive, :cheap) ]
            type1, type2 = type_tuple
            for ideal_point ∈ [ [], zeros(2) ]
                opt_settings = AlgoConfig(
                    max_iter = 50,
                    rbf_kernel = :multiquadric,
                    rbf_shape_parameter = cs -> let Δ = cs.iter_data.Δ; return 1/(10*Δ) end,
                    Δ₀ = 0.5,
                    all_objectives_descent = true,
                    use_max_points = type_tuple ==  (:expensive, :expensive) ? true : false,
                    max_model_points = 12,
                )
                mop = MixedMOP( lb = lb, ub = ub )
                add_objective!(mop, g1, type1)
                add_objective!(mop, g2, type2)
                x, fx = optimize!(opt_settings, mop, x0)
                @test x[1] ≈ x[2] atol = .5
            end
        end
    end

end

#=
# Convenience plotting during development
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
=#
