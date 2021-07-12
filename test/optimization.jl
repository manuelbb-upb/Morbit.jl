#using Morbit
#using Test

# Two standard (easy) test problems with known Pareto Data
# Many combinations of constaints, heterogenity, descent method and sampling algorithm are used

#using Logging
#global_logger(SimpleLogger(stdout, Logging.Debug))
#%%
@testset "1In1OutProblem_SteepestDescent" begin

    x0 = [5.0];
    f1(x) = x[1]^2;

    # I) unconstrained
    opt_settings = AlgoConfig(
        max_iter = 30
    );
    ## 1) treat objective as cheap
    mop = MixedMOP();
    add_objective!( mop, f1, :cheap );
    x,fx = optimize(mop, x0; algo_config = opt_settings);

    @test x ≈ [ 0.0 ] atol=1e-2

    ## 2) treat objective as expensive
    opt_settings = AlgoConfig(
        max_iter = 30
    );
    mop = MixedMOP();
    add_objective!( mop, f1, :expensive )
    x,fx = optimize(mop, x0; algo_config = opt_settings)

    @test x ≈ [ 0.0 ] atol = 1e-2

    #II) constrained

    ## 1) treat objective as cheap
    opt_settings = AlgoConfig(
        max_iter = 10
    );
    mop = MixedMOP( [-8.0], [8.0] );
    add_objective!( mop, f1, :cheap)
    x,fx = optimize(mop, x0; algo_config = opt_settings)

    @test x ≈ [ 0.0 ] atol=0.1

    ## 2) treat objective as expensive
    opt_settings = AlgoConfig(
        max_iter = 30
    );
    mop = MixedMOP( [-8.0], [8.0] );
    add_objective!( mop, f1, :expensive);
    x,fx = optimize(mop, x0; algo_config = opt_settings);

    @test x ≈ [ 0.0 ] atol=.1
end

#%%
@testset "2In2OutProblem_Steepest_Descent" begin
    x0 = [ π , π^2 ]
    ub = 10 .* ones(2);
    lb = -ub;

    g1(x) = sum( (x .- 1.0).^2 );
    g2(x) = sum( (x .+ 1.0).^2 );

    # unconstrained, cheap
    @testset "unbound_cheap_cheap" begin
        opt_settings = AlgoConfig(
            Δ_0= 0.2,
            max_iter = 30,
        );
        mop = MixedMOP();
        add_objective!(mop, g1, :cheap);
        add_objective!(mop, g2, :cheap);
        x,fx = optimize(mop, x0; algo_config = opt_settings);

        @test x[1] ≈ x[2] atol = .1
    end

    # unconstrained, expensive
    @testset "unbound_exp_exp" begin
        opt_settings = AlgoConfig(
            max_iter = 30
        );
        mop = MixedMOP();
        add_objective!(mop, g1, :expensive);
        add_objective!(mop, g2, :expensive);
        x,fx = optimize(mop, x0; algo_config = opt_settings);

        @test x[1] ≈ x[2] atol = .1
    end

    # unconstrained, heterogenous
    @testset "unbound_exp_cheap" begin
        opt_settings = AlgoConfig(
            max_iter = 30
        );
        mop = MixedMOP();
        add_objective!(mop, g1, :expensive);
        add_objective!(mop, g2, :cheap);
        x,fx = optimize(mop, x0; algo_config = opt_settings);

        @test x[1] ≈ x[2] atol = .1
    end

    # constrained, cheap
    @testset "bound_cheap_cheap" begin
        opt_settings = AlgoConfig(
            max_iter = 30
        );
        mop = MixedMOP(lb, ub);
        add_objective!(mop, g1, :cheap);
        add_objective!(mop, g2, :cheap);
        x,fx = optimize(mop, x0; algo_config = opt_settings);

        @test x[1] ≈ x[2] atol = .1
    end

    # constrained, expensive
    @testset "bound_exp_exp" begin
        opt_settings = AlgoConfig(
            max_iter = 50,
        );
        mop = MixedMOP(lb,ub);
        add_objective!(mop, g1, :expensive);
        add_objective!(mop, g2, :expensive);
        x,fx = optimize(mop, x0; algo_config = opt_settings);

        @test x[1] ≈ x[2] atol = .1
    end

    # constrained, heterogenous
    @testset "bound_exp_cheap" begin
        opt_settings = AlgoConfig(
            max_iter = 50,
            Δ_max = .4,
            _eps_crit = 0.1,
            strict_acceptance_test = true,
        );
        mop = MixedMOP(lb,ub);
        add_objective!(mop, g1, RbfConfig(kernel=:multiquadric));
        add_objective!(mop, g2, :cheap);
        x,fx = optimize(mop, x0; algo_config = opt_settings);

        @test x[1] ≈ x[2] atol = .1
    end
end
#%%
#=
@testset "direct_search" begin
    # 1D1D
    x0 = [5.0];
    f1(x) = x[1]^2;

    for lb_ub ∈ [ (Float32[],Float32[]), ([-5.0], [5.0]), ([-10.0], [10.0])]
        lb, ub = lb_ub
        for type ∈ [:cheap, :expensive ]
            for ideal_point ∈ [Float32[], [0.0]]
                opt_settings = AlgoConfig(
                    max_iter = 20,
                    descent_method = :direct_search,
                    ideal_point = ideal_point,
                )
                mop = MixedMOP(lb,ub)
                Morbit.scale( mop, x0)
                add_objective!(mop, f1, type)
                x, fx = optimize(mop, x0; algo_config = opt_settings)
                @test x[end] ≈ 0.0 atol = .1
            end
        end
    end

    # 2D2D
    #=
    x0 = [ π , π^2 ]
    ub = 10 .* ones(2);
    lb = -ub;

    g1(x) = sum( (x .- 1.0).^2 );
    g2(x) = sum( (x .+ 1.0).^2 );
    for lb_ub ∈ [ ( [],[] ), ([-10.0,-100], [10.0, 10.0])]
        lb,ub = lb_ub
        for type_tuple ∈ [ (:cheap, :cheap), (:expensive, :expensive), (:expensive, :cheap) ]
            type1, type2 = type_tuple
            for ideal_point ∈ [ [], zeros(2) ]
                opt_settings = AlgoConfig(
                    max_iter = 50,
                    rbf_kernel = :multiquadric,
                    rbf_shape_parameter = cs -> let Δ = cs.iter_data.Δ; return 1/(10*Δ) end,
                    Δ_0= 0.5,
                    all_objectives_descent = true,
                    use_max_points = type_tuple ==  (:expensive, :expensive) ? true : false,
                    max_model_points = 12,
                )
                mop = MixedMOP(lb,ub)
                add_objective!(mop, g1, type1)
                add_objective!(mop, g2, type2)
                x, fx = optimize(mop, x0; algo_config = opt_settings)
                @test x[1] ≈ x[2] atol = .5
            end
        end
    end
    =#
end
=#