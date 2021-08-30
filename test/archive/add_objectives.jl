using Morbit
using Test

const m = Morbit;

# TODO test for different configurations (RbfConfig, LagrangeConfig etc.)

@testset "Adding Objectives" begin
    global mop, lb, ub;
    global f1, f2, f3, F56, F78

    mop_unconstrained = MixedMOP()
    @test !mop_unconstrained.is_constrained

    ub = 4 .* ones(2);
    lb = -ub;
    mop = MixedMOP( lb = lb, ub = ub )
    @test mop.is_constrained

    # 1) add a scalar expensive objective
    f1(x) = (x[1]-1)^2 + (x[2] -1)^2
    add_objective!( mop , f1, :expensive )
    @test mop.n_objfs == 1

    # 2a) add a scalar cheap objective with gradient provided
    function f2(x)
        (x[1] + 1)^2 + (x[2]+1)^2;
    end
    ∇f2(x) = [ 2*(x[1]+1); 2*(x[2]+1) ];
    add_objective!( mop, f2, ∇f2 );
    @test mop.n_objfs == 2

    # 2b) add scalar cheap objective with autodiff enabled
    f3 = x -> 2 * x[1] + 2 * x[2];
    add_objective!( mop, f3, :cheap );
    @test mop.n_objfs == 3

    # test internal indexing and sorting
    m.set_non_exact_indices!(mop)
    @test mop.non_exact_indices == [1,]

    m.set_sorting!(mop)
    @test mop.internal_sorting == [1, 2, 3]
    add_objective!( mop , f1, :expensive )
    m.set_sorting!(mop)
    @test mop.internal_sorting == [2,3,1,4]
    @test mop.reverse_sorting == [3,1,2,4]
    @test m.apply_internal_sorting( mop, [1,2,3,4] ) == mop.internal_sorting
    
    # 3) add vector cheap objective ( not super effective )
    function F56(x)
        y1 = exp( sum(x) );
        y2 = sin( x[1] );
        return [y1;y2]
    end

    add_objective!( mop, F56, :cheap, 2 )
    @test mop.n_objfs == 6

    # 4) add vector expensive objective ( not super effective )
    function F78(x)
        y1 = sum(x);
        y2 = 1.0;
        return [y1;y2]
    end

    add_objective!( mop, F78, :expensive, 2 )
    @test mop.n_objfs == 8
    m.set_sorting!(mop)
    m.set_non_exact_indices!(mop)
    # now 1,4,7,8 (and 5,6) should have been grouped together
    @test mop.internal_sorting == [2, 3, 5, 6, 1, 4, 7, 8]
    @test mop.non_exact_indices == [5,6,7,8]
end

@testset "Objective Evaluations" begin
    global mop, lb, ub;
    global f1, f2, f3, F56, F78 # f1 & F78 are expensive

    # test evaluation for constrained problem

    X = lb .+ (ub .- lb) .* rand(2);
    @test m.unscale(mop, m.scale(mop, X)) ≈ X

    # internally all vectors are in [0,1]^n
    x = m.scale(mop, X);

    all_eval = m.eval_all_objectives( mop, x);
    exp_eval = all_eval[mop.non_exact_indices];

    exp_comparison = [f1(X); f1(X); F78(X)][:];
    @test exp_eval ≈ exp_comparison

    Y = [ f1(X); f2(X); f3(X); f1(X); F56(X); F78(X) ][:];
    @test all_eval ≈ m.apply_internal_sorting(mop, Y)
    @test m.reverse_internal_sorting(mop, all_eval) ≈ Y
end


@testset "Internal Evaluations" begin
    lb = zeros(2);
    ub = 4 .* ones(2);

    s = [ 4 .* rand(2) for i=1:10 ];

    f1(x) = 1.0;
    f2(x) = (x[1]+1)^2 + (x[2] +1)^2;
    f(x) = [ f1(x); f2(x) ];

    p = MixedMOP(lb = lb, ub = ub);
    add_objective!(p,f1,:cheap,1);
    add_objective!(p,f2,:expensive,1);

    eval1 = Morbit.eval_all_objectives(p, s[1])
    eval2 = Morbit.eval_all_objectives.(p, s)

    p = MixedMOP(lb=lb, ub = ub)
    add_objective!(p,f,:expensive, 2)
    eval3 = m.eval_all_objectives.(p,s);

    global counter = 0;
    function batch_fun(x)
        global counter
        counter += 1
        if isa(x, Vector{T} where{T<:Real})
            return f(x)
        else
            return f.(x)
        end
    end

    p = MixedMOP(lb=lb, ub = ub);
    add_objective!(p, batch_fun, :expensive, 2, true);
    eval4 = Morbit.eval_all_objectives.(p,s);

    @test isa( eval1, Vector{Float64})
    @test length(eval1) == 2
    @test all( isa(e, Vector{Vector{Float64}} ) for e in [eval2, eval3, eval4])
    @test eval2[1] == eval3[1]
    @test eval3 == eval4
    @test counter == 1
end
