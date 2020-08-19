using Morbit
using Test

const m = Morbit;

@testset "Adding Objectives" begin
    global mop, lb, ub;
    global f1, f2, f3, F45, F67

    mop_unconstrained = MixedMOP()
    @test !mop_unconstrained.is_constrained

    ub = 4 .* ones(2);
    lb = -ub;
    mop = MixedMOP( lb = lb, ub = ub )
    @test mop.is_constrained

    # 1) add a scalar expensive objective
    f1(x) = (x[1]-1)^2 + (x[2] -1)^2
    add_objective!( mop , f1, :expensive )
    @test mop.n_exp == 1

    # 2a) add a scalar cheap objective with gradient provided
    function f2(x)
        (x[1] + 1)^2 + (x[2]+1)^2;
    end
    ∇f2(x) = [ 2*(x[1]+1); 2*(x[2]+1) ];
    add_objective!( mop, f2, ∇f2 );
    @test mop.n_exp == 1
    @test mop.n_cheap == 1

    # 2b) add scalar cheap objective with autodiff enabled
    f3 = x -> 2 * x[1] + 2 * x[2];
    add_objective!( mop, f3, :cheap )
    @test mop.n_exp == 1
    @test mop.n_cheap == 2

    @test mop.internal_sorting == [1, 2, 3]
    @test mop.vector_of_gradient_funcs[1](zeros(2)) == mop.vector_of_gradient_funcs[2](zeros(2))

    # 3) add vector cheap objective ( not super effective )
    function F45(x)
        y1 = exp( sum(x) );
        y2 = sin( x[1] );
        return [y1;y2]
    end

    add_objective!( mop, F45, :cheap, 2 )
    @test mop.n_exp == 1
    @test mop.n_cheap == 4
    @test mop.internal_sorting == [1, 2, 3, 4, 5]

    # 4) add vector expensive objective ( not super effective )
    function F67(x)
        y1 = sum(x);
        y2 = 1.0;
        return [y1;y2]
    end

    add_objective!( mop, F67, :expensive, 2 )
    @test mop.n_exp == 3
    @test mop.n_cheap == 4
    @test mop.internal_sorting == [1, 6, 7, 2, 3, 4, 5]

end

@testset "Objective Evaluations" begin
    global mop, lb, ub;
    global f1, f2, f3, F45, F67 # f1 & F67 are expensive

    # test evaluation for constrained problem

    X = lb .+ (ub .- lb) .* rand(2);
    @test m.unscale(mop, m.scale(mop, X)) ≈ X

    # internally all vectors are in [0,1]^n
    x = m.scale(mop, X);

    exp_eval = m.eval_expensive_objectives( mop, x )
    @test length( exp_eval ) == mop.n_exp
    exp_comparison = [f1(X); F67(X)][:];
    @test exp_eval ≈ exp_comparison

    cheap_eval = m.eval_cheap_objectives( mop, x )
    @test length( cheap_eval ) == mop.n_cheap
    cheap_comparison = [ f2(X); f3(X); F45(X) ][:];
    @test cheap_eval ≈ cheap_comparison

    all_eval = m.eval_all_objectives( mop, x )
    @test all_eval ≈ [ exp_comparison; cheap_comparison ]

    Y = [ f1(X); f2(X); f3(X); F45(X); F67(X) ][:];
    @test all_eval ≈ m.apply_internal_sorting(mop, Y)
    @test m.reverse_internal_sorting(mop, all_eval) ≈ Y

    # test derivatives of cheap functions

    model = NamedTuple()

    ∇f2(x) = [ 2*(x[1]+1); 2*(x[2]+1) ];
    ∇f3(x) = 2 .* ones(2);
    ∇f4(x) = ones(2) .* exp(sum(x));
    ∇f5(x) = [cos(x[1]); 0.0];

    @test ∇f2(X) ≈ m.eval_grad( mop, model, x, 4 )
    @test ∇f3(X) ≈ m.eval_grad( mop, model, x, 5 )
    @test ∇f4(X) ≈ m.eval_grad( mop, model, x, 6 )
    @test ∇f5(X) ≈ m.eval_grad( mop, model, x, 7 )

    jacobian_compare = hcat( ∇f2(X), ∇f3(X), ∇f4(X), ∇f5(X) )'
    @test jacobian_compare ≈ m.eval_jacobian( mop, model, x )   # as model is empty named tuple, the expensive gradients are not calculated
end


@testset "Internal Evaluations" begin
    lb = zeros(2)
    ub = 4 .* ones(2);

    s = [ 4 .* rand(2) for i=1:10 ]

    f1(x) = 1.0#(x[1]-1)^2 + (x[2] -1)^2
    f2(x) = (x[1]+1)^2 + (x[2] +1)^2
    f(x) = [ f1(x); f2(x) ]

    p = MixedMOP(lb = lb, ub = ub)
    add_objective!(p,f1,:cheap,1)
    add_objective!(p,f2,:expensive,1)

    eval1 = Morbit.eval_all_objectives(p, s[1])
    eval2 = Morbit.eval_all_objectives.(p, s)

    p = MixedMOP(lb=lb, ub = ub)
    add_objective!(p,f,:expensive, 2)
    eval3 = Morbit.eval_all_objectives.(p,s)

    counter = 0;
    function g(x)
        global counter
        counter += 1
        if isa(x, Vector{T} where{T<:Real})
            return f(x)
        else
            return f.(x)
        end
    end

    p = MixedMOP(lb=lb, ub = ub)
    add_objective!(p, g, :expensive, 2, true)
    eval4 = Morbit.eval_all_objectives.(p,s)

    @test isa( eval1, Vector{Float64})
    @test length(eval1) == 2
    @test all( isa(e, Vector{Vector{Float64}} ) for e in [eval2, eval3, eval4])
    @test eval2[1] == reverse(eval3[1])
    @test eval3 == eval4
end
