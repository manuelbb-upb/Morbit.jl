project_path = joinpath(@__DIR__, "..");
using Pkg;
Pkg.activate(project_path);
using Morbit;

lb, ub = fill(-2,2), fill(2,2);
@show x0 = lb .+ (ub .- lb ) .* rand(2)
#%%
p = Morbit.MixedMOP(lb, ub)

f1 = x -> sum( (x.-1).^2 );
f2 = x -> sum( (x.+1).^2 );

cfg = Morbit.ExactConfig(gradients=:autodiff)

Morbit.add_objective!( p, f1, cfg );
Morbit.add_objective!( p, f2, cfg );

ac = Morbit.AlgoConfig(  
    db = Morbit.NoDB, 
    strict_backtracking = true,
    strict_acceptance_test = true,
    max_iter = 40)#, max_evals = 10 );

X,_ = Morbit.optimize( p, x0; algo_config = ac );
X