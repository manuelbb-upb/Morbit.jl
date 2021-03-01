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

cfg = Morbit.LagrangeConfig( degree = 2 );
cfg = Morbit.RbfConfig();
 
taylor_cfg = Morbit.TaylorConfig( degree = 2, gradients = :fdm)
Morbit.add_objective!( p, f1, cfg );
Morbit.add_objective!( p, f2, cfg );

ac = Morbit.AlgoConfig(  
    db = Morbit.ArrayDB, 
    strict_backtracking = true,
    strict_acceptance_test = true,
    Δ_critical = 1e-10,
    Δ_min = 1e-13,
    max_iter = 10)#, max_evals = 10 );

X,_ = Morbit.optimize( p, x0; algo_config = ac );
