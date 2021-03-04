project_path = joinpath(@__DIR__, "..");
using Pkg;
Pkg.activate(project_path);
using Morbit;
using Logging;
global_logger( ConsoleLogger(stderr, Morbit.loglevel4; meta_formatter = morbit_formatter ) );

lb, ub = fill(-2,2), fill(2,2);
@show x0 = lb .+ (ub .- lb ) .* rand(2)
#%%
p = Morbit.MixedMOP(lb, ub)

f1 = x -> sum( (x.-1).^2 );
f2 = x -> sum( (x.+1).^2 );

lag_cfg = Morbit.LagrangeConfig( degree = 2 );
cfg = Morbit.RbfConfig(shape_parameter ="10/Δ", use_max_points = true);
 
taylor_cfg = Morbit.TaylorConfig( degree = 2, gradients = :fdm)
Morbit.add_objective!( p, f1, lag_cfg );
Morbit.add_objective!( p, f2, lag_cfg );

ac = Morbit.AlgoConfig(  
    db = Morbit.ArrayDB, 
    strict_backtracking = true,
    strict_acceptance_test = true,
    Δ_critical = 1e-10,
    Δ_min = 1e-13,
    max_evals = 50,
    max_iter = 50,
    descent_method = :ps,
    ps_algo = :LD_MMA,
    ideal_point_algo = :LD_MMA,
    ps_polish_algo = :LD_MMA,
)

X,_, id = Morbit.optimize( p, x0; algo_config = ac );
