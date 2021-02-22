project_path = joinpath(@__DIR__, "..");
using Pkg;
Pkg.activate(project_path);
using Morbit;

lb, ub = fill(-2,2), fill(2,2);
p = Morbit.MixedMOP(lb, ub)
@show x0 = lb .+ (ub .- lb ) .* rand(2)

f1 = x -> sum( (x.-1).^2 );
f2 = x -> sum( (x.+1).^2 );

Morbit.add_objective!( p, f1, :cheap );
Morbit.add_objective!( p, f2, :cheap );

ac = Morbit.AlgoConfig(  db = Morbit.NoDB, max_iter = 20 );

X,_ = Morbit.optimize( p, x0; algo_config = ac );