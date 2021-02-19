project_path = joinpath(@__DIR__, "..");
using Pkg;
Pkg.activate(project_path);
using Morbit;

lb, ub = fill(-2,2), fill(2,2);
p = Morbit.MixedMOP(lb, ub)
x0 = rand(2)

f1 = x -> sum( (x.-1).^2 );
f2 = x -> sum( (x.+1).^2 );

Morbit.add_objective!( p, f1, :cheap );
Morbit.add_objective!( p, f2, :cheap );

Morbit.optimize( p, x0 )