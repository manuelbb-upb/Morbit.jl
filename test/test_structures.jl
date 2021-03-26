project_path = joinpath(@__DIR__, "..");
using Pkg;
Pkg.activate(project_path);

module T

src_path = joinpath( @__DIR__, "..", "src");

using Parameters: @with_kw
using MathOptInterface;
const MOI = MathOptInterface;
using Memoization: @memoize;
using ThreadSafeDicts
import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

include(joinpath(src_path,"shorthands.jl"));
include(joinpath(src_path,"Interfaces.jl"));
include(joinpath(src_path,"diff_wrappers.jl"))

# implementations
include(joinpath(src_path,"VectorObjectiveFunction.jl"))
include(joinpath(src_path,"MixedMOP.jl"))
include(joinpath(src_path,"ExactModel.jl"))

include(joinpath(src_path,"objectives.jl"));
end 

using .T 
using Test 

lb = fill(-1, 2);
ub = fill(2, 2);

p = T.MixedMOP( lb, ub )

f1 = x -> x[2];

T.add_objective!( p, f1, :cheap );
o1 = p.vector_of_objectives[1];
m1, _ = T.init_model( T.ExactModel, o1 , p, [] );
@show T.get_gradient(m1, rand(2),1);
@show T.get_jacobian(m1, rand(2));

x̂ = rand(Float32,2);
x = T.unscale( p, x̂ )

@test f1( x ) ≈ T.eval_models(m1, x̂ )[1]
@test T.eval_all_objectives( p, x̂ )[1] ≈ f1(x)
@test T.eval_and_sort_objectives( p, x̂ )[1] ≈ f1(x)

