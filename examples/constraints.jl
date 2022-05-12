# # Constrained Optimization

using Pkg #src
Pkg.activate(@__DIR__) #src
using Test #src
## #src
using Morbit 

# ## Box Constraints
# 
# Box constraints are supported and treated as "un-relaxable".
# The true problem functions won't be evaluated outside of global box constraints.
#
# The easiest way to define box constraints is by calling the `MOP` constructor 
# with a lower bound vector and an upper bound vector.
n_vars = 2
lb = [-1.0, -2.0]
ub = [3.0, 4.0]
mop = MOP(lb, ub)

# If the problem is fully finitely box constrained, i.e., all variables 
# are constrained to finite intervals, then the algorithm internally 
# scales the global domain to the unit hypercube ``[0,1]^n`` and the trust region 
# radius is defined with respect to the scaled domain.

# Alternatively, the problem can be set up similarly to how its done with 
# `MathOptInterface`:
mop = MOP()

var_1 = Morbit.add_variable!(mop)
Morbit.add_lower_bound!(mop, var_1, -1.0)
Morbit.add_upper_bound!(mop, var_1, 3.0)

var_2 = Morbit.add_variable!(mop)
Morbit.add_lower_bound!(mop, var_2, -2.0)
Morbit.add_upper_bound!(mop, var_2, 4.0)

# Or, if the variables have not been added manually:

mop = MOP( n_vars )
vars = Morbit.var_indices(mop)

Morbit.add_lower_bound!(mop, vars[1], -1.0)
Morbit.add_upper_bound!(mop, vars[1], 3.0)

Morbit.add_lower_bound!(mop, vars[2], -2.0)
Morbit.add_upper_bound!(mop, vars[2], 4.0)

# To delete the bound on a variable, use `del_lower_bound!(mop, var_index)`
# or `del_upper_bound!(mop, var_index)`. \
# The bound vectors can be inspected with `full_lower_bounds` and `full_upper_bounds`.

# ## Linear Constraints 
# Linear constraints are supported, but treated as "relaxable", that is, 
# the true problem functions might be evaluated outside of the global feasible 
# set. 
# In theory, the original trust region algorithm supports any convex constraints 
# natively. However, it is difficult to check for convexity and the constraints 
# also have to be supported by the inner solver for the descent step calculation.
# For this reason, only linear constraints are passed to the inner solver without 
# modification (except possibly scaling).
# 
# Internally, a `MOP` stores linear constraints as `MOI.VectorAffineFunction`s. 
# They also can be added as such, using the internal `_add_eq_constraint!` or 
# `_add_ineq_constraint!` method.
const MOI = Morbit.MOI
# Construct ``x₂ ≤ 4 - x₁ \;  ⇔  \; x₁ + x₂ - 4 ≤ 0`` :
x1_term = MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1, vars[1]))
x2_term = MOI.VectorAffineTerm(1, MOI.ScalarAffineTerm(1, vars[2]))
lin_const = MOI.VectorAffineFunction([x1_term, x2_term], [-4,])
c1 = Morbit._add_ineq_constraint!(mop,lin_const)

# It is much easier to provide matrices, e.g., for
# ``x₂ ≤ x₁ + 3   ⇔  -x₁ + x₂ - 3 ≤ 0`` :
c2 = add_ineq_constraint!(mop, [-1 1], [3])

# ## Nonlinear Constraints 
# Relaxable nonlinear constraints are supported via an algorithm 
# extension.
# To add them to the model, the `add_nl_eq_constraint!` and `add_nl_ineq_constraint!`
# methods can be used. 
# These methods work just like the `add_objective!` method.
# The constraint functions have to be reformulated so as to conform to ``g(x) ≦ 0``.
# For example, if want to add the constraint ``x₂ ≥ (x₁-1)² - 2``, we have 
# to add the function ``g(x) = (x₁-1)² - x₂ - 2``.
c3 = add_nl_ineq_constraint!(
	mop, x -> (x[1] - 1)^2 - x[2] - 1;
	n_out = 1, model_cfg = ExactConfig()
)

# As can be seen, the same mandatory keyword arguments (`n_out` and `model_cfg`)
# are required as for objectives.
# The constraint with index `c3` will be evaluated exactly and by default 
# its derivatives are calculated using automatic differentiation.
# First order derivative functions could also be provided with the `gradients`
# or `jacobian` keyword arguments.
#
# We also support inexact constraint gradients!
# Just like with objectives, ask for a derivative surrogate model 
# with the `model_cfg` keyword.
# For scalar functions, there are also shorthands:
# * `add_exact_nl_eq_constraint!` and `add_exact_nl_ineq_constraint!`
# * `add_rbf_nl_eq_constraint!` and `add_rbf_nl_ineq_constraint!`
# * `add_lagrange_nl_eq_constraint!` and `add_lagrange_nl_ineq_constraint!`
# * `add_taylor_nl_eq_constraint!` and `add_taylor_nl_ineq_constraint!`
#
# Let's do it for ``x₁²+x₂²-10 ≤ 0``:
c4 = add_rbf_nl_ineq_constraint!(mop, x -> sum(x.^2)-10)

# ## Optimization

# For the problem to be setup completely, objectives are still missing:
o1 = add_lagrange_objective!(mop, x -> sum( (x .- 1).^2 ) )
o2 = add_taylor_objective!(mop, x -> sum( (x .+ 1).^2 ) )

# Now, we can call `optimize` as usual. 
# The initial vector `x0` must not necessarily be feasible for the nonlinear constraints.
# If it is not feasible, then the first iteration will enter 
# the so called "restoration" procedure.
# At the moment, this procedure is very expensive, as the true constraints 
# are used by `NLopt` to reduce the constraint violation.
x0 = [3.5, -1]
x, fx, ret, sdb, id, filter = optimize(mop, x0; max_iter = 10, verbosity = 0);

# Constraint values can be either be calculated …
## linear eq, linear ineq constraint values:
Morbit.eval_vec_linear_constraints_at_unscaled_site( x, mop )
## nonlinear ineq constraint values
Morbit.eval_nl_ineq_constraints_to_vec_at_unscaled_site( mop, x )
# … or extracted from the final `AbstractIterate` object `id`:
## linear inequality constraint values & nonlinear inequality values:
id.l_i, id.c_i

# ### Results
# Let's plot everything …
using CairoMakie
using CairoMakie.GeometryBasics

# First, the box constraints:
fig, ax, _ = poly( Point2f[ Tuple(lb), (ub[1], lb[2]), Tuple(ub), (lb[1], ub[2]) ],
	color = RGBAf(0,0,0,0), strokecolor = :blue, strokewidth = 2
)
nothing #hide 

# Now, the constraint boundaries:
xs = LinRange(lb[1], ub[1], 50)

y1 = - xs .+ 4
y2 = xs .+ 3
y3 = (xs .- 1).^2 .- 1
lines!(xs,y1; color = RGBf(204/255,51/255,1), label = "c1")
lines!(xs,y2; color = RGBf(204/255,0,153/255), label = "c2" )
lines!(xs,y3; color = RGBf(102/255,0,51/255), label = "c3" )

φ = LinRange(0,2*π,100)
x4 = sqrt(10) .* cos.(φ)
y4 = sqrt(10) .* sin.(φ)
lines!(x4,y4, label = "c4")
nothing #hide

# And the constraint interior:
xs = LinRange(lb[1], ub[1], 300)
ys = LinRange(lb[2], ub[2], 300)
θ = function (x1,x2)
	constraint_vector = [
		lb[1] - x1;
		lb[2] - x2;
		x1 - ub[1];
		x2 - ub[2];
		x1 + x2 - 4;
		-x1 + x2 - 3;
		(x1 - 1)^2 - x2 - 1;
		x1^2 + x2^2 - 10
	]
	if maximum(constraint_vector) <= 0 
		return 0
	else
		return 1
	end
end
zs = [θ(x,y) for x = xs, y=ys]
image!(xs,ys,zs; colormap = [RGBAf(0,0,0,0.2), RGBAf(0,0,0,0)])
nothing #hide

# Finally, plot the unconstrained Pareto set (the line connecting (-1,-1) and (1,1))
# as well the iterates:
lines!([(-1,-1),(1,1)], color = :green, label = "PS")

x_iter = [ Tuple(iter.x) for iter = sdb.iter_data]
scatter!(x_iter, color = :red, markersize = 5 )
lines!(x_iter, color = :orange)

axislegend()
fig
