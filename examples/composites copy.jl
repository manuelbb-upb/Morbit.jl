# Composite Functions & Re-Used Functions

## #src
using Morbit 

# In some situations, it might be necessary to re-use some function as part 
# of multiple objective or constraint functions.
# For example, one might have an objective that also doubles as a constraint.
# It is then unnecessary to build multiple surrogates for this inner function. 
# 
# Luckily, a `MOP` can be setup in such a way to only build a single surrogate model 
# per inner function.
# In fact, there are two distinct workflows and we will introduce both here.

# ## Recommended Way: `CompositeVecFun` and `CompositeSurrogate`

# Most often, the inner function is considered *expensive* whilst the outer function 
# (including possibly ``\operatorname{id}_{ℝ^n}``) is cheap and differentiable.
# We want to demonstrate how to exploit this kind of structure in setting up two problems.

# ### Constraining an Objective
#
# An easy application of composing functions is to restrict an objective by value.
# For example, adding the constraint ``f_1(x) ≤ 1`` to the 
# two parabola problem 
# ```math
# \\min_x \\begin{bmatrix}
# Σ (x_i - 1)^2 \\ 
# Σ (x_i + 1)^2
# \\end{bmatrix}
# ```
# will effectively cut the segment ``[(-1,-1), (0,0)]`` from the unconstrained
# pareto set.
# To use ``f_1`` both as an objective and a constraint, we have to construct a 
# `VecFun` and store the function index `f_ind`:

mop = MOP(2)

f1 = Morbit.make_vec_fun( x -> sum((x.-1).^2); n_out = 1, model_cfg = RbfConfig())
f_ind = Morbit._add_function!( mop, f1 )
Morbit._add_objective!(mop, f_ind)

add_rbf_objective!(mop, x -> sum((x.+1).^2))

# Using `f_ind` and another `VecFun` with `ExactConfig()`,
# such that their composition is the constraint ``f_1(x) - 1 \le 0``, we add the constraint:
φ = Morbit.make_vec_fun( z -> z[end] - 1; n_out = 1, model_cfg = ExactConfig())
Morbit._add_nl_ineq_constraint!(mop, f_ind, φ)

x0 = [-2.0, 3.0]
x, fx, _ = optimize(mop, x0; verbosity = 4)

# ### A More Complicated Example 
# The DTLZ1 problem[^1] for ``M\in ℕ`` objectives is:
# ```math 
# \\min_{x \\in [0,1]^n}
# \\begin{bmatrix}
# f_1(x) &= 0.5 \\cdot x_1 x_2 \\dotsm x_{M-1} \\left(1 + g(\\mathbf{x}_M)\\right) \\\\
# f_2(x) &= 0.5 \\cdot x_1 x_2 \\dotsm x_{M-2} (1-x_{M-1}) \\left(1 + g(\\mathbf{x}_M)\\right) \\\\
# &\\vdots \\\\
# f_{M-1}(x) &= 0.5 \\cdot x_1 (1-x_2) \\left(1 + g(\\mathbf{x}_M)\\right) \\\\
# f_M(x) &=  0.5 \\cdot (1-x_1) \\left(1 + g(\\mathbf{x}_M)\\right) 
# \\end{bmatrix}, 
# \\quad 
# \\mathbf{x}_M = [x_{M}, …, x_n]^T \\in ℝ^k,
# ```
# where the expensive inner function is 
# ```math
# g(\\mathbf{x}_M) = 
# k + \\sum_{x_i ∈ \\mathbf{x}_M} (x_i - 0.5)^2 - \\cos\\left( 20 π(x_i - 0.5)\\right)
# ```
# The number of variables is ``n = M + k - 1``. 
# For any suitable values of ``n`` and ``M<n`` we can set up an `MOP`
# programmatically.
# Take for example the following values:

n = 3
M = 2
k = n - M + 1

# For the model construction we have to reformulate our problem
# so that each objective has the form ``f_ℓ = φ_ℓ \circ g̃``, 
# where ``g̃`` is an ``n``-variate function.
# Suitable functions are 
# ```math 
# \\begin{algined}
# g̃(x) &= [ x_1, …, x_{M-1}, g(\\mathbf{x}_M) ], \\\\
# φ_1( \\mathbf z ) &= 0.5 \\cdot \\prod_{j=1}^{M-1} z_j  \\cdot (1 + z_M), \\\\
# φ_ℓ( \\mathbf z ) &= 0.5 \\cdot \\prod_{j=1}^{M-ℓ} z_j \\cdot (1-z_{M-ℓ+1}) \\cdot (1 + z_M),
# \\quad ℓ=2,…,M
# \\end{aligned}
# ```
# The inner function is easy enough to implement: 
g̃ = function (x)
	global M,k;
	ξ = x[M:end] .- 0.5
	return [ x[1:M-1]; k + sum( ξ.^2 ) - sum(cos.( 20 * π .* ξ)) ]
end

# We initialize the box constrained problem and use the internal method 
# `_add_function` to add the inner function. 
# It requires a `VecFun` and returns an `NLIndex` that we need for referencing.
mop = MOP( zeros(n), ones(n) )

g̃_vfun = Morbit.make_vec_fun( g̃; model_cfg = RbfConfig(), n_out = M)
g_ind = Morbit._add_function!( mop, g̃_vfun )

# We then have to turn the outer functions to `VecFun`s (so that they can
# be differentiated).
# When that is done, they can be passed to `_add_objective!` together with the index 
# of the inner function.

φ1 = Morbit.make_vec_fun( 
	z -> 0.5 * prod( z[1:M-1] ) * (1 + z[M] ); n_out = 1,
	model_cfg = ExactConfig(), diff_method = Morbit.AutoDiffWrapper
)
Morbit._add_objective!(mop, g_ind, φ1)

# In the same way we add the remaining objectives:
for ℓ = 2 : M
	φℓ = Morbit.make_vec_fun( 
		z -> 0.5 * prod( z[1:M-ℓ] ) * (1 - z[M-ℓ+1]) * (1 + z[M] ); n_out = 1,
		model_cfg = ExactConfig(), diff_method = Morbit.AutoDiffWrapper
	)
	Morbit._add_objective!(mop, g_ind, φℓ)
end

# We can now pass `mop` to `optimize` as always:
x0 = rand(n)
x, fx, _ = optimize(mop, x0; verbosity = 0, max_iter = 300)

# [^1]: “Scalable Test Problems for Evolutionary Multi-Objective Optimization”, Deb, Thiele, Laumanns & Zitzler