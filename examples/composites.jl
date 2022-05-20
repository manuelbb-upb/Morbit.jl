# # Composite Functions & Re-Used Functions

## #src
using Morbit 

# In some situations, it might be necessary to re-use some function as part 
# of multiple objective or constraint functions.
# For example, one might have an objective that also doubles as a constraint.
# It is then unnecessary to build multiple surrogates for this *inner* function. 

# Luckily, a problem of type `MOP` can be setup in such a way to only build a 
# single surrogate model per *inner* function.
# More specificially, there is the possibility to add an expensive inner function 
# ``g\colon ℝ^n\to ℝ^m``
# and then re-use this inner function in objectives or constraints of the form 
# ```math
# f_ℓ (\symbf x) 
# = 
# \left( φ_ℓ ∘ (\operatorname{id}_{ℝ^n} \times g \right)(\symbf x)
# = 
# φ_ℓ \left( \symbf x, g(\symbf x) \right), \quad 
# φ_ℓ \colon ℝ^n \times ℝ^m \to ℝ^{k_ℓ}.
# ```
# We expect ``f_ℓ`` to be cheap, in the sense that the partial derivatives are either 
# provided or can be computed using automatic differentiation or finite differencing.
# If derivative free models (e.g. RBF models, Lagrange models or finite differencing Taylor models)
# are used, then ``g`` does not have to be differentiable.
# Elsewise, the jacobian of ``f(\symbf x,\symbf y) = f_ℓ(\symbf x,\symbf y)`` 
# is computed with help of the chain rule as 
# ```math
# ℝ^{k,n} ∋ Df(\symbf x_0) = 
# \begin{bmatrix}
# D_{\symbf x} f( \symbf x_0, g(\symbf x_0) ) & D_{\symbf y} f(\symbf x_0, g(\symbf x_0))
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
# I_{n\times,n} \\
# Dg (\symbf x_0)
# \end{bmatrix}
# ```
#
# The model ``f̃`` for ``f`` is ``φ∘g̃^{(k)}``, where ``g̃^{(k)}`` is the current model for ``g``.
# The jacobian is 
# ```math 
# Df̃(\symbf x_0) = 
# \begin{bmatrix}
# D_{\symbf x} f( \symbf x_0, g(\symbf x_0) ) & D_{\symbf y} f(\symbf x_0, g(\symbf x_0))
# \end{bmatrix}
# \cdot 
# \begin{bmatrix}
# I_{n\times,n} \\
# Dg̃^{(k)} (\symbf x_0)
# \end{bmatrix}
# ```

# ## Recommended Way: `CompositeVecFun` and `CompositeSurrogate`

# We want to demonstrate how to exploit this kind of composite structure in 
# setting up two problems, first a variation of the two parabolas problem and then 
# a more complex example from the DTLZ family.

# ### Constraining an Objective
#
# An easy application of composing functions is to restrict an objective by value.
# For example, adding the constraint ``f_1(x) ≤ 1`` to the 
# two parabola problem 
# ```math
# \min_x \begin{bmatrix}
# Σ (x_i - 1)^2 \\
# Σ (x_i + 1)^2
# \end{bmatrix}
# = 
# \min_x \begin{bmatrix} f_1 (\symbf x)\\ f_2(\symbf x) \end{bmatrix}
# ```
# will effectively cut the segment ``[(-1,-1), (0,0)]`` from the unconstrained
# pareto set.
#
# To use ``f_1`` both as an objective and a constraint,
# we set it up as an inner function and then combine it with 
# the identity function (as the outer function) to 
# make the objective and the constraint.
#
# Both the inner and the outer function must be of type `VecFun`.
# The inner function is converted/wrapped via `make_inner_function` and 
# then added to the model with the internal method `_add_function`:

mop = MOP(2)

f1 = Morbit.make_vec_fun( 
	x -> sum((x.-1).^2); 
	n_out = 1, model_cfg = RbfConfig()
)
f_ind = Morbit._add_function!( mop, f1 )

# Usually, we would also have to construct the outer function 
# in a similar way (but with `make_outer_fun` -- see below for the constraint).
# For the special case where the outer function is the identity,
# we can simply provide the index:
Morbit._add_objective!(mop, f_ind)

# The second objective is added as usual:
add_rbf_objective!(mop, x -> sum((x.+1).^2))

# Now, the constraint is ``f_1(\symbf x) - 1 \le 0`` and hence a suitable 
# outer function is ``φ(\symbf x, f ) = f - 1``.
# It is constructed with `make_outer_fun`, and we keep in mind that the 
# output of the inner function has vector output (of length 1).
# Additionally, we have to provide `n_vars`, the length of ``x``:
φ = Morbit.make_outer_fun( 
	(x,f) -> f[end] - 1; 
	n_out = 1, n_vars = 2
)
# To add it, we use `φ` as the third argument to `_add_XXX!`:
Morbit._add_nl_ineq_constraint!(mop, f_ind, φ)

x0 = [-2.0, 3.0]
x, fx, _ = optimize(mop, x0; 
	max_iter = 20, 
	verbosity = 0
)
x, fx
## #src

# ### A More Complicated Example 
#
# The DTLZ1 problem[^1] for ``M\in ℕ`` objectives is:
# ```math 
# \min_{x \in [0,1]^n}
# \left[ 
# \begin{aligned}
# f_1(x) &= 0.5 \cdot x_1 x_2 \dotsm x_{M-1} \left(1 + g(\symbf{x}_M)\right) \\
# f_2(x) &= 0.5 \cdot x_1 x_2 \dotsm x_{M-2} (1-x_{M-1}) \left(1 + g(\symbf{x}_M)\right) \\
# &\vdots \\
# f_{M-1}(x) &= 0.5 \cdot x_1 (1-x_2) \left(1 + g(\symbf{x}_M)\right) \\
# f_M(x) &=  0.5 \cdot (1-x_1) \left(1 + g(\symbf{x}_M)\right) 
# \end{aligned}
# \right], 
# \quad 
# \symbf{x}_M = [x_{M}, …, x_n]^T \in ℝ^k,
# ```
# where the expensive inner function is 
# ```math
# g(\symbf{x}_M) = 
# 100\left(k + \sum_{x_i ∈ \symbf{x}_M} (x_i - 0.5)^2 - \cos\left( 20 π(x_i - 0.5)\right)\right)
# ```
# The number of variables is ``n = M + k - 1``. 
# For any suitable values of ``n`` and ``M<n`` we can set up an `MOP`
# programmatically.
# Take for example the following values:

const n = 3
const M = 2
const k = n - M + 1

## #src
# For the model construction we have to reformulate our problem
# so that each objective has the form 
# ``f_ℓ = φ_ℓ \circ (\operatorname{id}_{ℝ^n} \times g̃)``, 
# where ``g̃`` is an ``n``-variate function.
# Suitable functions are 
# ```math 
# \begin{aligned}
# g̃(\symbf x) &= g(\symbf{x}_M), \\
# φ_1(\symbf x, z ) &= 0.5 ⋅ ∏_{j=1}^{M-1} x_j  ⋅ (1 + z), \\
# φ_ℓ(\symbf x, z ) &= 0.5 ⋅ ∏_{j=1}^{M-ℓ} x_j ⋅ (1-x_{M-ℓ+1}) ⋅ (1 + z),
# \quad ℓ=2,…,M
# \end{aligned}
# ```
# The inner function is easy enough to implement: 
g̃ = function (x)
	global M,k
	ξ = x[M:end] .- 0.5
	return 100*(k + sum( ξ.^2 ) - sum(cos.( 20 * π .* ξ)))
end

# We initialize the box constrained problem and use the internal method 
# `_add_function` to add the inner function. 
# It requires a `VecFun` and returns an `InnerIndex` that we need for referencing.
mop = MOP( zeros(n), ones(n) )

g̃_vfun = Morbit.make_vec_fun( g̃; model_cfg = RbfConfig(), n_out = 1)
g_ind = Morbit._add_function!( mop, g̃_vfun )

# We then have to turn the outer functions to `VecFun`s (so that they can
# be differentiated).
# When that is done, they can be passed to `_add_objective!` together with the index 
# of the inner function.
# As in the first example, `n_vars` is the dimension of ``\symbf x`` and `n_out=1` 
# indicates scaler valued outer functions:
φ1 = Morbit.make_outer_fun( 
	(x,z) -> 0.5 * prod( x[1:M-1] ) * (1 + z[end] ); 
	n_vars = n, n_out = 1,
	diff_method = Morbit.AutoDiffWrapper
)
Morbit._add_objective!(mop, g_ind, φ1)

# In the same way we add the remaining objectives:
for ℓ = 2 : M
	φℓ = Morbit.make_outer_fun( 
		(x,z) -> 0.5 * prod( x[1:M-ℓ] ) * (1 - x[M-ℓ+1]) * (1 + z[end] ); 
		n_vars = n, n_out = 1,
		diff_method = Morbit.AutoDiffWrapper
	)
	Morbit._add_objective!(mop, g_ind, φℓ)
end

# We can now pass `mop` to `optimize` as always:
x0 = rand(n) 
x, fx, _ = optimize(mop, x0; verbosity = 0, max_iter = 100)
x, fx

# Now, the inner function ``g`` is Rastrigin's function and has ``11^k-1`` local 
# optima, so we cannot expect that `x` belongs to the global Pareto Set with 
# ``\symbf x_M = 0.5``.
# Rather, the problem was selected to demonstrate the modelling.

# ## Automatic Function Generation
#
# The second way to add composite objectives is provided by automatic generation 
# of outer functions from special expression strings.
# I guess it is slower.
# 
# Suppose you have a problem `mop::MOP` and an inner function with index `g_ind`.
# You can then call `_add_objective!(mop, g_ind, expr_str)` 
# (or `add_nl_eq_constraint!(…)` or `add_nl_ineq_constraint!(…)`) to add a 
# composite function to the model.
# `expr_str` must be a string with describing a function of a variable `x`.
# Each occurence of the word "VREF" is substituted with an evaluation of the inner 
# function at `x`.

mop = MOP(2)

g = Morbit.make_vec_fun( x -> sum(x.^2); model_cfg = RbfConfig(), n_out = 1)
g_ind = Morbit._add_function!(mop, g)

objf_ind1 = Morbit._add_objective!(mop, g_ind, "sin( VREF[1] ) * sum(x)";
	n_vars = 2, n_out = 1
)

# If you cant to use a custom function in the expression string, it has to be 
# registered first:
Morbit.register_func(x -> cos(x), :my_func)

objf_ind2 = Morbit._add_objective!(mop, g_ind, "my_func( VREF[1] ) * sum(x)";
	n_vars = 2, n_out = 1
)

x, fx, _ = optimize(mop, rand(2); verbosity = 0)
x, fx
# [^1]: “Scalable Test Problems for Evolutionary Multi-Objective Optimization”, Deb, Thiele, Laumanns & Zitzlera