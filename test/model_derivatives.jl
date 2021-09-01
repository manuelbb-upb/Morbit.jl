module ModelDerivatives

using ForwardDiff
using Morbit
using Test

using Combinatorics

const scalar_test_funcs = [
	x -> sum(x),
	x -> sum( x.^2 ),
	x -> sum( x.^(2 : (length(x) + 1)) ),
	x -> exp( sum( x.^2 ) / 100 ),
	x -> ( 1 / (10 + abs(sum(x))) ) * sin( sum(x) )
]

const configs = [
	ExactConfig(;gradients = :autodiff),
	ExactConfig(;gradients = :fdm),
	LagrangeConfig(; degree = 1 ),
	LagrangeConfig(; degree = 2 ),
	RbfConfig(;kernel = :gaussian),
	RbfConfig(;kernel = :multiquadric),
	RbfConfig(;kernel = :inv_multiquadric),
	RbfConfig(;kernel = :cubic),
	RbfConfig(;kernel = :gaussian, polynomial_degree = -1),
	RbfConfig(;kernel = :inv_multiquadric, polynomial_degree = -1),
	RbfConfig(;kernel = :cubic, polynomial_degree = -1),
	RbfConfig(;kernel = :gaussian, polynomial_degree = 0),
	RbfConfig(;kernel = :inv_multiquadric, polynomial_degree = 0),
	RbfConfig(;kernel = :cubic, polynomial_degree = 0),
	TaylorConfig(;degree=1),
	TaylorConfig(;degree=2),
	TaylorApproximateConfig(mode=:autodiff),
	TaylorApproximateConfig(mode=:fdm)
]

# TODO `TaylorCallbackConfig` cannot be tested this way 

NVAR_RANGE = 1:3
NOUT_RANGE = 1:2
#%%

function perform_tests( cfg, n_vars, n_out, k, unconstrained, other_func_indices, x0 = Float64[] )
	f1 = scalar_test_funcs[k]

	lb = fill(-5.0, n_vars )
	ub = fill(6.0, n_vars )

	if unconstrained
		mop = Morbit.MixedMOP(n_vars)
	else						
		mop = Morbit.MixedMOP(lb,ub)
	end

	test_func = isempty(other_func_indices) ? f1 : x -> [ f1(x); [ φ(x) for φ in scalar_test_funcs[other_func_indices] ] ]
	Morbit.add_vector_objective!( mop, test_func, cfg; n_out )

	if isempty(x0) x0 = Morbit._rand_box_point(lb, ub) end
	@show x0

	smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )

	_mod = sc.surrogates[1].model
	objf = Morbit.list_of_objectives(smop)[1]
	meta = sc.surrogates[1].meta

	# `Morbit.eval_objf` evaluates at a site from the original domain
	_eval_objf_original = x -> Morbit.eval_objf(objf, x)
	_eval_objf_scaled = x -> Morbit.eval_objf( objf, Morbit.unscale(x, smop) )

	# `eval_models`, `get_gradient` etc., on the other hand,
	# take input from [0,1]^n:
	_eval_model = x -> Morbit.eval_models( _mod, x )

	x0_s = Morbit.scale( x0, smop )

	@test x0_s ≈ Morbit.get_x( iter_data )
	@test _eval_objf_original( x0 ) ≈ _eval_objf_scaled( x0_s )

	@test isapprox(Morbit.get_gradient( _mod, x0_s, 1), ForwardDiff.gradient( x -> _eval_model(x)[1], x0_s ); rtol = 0.001 )
	@test isapprox(Morbit.get_jacobian( _mod, x0_s ), ForwardDiff.jacobian( _eval_model, x0_s ); rtol = 0.001 )

	for i = 1 : 10
		ξ0= Morbit._rand_box_point( lb, ub )
		ξ0_s = Morbit.scale( ξ0, smop )
		@test isapprox(Morbit.get_gradient( _mod, ξ0_s, 1), ForwardDiff.gradient( x -> _eval_model(x)[1], ξ0_s ); rtol = 0.001 )
		@test isapprox(Morbit.get_jacobian( _mod, ξ0_s ), ForwardDiff.jacobian( _eval_model, ξ0_s ); rtol = 0.001 )
	end
end#function

for cfg in configs
	for n_vars = NVAR_RANGE
		for n_out = NOUT_RANGE
			for k in eachindex(scalar_test_funcs)
				for unconstrained in [false,true]
					@info n_vars, n_out, k, unconstrained
					@info cfg

					other_func_indices = rand( collect(eachindex(scalar_test_funcs)), n_out - 1 )

#					for other_func_indices in combinations( eachindex(scalar_test_funcs), n_out - 1 )
						perform_tests(cfg, n_vars, n_out, k, unconstrained, other_func_indices )
#					end
				end#for
			end#for
		end#for
	end#for
end#for

end#module 

using .ModelDerivatives