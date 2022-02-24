module LagrangeTests
using ForwardDiff
using Morbit
using Test
#%%
mop = MixedMOP(2)
add_objective!(mop, x -> sum(x.^4), LagrangeConfig(;optimized_sampling=false))
x0 = rand(2)
smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )

mod = sc.surrogates[1].model
objf = Morbit.list_of_objectives(smop)[1]
meta = sc.surrogates[1].meta

# `Morbit.eval_vfun` evaluates at a site from the original domain
_eval_vfun_original = x -> Morbit.eval_vfun(objf, x)
_eval_vfun_scaled = x -> Morbit.eval_vfun( objf, Morbit.unscale(x, smop) )

# `eval_models`, `get_gradient` etc., on the other hand,
# take input from [0,1]^n:
_eval_model = x -> Morbit.eval_models( mod, x )

x0_s = Morbit.scale( x0, smop )

#%%
# Test some configurations
# Only degrees 1 and 2 are allowed
@test_throws Any Morbit.LagrangeConfig(;degree=0)
@test_throws Any Morbit.LagrangeConfig(;degree=-1)
@test_throws Any Morbit.LagrangeConfig(;degree=3)

# Λ must be >1
@test_throws Any Morbit.LagrangeConfig(;LAMBDA=-1)
@test_throws Any Morbit.LagrangeConfig(;LAMBDA=1.0)

# Algorithm must be derivative-free 
@test_throws Any Morbit.LagrangeConfig(; algo1_solver = :LD_MMA)
@test_throws Any Morbit.LagrangeConfig(; algo2_solver = :LD_MMA)
@test_throws Any Morbit.LagrangeConfig(; algo2_solver = :L)

#%%
# preparation

const scalar_test_funcs = [
	x -> sum(x),
	x -> sum( x.^2 ),
	x -> sum( x.^(2 : length(x) + 1) ),
	x -> exp( sum( x.^2 ) / 100 ),
	x -> 1 / (10 + abs(sum(x))) * sin( sum(x) )
]

const lagrange_configs = [
	LagrangeConfig(; degree = 1 ),
	LagrangeConfig(; degree = 1, LAMBDA = 10 ),
	LagrangeConfig(; degree = 2 ),
	LagrangeConfig(; degree = 2, LAMBDA = 10 ),
	LagrangeConfig(; optimized_sampling = false, degree = 1 ),
	LagrangeConfig(; optimized_sampling = false, degree = 2 ),
]

#%%
for cfg in lagrange_configs
	for n_vars = 1 : 3
		for n_out = 1 : 2
			for f1 in scalar_test_funcs 
				for unconstrained in [true, false]

					lb = fill(-5.0, n_vars )
					ub = fill(6.0, n_vars )

					if unconstrained
						mop = Morbit.MixedMOP(n_vars)
					else						
						mop = Morbit.MixedMOP(lb,ub)
					end
					
					if n_out == 1 
						Morbit.add_objective!( mop, f1, cfg )
					else
						f_remaining = rand( scalar_test_funcs, n_out - 1 )
						test_func = x -> [ f1(x); [φ(x) for φ in f_remaining] ]
						Morbit.add_vector_objective!( mop, test_func, cfg; n_out )
					end

					x0 = Morbit._rand_box_point(lb, ub)

					smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )

					mod = sc.surrogates[1].model
					objf = Morbit.list_of_objectives(smop)[1]
					meta = sc.surrogates[1].meta

					# `Morbit.eval_vfun` evaluates at a site from the original domain
					_eval_vfun_original = x -> Morbit.eval_vfun(objf, x)
					_eval_vfun_scaled = x -> Morbit.eval_vfun( objf, Morbit.unscale(x, smop) )

					# `eval_models`, `get_gradient` etc., on the other hand,
					# take input from [0,1]^n:
					_eval_model = x -> Morbit.eval_models( mod, x )

					x0_s = Morbit.scale( x0, smop )

					#%%
					# check if scaling went alright:
					@test x0_s ≈ Morbit.get_x( iter_data )
					@test _eval_vfun_original( x0 ) ≈ _eval_vfun_scaled( x0_s )

					# did we succeed in making an orthogonal basis?
					X = Morbit.get_site.( data_base, meta.interpolation_indices )
					Y = Morbit.get_value.( data_base, meta.interpolation_indices )

					num_polys = length(X)

					for (i,poly) in enumerate(meta.lagrange_basis)
						res_vec = zeros(num_polys)
						res_vec[i] = 1
						@test poly.(X) ≈ res_vec
					end

					for (i,poly) in enumerate(mod.basis)
						res_vec = zeros(num_polys)
						res_vec[i] = 1
						@test poly.(X) ≈ res_vec
					end

					# if the database is transformed, then the sites are from the 
					# original domain => we must scale them for the model. 
					# else they are in [0,1] and can be passed to the model.
					_eval_func = Morbit.is_transformed( data_base ) ? x -> _eval_model( Morbit.scale(x, smop) ) : _eval_model
			
					@test all([ isapprox(_eval_func( X[i] ), Y[i][ meta.out_indices ]; atol = 1e-2, rtol = 1e-3) for i = eachindex(X) ])

					#%%
					# does evaluation work alright?
					if 1 in meta.interpolation_indices
						@test _eval_model( x0_s ) ≈ _eval_vfun_original( x0 )
					end
					@test Morbit.get_gradient( mod, x0_s, 1) ≈ ForwardDiff.gradient( x -> _eval_model(x)[1], x0_s )
					@test Morbit.get_jacobian( mod, x0_s ) ≈ ForwardDiff.jacobian( _eval_model, x0_s )
				end#for
			end#for
		end#for
	end#for
end#for

end#module 

using .LagrangeTests