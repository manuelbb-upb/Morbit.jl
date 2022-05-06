
using Morbit 
using Test

f1 = x -> sum( x.^2 )

function _initialize( model_cfg, algo_config = nothing; num_vars = 3, constrained = false  )
	global f1

	if constrained 
		mop = MOP( fill(.25, num_vars), fill(.75, num_vars) )
	else
		mop = MOP(num_vars)
	end

	f1_wrapped = Morbit.make_vec_fun( f1; n_out = 1, model_cfg )

	nl_ind = Morbit._add_function!( mop, f1_wrapped )
	objf_ind = Morbit._add_objective!( mop, nl_ind )

	x0 = rand( num_vars )
	
	smop, id, sdb, sc, ac, filter, scal = Morbit.initialize_data(mop, x0; algo_config);
	return nl_ind, objf_ind, smop, id, sdb, sc, ac, filter, scal
end

for num_vars ∈ [2,5,10]
	for kernel in Morbit.RbfKernels
		for polynomial_degree = -1:1
			for constrained ∈ [true, false]

				# First, test if 
				# i) `max_evals` is respected and 
				# ii) the models can be build with too few points
				model_cfg = RbfConfig(;
					kernel = kernel,
					polynomial_degree,
					max_evals = 1,
					max_model_points = 1, 	# avoid round 4 for now
				)

				nl_ind, objf_ind, smop, id, sdb, sc, ac, filter, scal = _initialize(model_cfg; num_vars, constrained)
				@test Morbit.num_evals( Morbit._get(smop, nl_ind ) ) == 1
				@test Morbit.num_evals( Morbit._get(smop, objf_ind ) ) == 1
				
				# will the model be fully linear if we update after putting many points in the db? 
				if polynomial_degree == 1
					Δ = Morbit.get_delta(id)
					x = Morbit.get_x_scaled(id)
					db = Morbit.get_sub_db( sdb, (nl_ind,) )
					lb, ub = Morbit.local_bounds( scal, x, Δ )
			 		w = ub .- lb
					for i = 1 : 50*num_vars
						ξ = lb .+ w .* rand(num_vars)
						Morbit.new_result!( db, ξ )
					end
					Morbit.update_surrogates!(sc, smop, scal, id, sdb, ac)
					@test Morbit.fully_linear( sc )
				end

				## now set the budget via algo_config
				model_cfg = RbfConfig(;
					kernel,
					polynomial_degree,
				)

				algo_config = AlgorithmConfig(; max_evals = 1 )

				nl_ind, objf_ind, smop, id, sdb, sc, ac, filter, scal = _initialize(model_cfg, algo_config; num_vars, constrained )
				@test Morbit.num_evals( Morbit._get(smop, nl_ind ) ) == 1
				@test Morbit.num_evals( Morbit._get(smop, objf_ind ) ) == 1

				## does round 4 run, even with fewer than `num_vars + 1` points?  
				db = Morbit.get_sub_db( sdb, (nl_ind,) )
				Δ = Morbit.get_delta(id)
				Δ_max = Morbit.delta_max(ac)
				θ = model_cfg.θ_enlarge_2 
				x = Morbit.get_x_scaled(id)
				indices = [ Morbit.get_x_index(id, (nl_ind,)) ]
				lb, ub = Morbit.local_bounds( scal, x, θ * Δ_max)
				w = ub .- lb
				for i = 1 : 10*num_vars
					ξ = lb .+ w .* rand(num_vars)
					Morbit.new_result!( db, ξ )
				end
				Morbit._rbf_round4(db, lb, ub, x, Δ, indices, model_cfg)

				## is the model fully linear if we allow to sample enough points?
				if polynomial_degree ==1 
					nl_ind, objf_ind, smop, id, sdb, sc, ac, filter, scal = _initialize(
						model_cfg; num_vars, constrained
					)
					@test Morbit.fully_linear( Morbit.get_surrogates(sc, nl_ind) )
					@test Morbit.fully_linear( Morbit.get_surrogates(sc, objf_ind) )
					@test Morbit.fully_linear( sc )
				end

				## are the values and derivatives correct?
				x = Morbit.get_x_scaled(id)
				x_unscaled = Morbit.get_x(id)
				mod = Morbit.get_surrogates( sc, nl_ind )
				dm = Morbit.get_gradient( mod, scal, x, 1 )
				
				@test Morbit.eval_models( mod, scal, x )[end] ≈ f1(x_unscaled) 
				@test begin 
					dm == vec(Morbit.eval_container_jacobian_at_func_index_at_scaled_site(
						sc, scal, x, nl_ind
					))
				end
				try 
					@test dm ≈ Morbit.AD.gradient( ξ -> Morbit.eval_models(mod, scal, ξ)[end], x  )
				catch
					@info num_vars, kernel, polynomial_degree, constrained
					rethrow()
				end
			end
		end 
	end
end

#%% test if different RbfConfigs lead to the same round1 - round3 points:

model_cfg_1 = RbfConfig(;
	kernel = :gaussian,
)

model_cfg_2 = RbfConfig(;
	kernel = :multiquadric
)

mop = MOP(2)

objf_ind_1 = add_objective!(mop, f1; n_out = 1, model_cfg = model_cfg_1 )
objf_ind_2 = add_objective!(mop, x -> sum( abs.(x) ); n_out = 1, model_cfg = model_cfg_2 )

x0 = rand(2)
algo_config = AlgorithmConfig(;max_evals = 1)
smop, id, sdb, sc, ac, filter, scal = Morbit.initialize_data(mop, x0; algo_config);

nl_ind_1 = Morbit.get_surrogates( sc, objf_ind_1 ).nl_index
nl_ind_2 = Morbit.get_surrogates( sc, objf_ind_2 ).nl_index

db_1 = Morbit.get_sub_db( sdb, (nl_ind_1,))
db_2 = Morbit.get_sub_db( sdb, (nl_ind_2,))
for i = 1 : 20 
	ξ = rand(2)
	Morbit.new_result!( db_1, ξ)
	Morbit.new_result!( db_2, ξ)
end

Morbit.update_surrogates!(sc, smop, scal, id, sdb, ac; ensure_fully_linear = true )

gs_1 = sc.surrogates[1]
gs_2 = sc.surrogates[2]
meta_1 = Morbit.get_meta( gs_1 )
meta_2 = Morbit.get_meta( gs_2 )

for fn = [:round1_indices, :round2_indices, :round3_indices ]
	ind_1 = getfield( meta_1, fn)
	ind_2 = getfield( meta_2, fn)
	@test all( Morbit.get_site( db_1, i1) == Morbit.get_site( db_2, i2 ) for (i1,i2) = zip(ind_1, ind_2) )
end

x = Morbit.get_x_scaled( id )
@test begin 
	Morbit.eval_container_objectives_jacobian_at_scaled_site( sc, scal, x ) ≈
	Morbit.AD.jacobian( ξ -> Morbit.eval_container_objectives_at_scaled_site(sc, scal, ξ), x )
end