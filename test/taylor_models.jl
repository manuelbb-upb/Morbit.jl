module TaylorTests
#%%
using ForwardDiff
using Morbit
using Test

using LinearAlgebra: norm

const RFD = Morbit.RFD 
const Trees = RFD.Trees 

flatten_vecs( x ) = vec(x)
flatten_vecs( x :: Number ) = [x,]

const test_funcs = [
	x -> [x[1],],
	x -> [π + ℯ * sum(x.^2),],
	x -> x.^2,
	x -> [ exp(sum(x)), x[1] + x[end]^3 ]
]

const test_stamps64 = [
	RFD.CFDStamp(1,2),
	RFD.CFDStamp(1,4),
	RFD.CFDStamp(1,6),
	RFD.FFDStamp(1,1),
	RFD.FFDStamp(1,2),
	RFD.FFDStamp(1,3),
	RFD.BFDStamp(1,1),
	RFD.BFDStamp(1,2),
	RFD.BFDStamp(1,3),
]

h32 = RFD.stepsize(RFD.stepsize(Float32))
const test_stamps32 = [
	RFD.CFDStamp(1,2, h32),
	RFD.CFDStamp(1,4, h32),
	RFD.CFDStamp(1,6, h32),
	RFD.FFDStamp(1,1, h32),
	RFD.FFDStamp(1,2, h32),
	RFD.FFDStamp(1,3, h32),
	RFD.BFDStamp(1,1, h32),
	RFD.BFDStamp(1,2, h32),
	RFD.BFDStamp(1,3, h32),
]

h16 = RFD.stepsize(RFD.stepsize(Float16))
const test_stamps16 = [
	RFD.CFDStamp(1,2, h16),
	RFD.CFDStamp(1,4, h16),
	RFD.CFDStamp(1,6, h16),
	RFD.FFDStamp(1,1, h16),
	RFD.FFDStamp(1,2, h16),
	RFD.FFDStamp(1,3, h16),
	RFD.BFDStamp(1,1, h16),
	RFD.BFDStamp(1,2, h16),
	RFD.BFDStamp(1,3, h16),
]
#%%

@testset "Recursive Finite Difference Trees" begin
	for ET in [Float16, Float32, Float64]
		stamps = ET == Float16 ? test_stamps16 : ET == Float32 ? test_stamps32 : test_stamps64

		for (j,func) in enumerate(test_funcs)
			n_in = rand(1:5)
			x0 = flatten_vecs(rand(ET, n_in))	
			fx0 = func(x0)
			for stamp in stamps
				for order in [1,2]
					# first, use convenience function to fill tree
					dw_1 = RFD.DiffWrapper(; x0, fx0, stamp, order)
					RFD.prepare_tree!(dw_1, func)

					# secondly, use the Morbit routine
					dw_2 = RFD.DiffWrapper(; x0, fx0, stamp, order)
					RFD.substitute_leaves!(dw_2)
					all_sites = RFD.collect_leave_sites( dw_2 )
					all_vals = func.(all_sites)
					RFD.set_leave_values!(dw_2, all_vals)

					if order > 1 
						for i = 1 : length(fx0)
							Hi = RFD.hessian(dw_1; output_index = i)
							@test Hi ≈ RFD.hessian( dw_2; output_index = i)
							# I don't test against automatic differencing because 
							# finite diff hessians become somewhat instable without careful tuning 
							# of stepsize …
							# H_ad = ForwardDiff.hessian( x -> func(x)[i], x0 )
							# ET == Float16 || @test norm( Hi .- H_ad ) <= .1
						end
					end

					J1 = RFD.jacobian(dw_1)

					for i = 1 : length(fx0)
						Gi = RFD.gradient( dw_1; output_index = i )
						@test Gi ≈ RFD.gradient( dw_2; output_index = i)
						@test Gi ≈ J1[i,:]
					end

					# does it roughly compare to a jacobian determined by automatic differencing
					J_ad = ForwardDiff.jacobian( func, x0 )
					ET == Float16 || @test norm( J1 .- J_ad ) <= 0.1
				
				end
			end
		end
	end

end#testset

@testset "Accurate Linear Models, Unconstrained" begin
	mop = Morbit.MOP(2)
	
	# The gradients for a linear objective will be exact and a linear model 
	# should then equal the linear objective globally
	Morbit.add_objective!( mop, x -> sum(x); model_cfg = Morbit.TaylorConfig(;degree=1) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0)
#	Morbit.update_surrogates!( sc, mop, scal, iter_data, data_base, ac )

	mod = Morbit.get_model(Morbit.list_of_wrappers(sc)[1])
	objf = Morbit.list_of_objectives(smop)[1]

	@test Morbit.eval_models( mod, scal, x0 ) ≈ Morbit.eval_objf(objf,x0)
	@test Morbit.get_gradient( mod, scal, x0, 1 ) ≈ [1, 1]
	@test Morbit.get_jacobian( mod, scal, x0) ≈ [ 1 1 ]
	@test Morbit.eval_container_objectives_jacobian_at_scaled_site( sc, scal, x0 ) ≈ [1 1]
end#testset

@testset "Accurate Linear Models, Constrained" begin
	mop = Morbit.MOP( fill(-10.0, 2), fill(10.0, 2) )

	objf_ind = Morbit.add_objective!( mop, x -> sum(x); model_cfg = Morbit.TaylorConfig(;degree=1) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0)
	#Morbit.update_surrogates!( sc, mop, iter_data, data_base, ac )

	mod = Morbit.get_model(Morbit.list_of_wrappers(sc)[1])
	objf = Morbit.list_of_objectives(smop)[1]

	x̂0 = Morbit.transform( x0, scal )

	@test Morbit.eval_models( mod, scal, x̂0 ) ≈ Morbit.eval_objf(objf, x0 )
	@test Morbit.get_gradient( mod, scal, x̂0, 1 ) ≈ ForwardDiff.gradient( ξ -> Morbit.eval_objf(objf, Morbit.untransform(ξ, scal) )[end], x̂0 )
	@test Morbit.get_jacobian( mod, scal, x̂0 ) ≈ ForwardDiff.jacobian( ξ -> Morbit.eval_vec_mop_at_func_indices_at_scaled_site(smop, [objf_ind,], ξ, scal), x̂0 )

end#testset


@testset "Quadratic Models, Unconstrained " begin
	mop = Morbit.MOP(2)
	
	# The gradients for a linear objective will be exact and a linear model 
	# should then equal the linear objective globally
	Morbit.add_objective!( mop, x -> sum(x.^2); model_cfg = Morbit.TaylorConfig(;degree=1) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0)

	mod = Morbit.get_model(Morbit.list_of_wrappers(sc)[1])
	objf = Morbit.list_of_objectives(smop)[1]

	@test Morbit.eval_models( mod, scal, x0 ) ≈ Morbit.eval_objf(objf,x0)
	@test Morbit.get_gradient( mod, scal, x0, 1 ) ≈ 2 .* x0
	@test Morbit.get_jacobian(mod, scal, x0) ≈ 2 .* x0'
end#testset

@testset "Two Parabolas, Taylor Models, 3D2D" begin
	mop = MOP(3)

	add_objective!( mop, x -> sum( ( x .- 1 ).^2 ); model_cfg = Morbit.TaylorCallbackConfig(;degree=2), diff_method = Morbit.FiniteDiffWrapper )
	add_objective!( mop, x -> sum( ( x .+ 1 ).^2 ); model_cfg = Morbit.TaylorCallbackConfig(;degree=2), diff_method = Morbit.AutoDiffWrapper  )

	x_fin, f_fin, _ = optimize( mop, [-π, ℯ, 0]; f_tol_rel = 1e-5, x_tol_rel = 1e-6 )

	@test isapprox(x_fin[1],x_fin[2], atol = 1e-2) && isapprox(x_fin[2], x_fin[3]; atol = 1e-2) 

	mop = MOP(3)

	add_objective!( mop, x -> sum( ( x .- 1 ).^2 ); 
		model_cfg = TaylorCallbackConfig(; degree = 1), 
		gradients = [ x -> 2 .* ( x .- 1 ), ] 
	)
	add_objective!( mop, x -> sum( ( x .+ 1 ).^2 ); 
		model_cfg = TaylorCallbackConfig(; degree=2), 
		gradients = [ x -> 2 .* ( x .+ 1 ), ],
		hessians = [ x -> [ 2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0] ] 
	)

	x_fin, f_fin, _ = optimize( mop, [-π, ℯ, 0]; f_tol_rel = 1e-5, x_tol_rel = 1e-6 )

	@test isapprox(x_fin[1],x_fin[2], atol = 1e-2) && isapprox(x_fin[2], x_fin[3]; atol = 1e-2) 

end#testset

@testset "Two Parabolas, Derivative Accuracy" begin
	# deg 1 autodiff
	mop = Morbit.MixedMOP( 2 )

	Morbit.add_objective!( mop, x -> sum(x), Morbit.TaylorApproximateConfig(;degree=1, mode=:autodiff) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )
#	Morbit.update_surrogates!( sc, mop, iter_data, data_base, ac )

	mod = sc.surrogates[1].model
	objf = Morbit.list_of_objectives(smop)[1]

	x̂0 = Morbit.scale( x0, smop)

	@test Morbit.eval_models( mod, x0 ) ≈ Morbit.eval_objf(objf,x0)
	@test Morbit.get_gradient( mod, x0, 1 ) ≈ [1, 1]
	@test Morbit.get_jacobian(sc, x0) ≈ [ 1 1 ]

	#===================================================================#
	# deg 2 autodiff
	mop = Morbit.MixedMOP( 2 )

	Morbit.add_objective!( mop, x -> sum(x.^2), Morbit.TaylorApproximateConfig(;degree=2, mode=:autodiff) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )
	# Morbit.update_surrogates!( sc, mop, iter_data, data_base, ac )

	mod = sc.surrogates[1].model
	objf = Morbit.list_of_objectives(smop)[1]

	x̂0 = Morbit.scale( x0, smop)

	@test Morbit.eval_models( mod, x0 ) ≈ Morbit.eval_objf(objf,x0)
	@test Morbit.get_gradient( mod, x0, 1 ) ≈ 2 .* x0
	@test Morbit.get_jacobian(sc, x0) ≈ 2 .* x0'
	@test mod.H[1] ≈ [ 2.0 0.0; 0.0 2.0 ]	# hessian of Taylor Model

	#===================================================================#
	# deg 1 finite diff
	mop = Morbit.MixedMOP( 2 )

	Morbit.add_objective!( mop, x -> sum(x), Morbit.TaylorApproximateConfig(;degree=1, mode=:fdm) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )
	#Morbit.update_surrogates!( sc, mop, iter_data, data_base, ac )

	mod = sc.surrogates[1].model
	objf = Morbit.list_of_objectives(smop)[1]

	x̂0 = Morbit.scale( x0, smop)

	@test Morbit.eval_models( mod, x0 ) ≈ Morbit.eval_objf(objf,x0)
	@test Morbit.get_gradient( mod, x0, 1 ) ≈ [1, 1]
	@test Morbit.get_jacobian(sc, x0) ≈ [ 1 1 ]

	#===================================================================#
	# deg 2 autodiff
	mop = Morbit.MixedMOP( 2 )

	Morbit.add_objective!( mop, x -> sum(x.^2), Morbit.TaylorApproximateConfig(;degree=2, mode=:fdm) )

	x0 = [π, -ℯ]

	smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )
#	Morbit.update_surrogates!( sc, mop, iter_data, data_base, ac )

	mod = sc.surrogates[1].model
	objf = Morbit.list_of_objectives(smop)[1]

	x̂0 = Morbit.scale( x0, smop)

	@test Morbit.eval_models( mod, x0 ) ≈ Morbit.eval_objf(objf,x0)
	@test Morbit.get_gradient( mod, x0, 1 ) ≈ 2 .* x0
	@test Morbit.get_jacobian(sc, x0) ≈ 2 .* x0'
	@test norm(mod.H[1] .- [ 2.0 0.0; 0.0 2.0 ]) <= 0.1	# hessian of Taylor Model
	
end

end#module

using .TaylorTests