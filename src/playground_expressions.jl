using Morbit
M = Morbit
#%%
#mop = MOP(zeros(2), ones(2))
mop = MOP(2)
F = x -> sum(x)
o1 = add_objective!(mop, F; model_cfg = ExactConfig(), n_out = 1, diff_method = AutoDiffWrapper)

G = x -> sum(x.^2)
VG = M.make_vec_fun( G; n_out = 1, model_cfg = RbfConfig() )
#VG = M.make_vec_fun( G; n_out = 1, model_cfg = ExactConfig() )
gind = M._add_function!( mop, VG )
o2 = M._add_objective!(mop, gind)

sin_vfunc = M.make_vec_fun( x -> sin(x[end]); n_out = 1, model_cfg = ExactConfig(), diff_method = AutoDiffWrapper )
o3 = M._add_objective!(mop, gind, sin_vfunc)

o4 = M._add_objective!(mop, gind, "(sin(VREF(x)[1]))", 1)
#=
T = x -> sin(x)
M.register_func(T, :T)

o5 = M._add_objective!(mop, VG, "T(VREF(x)[1])", 1)
o5 = M._add_objective!(mop, gind, "T(VREF(x)[1])", 1)

for ind = [o1,o2,o3,o4,o5]
	@show M.eval_vfun( M._get(mop, ind), ones(2) )
end
=#

#=
objf1 = M._get( mop, o1 )
@show M.eval_vfun( objf1, ones(2) )
@show M._get_gradient( objf1, ones(2), 1)

objf2 = M._get( mop, o2 )
@show M.eval_vfun( objf2, ones(2) )
=#
#@show M._get_gradient( objf2, ones(2), 1)
#=
objf3 = M._get( mop, o3 )
@show M.eval_vfun( objf3, ones(2) )
@show M._get_gradient( objf3, ones(2), 1)

objf4 = M._get( mop, o4 )
@show M.eval_vfun( objf4, ones(2) )
@show M._get_gradient( objf4, ones(2), 1)
=#
#%%
x0 = rand(2)
smop, id, sdb, sc, ac, filter, scal = M.initialize_data(mop, x0);
#%%
M.optimize(mop,x0; max_iter = 10,verbosity=4, f_tol_rel = 1e-8, x_tol_rel = 1e-8)