using Morbit
M = Morbit
#%%
#mop = MOP(zeros(2), ones(2))
mop = MOP(2)
F = x -> sum(x)
o1 = add_objective!(mop, F; model_cfg = ExactConfig(), n_out = 1, diff_method = AutoDiffWrapper)

G = x -> sum(x.^2)
VG = M.make_vec_fun( G; n_out = 1, model_cfg = RbfConfig(kernel = :multiquadric) )
#VG = M.make_vec_fun( G; n_out = 1, model_cfg = ExactConfig() )
gind = M._add_function!( mop, VG )
#o2 = M._add_objective!(mop, gind)

o3 = M._add_objective!(mop, gind, "(sin(VREF(x)[1]))", 1)

T = x -> sin(x)
M.register_func(T, :T)

#o4 = M._add_objective!(mop, VG, "T(VREF(x)[1])", 1)
o4 = M._add_objective!(mop, gind, "T(VREF(x)[1])", 1)

#%%
#=
objf1 = M._get( mop, o1 )
@show M.eval_objf( objf1, ones(2) )
@show M._get_gradient( objf1, ones(2), 1)

objf2 = M._get( mop, o2 )
@show M.eval_objf( objf2, ones(2) )
=#
#@show M._get_gradient( objf2, ones(2), 1)
#=
objf3 = M._get( mop, o3 )
@show M.eval_objf( objf3, ones(2) )
@show M._get_gradient( objf3, ones(2), 1)

objf4 = M._get( mop, o4 )
@show M.eval_objf( objf4, ones(2) )
@show M._get_gradient( objf4, ones(2), 1)
=#
#%%
x0 = rand(2)
smop, id, sdb, sc, ac, filter, scal = M.initialize_data(mop, x0);
#%%
M.optimize(mop,x0; max_iter = 10,verbosity=4, f_tol_rel = 1e-8, x_tol_rel = 1e-8)