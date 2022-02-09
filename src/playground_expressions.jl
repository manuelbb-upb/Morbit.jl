using Morbit
M = Morbit 

#%%
mop = M.MOP(2)

F = x -> sum(x)
VF = M._wrap_func( M.VecFun, F; n_out = 1, model_cfg = M.ExactConfig() )
o1 = M._add_objective!(mop, VF)

G = x -> sum(x.^2)
VG = M._wrap_func( M.VecFun, G; n_out = 1, model_cfg = M.ExactConfig() )
gind = M._add_function!( mop, VG )
o2 = M._add_objective!(mop, gind)
o3 = M._add_objective!(mop, gind, "(sin(VREF(x)[1]))", 1)
#%%
objf1 = M._get( mop, o1 )
@show M.eval_objf( objf1, ones(2) )

objf2 = M._get( mop, o2 )
@show M.eval_objf( objf2, ones(2) )

objf3 = M._get( mop, o3 )
@show M.eval_objf( objf3, ones(2) )