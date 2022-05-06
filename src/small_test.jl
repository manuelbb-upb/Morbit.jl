using Morbit
const M = Morbit

#%%
n_vars = 2 

mop = M.MOP(n_vars)

vars = M.var_indices(mop)

f1 = x -> sum((x.-1).^2)
f2 = x -> sum((x.+1).^2)
F = x -> [f1(x);f2(x)]

oind1 = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), n_out = 1) #LagrangeConfig(; optimized_sampling=true) )
oind2 = M.add_nl_ineq_constraint!(mop, f2; model_cfg = M.ExactConfig(), n_out = 1)#, diff_method = M.AutoDiffWrapper )
#%%
x0 = [-6.0, 10.0]
x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 4)