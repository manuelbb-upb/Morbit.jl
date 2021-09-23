using Morbit
M = Morbit

M.print_all_logs()

mop = M.MOP(3)

vars = M.var_indices(mop)
#=
M.add_lower_bound!( mop, vars[1], -10 )
M.add_lower_bound!( mop, vars[2], -3 )
M.add_upper_bound!( mop, vars[1], 10 )
M.add_upper_bound!( mop, vars[2], 10)
M.add_lower_bound!( mop, vars[3], -10)
M.add_upper_bound!( mop, vars[3], 10)
=#
f1 = x -> sum((x.-1).^2)
f2 = x -> sum((x.+1).^2)
F = x -> [f1(x);f2(x)]

oind = M.add_objective!(mop, f1; model_cfg = M.RbfConfig() )
#oind = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), diff_method =M.AutoDiffWrapper )
#oind2 = M.add_objective!(mop, x -> sum((x.+2).^2); model_cfg = M.RbfConfig() )
#oind2 = M.add_objective!(mop, x -> sum((x.+1).^2); model_cfg = M.TaylorConfig() )
#oind2 = M.add_objective!(mop, x -> sum((x.+1).^2); model_cfg = M.TaylorCallbackConfig(), diff_method = M.AutoDiffWrapper )
oind2 = M.add_objective!(mop, f2; model_cfg = M.ExactConfig(), diff_method = M.AutoDiffWrapper )

#%% manual iteration:
#=
smop, id, sdb, sc, ac = M.initialize_data( mop, ones(2));
M.iterate!(id, sdb, smop, sc, ac)
=#

#%%
# single-call 
x, fx, ret, sdb, id = M.optimize( mop, rand(3) )