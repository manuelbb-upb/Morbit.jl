using Morbit
M = Morbit

M.print_all_logs()

mop = M.MOP(2)

vars = M.var_indices(mop)

#=
M.add_lower_bound!( mop, vars[1], -10 )
M.add_lower_bound!( mop, vars[2], -3 )
M.add_upper_bound!( mop, vars[1], 10 )
M.add_upper_bound!( mop, vars[2], 10)
=#

#oind = M.add_objective!(mop, x -> sum((x.-1).^2); model_cfg = M.RbfConfig() )
#oind2 = M.add_objective!(mop, x -> sum((x.+1).^2); model_cfg = M.RbfConfig(;kernel = :gaussian) )
oind2 = M.add_objective!(mop, x -> sum((x.+1).^2); model_cfg = M.TaylorConfig() )

#%% manual iteration:
#=
mop, id, sdb, sc, ac = M.initialize_data( mop, ones(2));
M.iterate!(id, sdb, mop, sc, ac)
=#

#%%
# single-call 
x, fx, ret, sdb, id = M.optimize( mop, ones(2))