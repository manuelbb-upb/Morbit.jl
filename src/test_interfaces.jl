using Morbit
M = Morbit

M.print_all_logs()
n_vars = 2
mop = M.MOP(n_vars)

vars = M.var_indices(mop)

f1 = x -> [ (x[1] - 2)^2 + sum( x[2:end] .- 1 ).^2 ] #x -> sum((x.-1).^2)
f2 = x -> [ (x[1] - 2)^2 + sum( x[2:end] .+ 1 ).^2 ]#x -> sum((x.+1).^2)
F = x -> [f1(x);f2(x)]

oind = M.add_objective!(mop, f1; model_cfg = M.RbfConfig() )
#oind = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), diff_method = M.AutoDiffWrapper )
#oind2 = M.add_objective!(mop, f2; model_cfg = M.RbfConfig() )
#oind2 = M.add_objective!(mop, f2; model_cfg = M.TaylorConfig() )
#oind2 = M.add_objective!(mop, f2; model_cfg = M.TaylorCallbackConfig(), diff_method = M.AutoDiffWrapper )
oind2 = M.add_objective!(mop, f2; model_cfg = M.ExactConfig() )#, diff_method = M.AutoDiffWrapper )

#= ineqconst = M.add_ineq_constraint!(mop, 
	[1 zeros(n_vars-1)'])
=#
ineqconst = M.add_nl_ineq_constraint!(mop, x -> 1 - sum( x.^2 ); model_cfg = M.ExactConfig())


#algo_config = M.AlgoConfig( var_scaler = M.NoVarScaling)

#=
M.add_lower_bound!( mop, vars[1], -10 )
M.add_lower_bound!( mop, vars[2], -3 )
M.add_upper_bound!( mop, vars[1], 10 )
M.add_upper_bound!( mop, vars[2], 10)
M.add_lower_bound!( mop, vars[3], -10)
M.add_upper_bound!( mop, vars[3], 10)
=#

#%% manual iteration:
#=
smop, id, sdb, sc, ac, filter, scal = M.initialize_data( mop, ones(n_vars));
M.iterate!(id, sdb, smop, sc, ac, filter, scal)
=#

#%%
# single-call 
# TODO fix bug when input is int
x, fx, ret, sdb, id = M.optimize( mop, [-2.0; 0.0]; algo_config = M.AlgoConfig(max_iter = 35))
x