using Morbit 
M = Morbit 
MOI = M.MOI
#%%
 
Morbit.print_all_logs()

mop = M.MixedMOP(2)
f1 = x -> sum( (x .- 1).^2 )
f2 = x -> sum( (x .+ 1).^2 )

#M.add_objective!(mop, f1, M.ExactConfig())
#M.add_objective!(mop, f2, M.ExactConfig())
M.add_objective!(mop, f1, M.RbfConfig(;use_max_points = true))
M.add_objective!(mop, f2, M.RbfConfig(kernel = :gaussian))
#M.add_objective!(mop, f2, M.RbfConfig())

algo_config = AlgoConfig(;
	max_iter = 3
)
populated_db = nothing

x0 = rand(2)
fx0 = []
#%%
#mop, id, db, sc, ac = M.initialize_data(mop,x0);
#%%
X, FX, ret, db = Morbit.optimize( mop, x0; algo_config )