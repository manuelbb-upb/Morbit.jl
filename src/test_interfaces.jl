using Morbit 
M = Morbit 
MOI = M.MOI
#%%
mop = M.MixedMOP(2)
f1 = x -> sum( (x .- 1).^2 )
f2 = x -> sum( (x .+ 1).^2 )

M.add_objective!(mop, f1, M.ExactConfig())
M.add_objective!(mop, f2, M.ExactConfig())

algo_config = nothing
populated_db = nothing

x0 = rand(2)
fx0 = []
mop, id, db, sc, ac = M.initialize_data(mop,x0)