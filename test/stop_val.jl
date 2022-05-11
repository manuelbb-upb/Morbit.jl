module testStopVal

using Morbit
using Test
M = Morbit

#%%
n_vars = 2 

mop = M.MOP(n_vars)

f1 = x -> sum(x.^2)

oind1 = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), n_out = 1) #LagrangeConfig(; optimized_sampling=true) )
x0 = fill(1.0, n_vars) # => f1(x0) == 2

#%%
algo_config = AlgorithmConfig{Float64}(;
	stop_value_dict = Dict( oind1 => 3.0 )
)

x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)


@test x == x0 
@test fx[end] <= 3.0

#%%
algo_config = AlgorithmConfig{Float64}(;
	stop_value_dict = Dict( oind1 => [2.0] )
)

x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)

@test x == x0 
@test fx[end] <= 2.0

#%%
# linear constraint: y <= x - 1 ⇔ x[2] - x[1] <= - 1 ⇔ x[2] - x[1] + 1 <= 0
# c(x0) == 1
c = M.add_ineq_constraint!(mop, [-1 1], [-1,])
# without including c in the stop val dict, we stop at infeasible x0:
x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)

@test x == x0 

#%%
# inlusion of constraint into stop val test
algo_config = AlgorithmConfig{Float64}(;
	stop_value_dict = Dict(
		oind1 => [2.0],
		c => .5
	)
)

x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)
@test x != x0
@test id.l_i[end] <= .5

#%%
# similarly, the option `stop_val_only_if_feasible`
# can be set to respect the values only if `x` is feasible
mop = M.MOP(n_vars)
oind1 = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), n_out = 1) #LagrangeConfig(; optimized_sampling=true) )
c = M.add_ineq_constraint!(mop, [-1 1], [-1,])

algo_config = AlgorithmConfig{Float64}(;
	stop_value_dict = Dict( oind1 => [2.0] ),
	stop_val_only_if_feasible = true,
)

x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)
@test fx[end] <= 2
@test id.l_i[end] <= 1e-5

#%%
g = x -> sum( (x .- 2 ).^2 ) - 1
mop = M.MOP(n_vars)
oind1 = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), n_out = 1) #LagrangeConfig(; optimized_sampling=true) )
c = M.add_nl_ineq_constraint!(mop, g; model_cfg = M.ExactConfig(), n_out = 1)
algo_config = AlgorithmConfig{Float64}(;
	stop_value_dict = Dict(
		oind1 => [2.0],
		c => 1
	)
)

x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)

@test x0 == x

#%%
g = x -> sum( (x .- 2 ).^2 ) - 1
mop = M.MOP(n_vars)
oind1 = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), n_out = 1) #LagrangeConfig(; optimized_sampling=true) )
c = M.add_nl_ineq_constraint!(mop, g; model_cfg = M.ExactConfig(), n_out = 1)
algo_config = AlgorithmConfig{Float64}(;
	stop_value_dict = Dict(
		oind1 => [4.0],
		c => .5
	)
)

x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 0, algo_config)

if ret == M.STOP_VAL
	@test fx[end] <= 4
	@test id.c_i[end] <= .5
end

end

using .testStopVal