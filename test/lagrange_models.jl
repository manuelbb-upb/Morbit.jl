using ForwardDiff
using Morbit
using Test
#%%
mop = Morbit.MixedMOP(2)

# The gradients for a linear objective will be exact and a linear model 
# should then equal the linear objective globally
Morbit.add_objective!( mop, x -> sum(x), Morbit.LagrangeConfig(;degree=1, LAMBDA=10) )

x0 = [π, -ℯ]

smop, iter_data, data_base, sc, ac = Morbit.initialize_data(mop, x0, Float32[] )
#Morbit.update_surrogates!( sc, mop, iter_data, data_base, ac )

mod = sc.surrogates[1].model
objf = Morbit.list_of_objectives(smop)[1]

@test Morbit.eval_models( mod, x0 )[end] ≈ sum(x0)