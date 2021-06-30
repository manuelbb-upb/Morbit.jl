using Morbit 
M = Morbit 

#%%
mop = M.MixedMOP(2)
f = x -> rand([Float16,Float32]).(x)
vo = M._wrap_func( M.VectorObjectiveFunction, f, M.ExactConfig(), 2, 1 )
vo2 =M._wrap_func( M.VectorObjectiveFunction, f, M.ExactConfig(;gradients = [ x -> 1]), 2, 1 )

M._add!(mop, vo)
M._add!(mop, vo2)

smop = M.StaticMOP(mop)