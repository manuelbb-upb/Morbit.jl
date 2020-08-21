using Morbit
using Test

fname = string( tempname(), ".jld" )
c = AlgoConfig()
p = save_config(c, fname)

@test p == fname

C = load_config( p )

@test isa( C , AlgoConfig )
rm(fname)
