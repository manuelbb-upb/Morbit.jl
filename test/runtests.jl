using Morbit
using Test
using Random
Random.seed!(1234)
#%%
test_files = ["taylor_models.jl", "test_batching.jl", "test_scaling.jl", "rbf_models.jl"] # "lagrange_models.jl", "model_derivatives.jl"]
for fn in test_files
    include( joinpath(@__DIR__, fn) )   # prefixe with path for PackageCompiler
end
