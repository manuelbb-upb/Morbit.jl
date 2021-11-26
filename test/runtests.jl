using Morbit
using Test

#%%

test_files = ["taylor_models.jl",] # "lagrange_models.jl", "model_derivatives.jl"]
for fn in test_files
    include( joinpath(@__DIR__, fn) )   # prefixe with path for PackageCompiler
end
