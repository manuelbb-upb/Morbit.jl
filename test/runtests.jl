using Morbit
using Test

#%%
#test_files = ["optimization.jl", "batch_objectives.jl",]

test_files = ["taylor_models.jl"]
for fn in test_files
    include( joinpath(@__DIR__, fn) )   # prefixe with path for PackageCompiler
end
