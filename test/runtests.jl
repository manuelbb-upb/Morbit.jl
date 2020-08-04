
test_files = ["add_objectives.jl", "rbf_derivatives.jl", "batch_objectives.jl"]
for fn in test_files
    include( joinpath(@__DIR__, fn) )   # prefixe with path for PackageCompiler
end
