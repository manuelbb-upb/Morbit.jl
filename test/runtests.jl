using Morbit
using Test

test_files = ["add_objectives.jl", "lagrange_model.jl", "optimization.jl", "x_stop_function.jl"]
# "rbf_derivatives.jl", "batch_objectives.jl",, "saving.jl"]

for fn in test_files
    include( joinpath(@__DIR__, fn) )   # prefixe with path for PackageCompiler
end
