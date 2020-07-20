
test_files = ["add_objectives.jl"]
for fn in test_files
    include( joinpath(@__DIR__, fn) )
end
