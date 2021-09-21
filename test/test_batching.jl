using Test
include(joinpath(@__DIR__, "..", "src", "shorthands.jl"))
#%%

counter = 0

function inner( x :: Vec )
	global counter += 1
	return x
end

function inner( x :: VecVec )
	global counter += 1
	return x
end

#%%

counter = 0
F = VecFuncWrapper( inner; can_batch = true )
F( rand(3) )
F.([rand(2) for i = 1 : 5])

@test counter == 2

counter = 0
Φ = VecFuncWrapper( inner; can_batch = false )
Φ( rand(3) )
Φ.([rand(2) for i = 1 : 5])

@test counter == 6