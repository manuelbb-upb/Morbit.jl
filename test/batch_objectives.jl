using Morbit
using Test

@testset "batch_objectives" begin
global counter_1 = 0;
global counter_2 = 0;

f = x -> sum( x.^2 )

# this function does NOT handle array of vector inputs and uses standard broadcasting
function objf1( x_vec )
    global counter_1
    counter_1 += 1;
    f(x_vec)
end

# this function handles batch evaluation all by itself, 
# and broadcasting should be overwritten internally
function objf2( x_vec_or_array )
    global counter_2
    counter_2 += 1;
    if isa( x_vec_or_array , Vector{<:Real} )
        return f(x_vec_or_array)
    elseif isa(x_vec_or_array , Vector{<:Vector{<:Real}} )
        return [ f(x) for x ∈ x_vec_or_array ]
    end
end

n_vars = 3
sites = [ rand(n_vars) for i = 1 : 5 ];

mop = MixedMOP(n_vars)
add_objective!(mop, objf1, :expensive, 1, false)
add_objective!(mop, objf2, :expensive, 1, true)

Y = Morbit.eval_all_objectives.(mop, sites)

@test counter_1 == 5
@test counter_2 == 1
@test all( y[1] == y[2] for y in Y )
end