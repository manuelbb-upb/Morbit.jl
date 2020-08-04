using Morbit
using Test

counter_1 = 0;
counter_2 = 0;

# this function does NOT handle array of vector inputs and uses standard broadcasting
function objf1( x_vec )
    global counter_1
    counter_1 += 1;
    sum(x_vec.^2);
end

# this function handles looping all by itself, and broadcasting should be overwritten internally
function objf2( x_vec_or_array )
    global counter_2
    counter_2 += 1;
    ret_array = [ sum(x_vec.^2) for x_vec ∈ x_vec_or_array ];
    if length(ret_array) == 1
        return ret_array[end];
    else
        return ret_array
    end
end

sites = [ rand(3) for i = 1 : 5 ];

o1 = Morbit.ObjectiveFunction( objf1 )
o2 = Morbit.BatchObjectiveFunction( objf2 )

y1 = o1.(sites)
y2 = o2.(sites)

@test counter_1 == 5
@test counter_2 == 1

@test y1 == y2

# check internal broadcasting mechanism

counter_1 = 0
counter_2 = 0

mop = MixedMOP()
add_objective!(mop, objf1, :expensive, 1, false)
add_objective!(mop, objf2, :expensive, 1, true)

Y = Morbit.eval_expensive_objectives.(mop, sites)

@test counter_1 == 5
@test counter_2 == 1
@test all( y[1] == y[2] for y in Y )
