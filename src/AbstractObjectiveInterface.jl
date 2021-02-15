abstract type AbstractObjective end;

num_evals( :: AbstractObjective ) = 0 :: Int;
max_evals( :: AbstractObjective) = typemax(Int) :: Int;

# set evaluation counter to `N`
num_evals!( :: AbstractObjective, N :: Int) = nothing :: Nothing;
# increase evaluation count
inc_evals!( :: AbstractObjective ) = nothing :: Nothing;
# increase evaluation count by N
inc_evals!( :: AbstractObjective, N :: Int ) = nothing :: Nothing;

# set upper bound of № evaluations to `N`
max_evals!( :: AbstractObjective, N :: Int ) = nothing :: Nothing;

base_function( :: AbstractObjective ) = nothing :: Function;