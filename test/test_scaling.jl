module ScalingTests
using Morbit 
using Test 

using LinearAlgebra: Diagonal 

# the current default scaling behavior is as follows
# if a problem is fully finitely box constrained (*) then all 
# variables are scaled to [0,1]^n 
# if (*) does not hold, then no scaling happens.

# a user can specify a custom scaling `AbstractVarScaler` object 
# in the `var_scaler` field of `AlgorithmConfig`

# there also is a dumb `:auto` option 
# (‘dumb’ because it is only sensible for linear functions)
# in this case, the jacobian `J` is estimated via finite differences
# and we try to find variable scaling factors `c` to bring 
# the absolute values of `J⋅diagm(c)` as close to one as possible.
# Likewise, there is the option to update scaling factors in each iteration 
# (which is also deactivated by default via `var_scaler_update = :none`)

@testset "Unconstrained Problem => No scaling" begin
mop = MOP(2)
add_objective!(mop, x -> sum(x))
x0 = rand(2)
smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0);

@test scal isa Morbit.NoVarScaling
@test x0 == Morbit.get_x( iter_data )
@test x0 == Morbit.get_x_scaled( iter_data )
@test all( isinf.( Morbit.full_lower_bounds_internal( scal ) ) )
@test all( isinf.( Morbit.full_upper_bounds_internal( scal ) ) )
end

@testset "Constrained Problem => Unit Scaling" begin

lb = fill(-5, 2)
ub = fill(6, 2)

unit_scaling_fn = x -> (x .- lb) ./ (ub .- lb )

mop = MOP(lb, ub )
o_ind = add_objective!(mop, x -> sum(x))
x0 = lb .+ (ub .- lb ) .* rand(2)
x0_scaled = unit_scaling_fn(x0)
smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0);

@test scal isa Morbit.LinearScaling
@test x0 == Morbit.get_x( iter_data )
@test x0_scaled ≈ Morbit.get_x_scaled( iter_data )

@test all( abs.( Morbit.full_lower_bounds_internal( scal ) ) .<= 1e-10 )
@test all( abs.( Morbit.full_upper_bounds_internal( scal ) .- 1 ) .<= 1e-10) 

sdb = Morbit.get_sub_db( data_base, (o_ind,) )

@test all( all(0 .<= x .<= 1) for x = Morbit.get_sites(sdb) )

xf, _ = optimize( smop, x0; max_iter = 0 )

@test xf ≈ x0

end

@testset "Constrained Problem, User Scaling" begin

lb = fill(-5, 2)
ub = fill(5, 2)

factors = 1 ./ (2 .* (ub .- lb) )
scaler = Morbit.LinearScaling(lb, ub, Diagonal( factors ), (- lb .* factors) )

ac = Morbit.AlgorithmConfig(; var_scaler = scaler )

mop = MOP(lb, ub )
o_ind = add_objective!(mop, x -> sum(x))
x0 = lb .+ (ub .- lb ) .* rand(2)
x0_scaled = Morbit.transform( x0, scaler )

smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0;
	algo_config = ac)

@test scal == scaler
@test x0 == Morbit.get_x( iter_data )
@test x0_scaled ≈ Morbit.get_x_scaled( iter_data )

@test all( abs.( Morbit.full_lower_bounds_internal( scal ) ) .<= 1e-10 )
@test all( abs.( Morbit.full_upper_bounds_internal( scal ) .- .5 ) .<= 1e-10) 

sdb = Morbit.get_sub_db( data_base, (o_ind,) )

@test all( all(0 .<= x .<= .5) for x = Morbit.get_sites(sdb) )

xf, _ = optimize( smop, x0; max_iter = 0 )

@test xf ≈ x0
end

@testset "Unconstrained Problem, User Scaling" begin
scaler = Morbit.LinearScaling( fill(-Inf, 2), fill(Inf, 2), Diagonal([1, 2]), zeros(2) )

ac = Morbit.AlgorithmConfig(; var_scaler = scaler )

mop = MOP(2)
o_ind = add_objective!(mop, x -> sum(x))
x0 = ones(2)
x0_scaled = Morbit.transform( x0, scaler )

@test x0_scaled ≈ [1, 2]

smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0;
	algo_config = ac);

@test scal == scaler
@test x0 == Morbit.get_x( iter_data )
@test x0_scaled ≈ Morbit.get_x_scaled( iter_data )

@test all( isinf.( Morbit.full_lower_bounds_internal( scal ) ) )
@test all( isinf.( Morbit.full_upper_bounds_internal( scal ) ) ) 

xf, _ = optimize( smop, x0; max_iter = 0 )

@test xf ≈ x0
end

@testset "Scaling changes" begin 

scaler = Morbit.LinearScaling( fill(-Inf, 2), fill(Inf, 2), Diagonal([1, 2]), zeros(2) )

ac = Morbit.AlgorithmConfig(; var_scaler = scaler, var_scaler_update = :model )

mop = MOP(2)
o_ind = add_objective!(mop, x -> sum(x))
x0 = ones(2)
x0_scaled = Morbit.transform( x0, scaler )

@test x0_scaled ≈ [1, 2]

smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0;
	algo_config = ac);

@test scal == scaler
@test x0 == Morbit.get_x( iter_data )
@test x0_scaled ≈ Morbit.get_x_scaled( iter_data )

_, _, scal2, _ =  Morbit.iterate!(iter_data, data_base, smop, sc, ac, filter, scal; iter_counter = 2)

@test scal2 != scal 

end

end#module 

using .ScalingTests