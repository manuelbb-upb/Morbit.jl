module BatchEval

using Morbit
using Test 
#%%
mutable struct CountedFn{F} <: Function
	f :: F
	counter :: Int 

	CountedFn( f :: F ) where F = new{F}(f, 0)
end

function (F::CountedFn)(args...)
	F.counter += 1
	return F.f(args...)
end

batchable_fn( x :: AbstractVector{<:Real} ) = sum(x)
batchable_fn( x :: AbstractVector{<:AbstractVector{<:Real}}) = batchable_fn.(x)

#%%

@testset "Batch Evaluation (Static MOP)" begin
batchable_FN = CountedFn(batchable_fn)

mop = MOP(2);
objf_ind = add_objective!(mop, batchable_FN; model_cfg = Morbit.ExactConfig(), can_batch = true)

x0 = rand(2);
smop, iter_data, data_base, sc, ac, filter, scal = Morbit.initialize_data(mop, x0)

batchable_FN.counter = 0
stored_objf = Morbit._get( smop, objf_ind )

Morbit.eval_objf( stored_objf, rand(2) )
@test batchable_FN.counter == 1
Morbit.eval_objf( stored_objf, rand(2) )
@test batchable_FN.counter == 2

# test it broadcasting works as overwritten in Morbit
Morbit.eval_objf.( stored_objf, [rand(2) for i = 1 : 3])
@test batchable_FN.counter == 3

x = rand(2)
X = [rand(2) for i = 1 : 3]

scal = Morbit.NoVarScaling( rand(2), rand(2) )
Morbit.eval_vec_mop_at_func_indices_at_unscaled_site( mop, [objf_ind,], x )
@test batchable_FN.counter == 4
Morbit.eval_vec_mop_at_func_indices_at_scaled_site( mop, [objf_ind,], x, scal )
@test batchable_FN.counter == 5

Morbit.eval_vec_mop_at_func_indices_at_unscaled_sites( mop, [objf_ind,], X )
@test batchable_FN.counter == 6
Morbit.eval_vec_mop_at_func_indices_at_scaled_sites( mop, [objf_ind,], X, scal )
@test batchable_FN.counter == 7
end#testset 

@testset "Batch Evaluation (True MOP)" begin
batchable_FN = CountedFn(batchable_fn)

mop = MOP(2);
objf_ind = add_objective!(mop, batchable_FN; model_cfg = Morbit.ExactConfig(), can_batch = true)

stored_objf = Morbit._get( mop, objf_ind );

Morbit.eval_objf( stored_objf, rand(2) )
@test batchable_FN.counter == 1
Morbit.eval_objf( stored_objf, rand(2) )
@test batchable_FN.counter == 2

# test it broadcasting works as overwritten in Morbit
Morbit.eval_objf.( stored_objf, [rand(2) for i = 1 : 3])
@test batchable_FN.counter == 3

x = rand(2)
X = [rand(2) for i = 1 : 3]

scal = Morbit.NoVarScaling( rand(2), rand(2) )
Morbit.eval_vec_mop_at_func_indices_at_unscaled_site( mop, [objf_ind,], x )
@test batchable_FN.counter == 4
Morbit.eval_vec_mop_at_func_indices_at_scaled_site( mop, [objf_ind,], x, scal )
@test batchable_FN.counter == 5

Morbit.eval_vec_mop_at_func_indices_at_unscaled_sites( mop, [objf_ind,], X )
@test batchable_FN.counter == 6
Morbit.eval_vec_mop_at_func_indices_at_scaled_sites( mop, [objf_ind,], X, scal )
@test batchable_FN.counter == 7
end#testset 

# TODO batch evaluation of gradients!
end#module

using .BatchEval

