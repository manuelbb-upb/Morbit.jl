using Test
import ForwardDiff as AD
import FiniteDiff as FD 
using Parameters: @with_kw
include(joinpath(@__DIR__, "..", "src", "shorthands.jl"))
#%%
include(joinpath(@__DIR__, "..", "src", "diff_wrappers.jl"))
#%%

F = x -> [ exp(sum(x.^2)); sum(x.^(1:length(x))) ]
∇F1 = x -> AD.gradient( ξ -> F(ξ)[1], x )
∇F2 = x -> AD.gradient( ξ -> F(ξ)[2], x )
J = x -> AD.jacobian( F, x )
H1 = x -> AD.hessian( ξ -> F(ξ)[1], x )
H2 = x -> AD.hessian( ξ -> F(ξ)[2], x )

mutable struct CountedFN{F} <: Function
	func :: F
	counter :: Int
	CountedFN( func  :: F) where F = new{F}(func,0)
end
function(FN::CountedFN)(args...)
	FN.counter += 1
	return FN.func(args...)
end

Fc = CountedFN(F)
∇F1c = CountedFN(∇F1)
∇F2c = CountedFN(∇F2)
Jc = CountedFN(J)
H1c = CountedFN(H1)	
H2c = CountedFN(H2)	

function _reset_counter!( args... )
	for f in args 
		f.counter = 0
	end
	return nothing
end

x0 = rand(2)

@testset "CallbackGradsNoJacobian" begin 

	w = CallBackWrapper(;
		objf = F,
		gradients = [∇F1c,∇F2c]
	)
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)

	@test get_gradient( w, x0, 1 ) == ∇F1(x0)
	@test ∇F1c.counter == 1
	@test get_gradient( w, x0, 2 ) == ∇F2(x0)
	@test ∇F2c.counter == 1
	# we have not given a jacobian handle in w1, so it is built from the gradients:
	@test get_jacobian( w, x0 ) == J(x0)
	@test ∇F1c.counter == ∇F2c.counter == 2
end


@testset "CallbackGradsAndJacobian" begin

	w = CallBackWrapper(;
		objf = F,
		gradients = [∇F1c,∇F2c],
		jacobian = Jc
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)

	@test get_gradient( w, x0, 1 ) == ∇F1(x0)
	@test ∇F1c.counter == 1
	@test get_gradient( w, x0, 2 ) == ∇F2(x0)
	@test ∇F2c.counter == 1
	@test get_jacobian( w, x0 ) == J(x0)
	@test ∇F1c.counter == ∇F2c.counter == 1
	@test Jc.counter == 1
end

@testset "CallbackNoGradsButJacobian" begin

	w = CallBackWrapper(;
		objf = F,
		jacobian = Jc
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)

	@test get_gradient( w, x0, 1 ) == ∇F1(x0)
	@test ∇F1c.counter == 0
	@test Jc.counter == 1
	@test get_gradient( w, x0, 2 ) == ∇F2(x0)
	@test ∇F2c.counter == 0
	@test Jc.counter == 2
	@test get_jacobian( w, x0 ) == J(x0)
	@test ∇F1c.counter == ∇F2c.counter == 0 
	@test Jc.counter == 3
end

@test_throws AssertionError CallBackWrapper()
#%%
ATOL = 1e-4

for WType in [AutoDiffWrapper, FiniteDiffWrapper]
@testset "$(WType)NoGradsButJacobian" begin

	w = WType(;
		objf = Fc,
		jacobian = Jc
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)

	@test get_gradient( w, x0, 1 ) ≈ ∇F1(x0) atol = ATOL
	@test ∇F1c.counter == 0
	@test Fc.counter == 0
	@test Jc.counter == 1
	@test get_gradient( w, x0, 2 ) ≈ ∇F2(x0) atol = ATOL
	@test ∇F2c.counter == 0
	@test Fc.counter == 0
	@test Jc.counter == 2
	@test get_jacobian( w, x0 ) ≈ J(x0) atol = ATOL
	@test ∇F1c.counter == ∇F2c.counter == Fc.counter == 0 
	@test Jc.counter == 3
end

@testset "$(WType)GradsNoJacobian" begin

	w = WType(;
		objf = Fc,
		gradients = [∇F1c, ∇F2c]
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)

	@test get_gradient( w, x0, 1 ) ≈ ∇F1(x0) atol = ATOL
	@test ∇F1c.counter == 1
	@test Fc.counter == 0
	@test Jc.counter == 0
	@test get_gradient( w, x0, 2 ) ≈ ∇F2(x0) atol = ATOL
	@test ∇F2c.counter == 1
	@test Fc.counter == 0
	@test Jc.counter == 0
	@test get_jacobian( w, x0 ) ≈ J(x0) atol = ATOL
	@test ∇F1c.counter == ∇F2c.counter == 2
	@test Fc.counter == 0
	@test Jc.counter == 0
end

@testset "$(WType)NoGradsNoJacobian" begin

	w = WType(;
		objf = Fc,
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)

	@test get_gradient( w, x0, 1 ) ≈ ∇F1(x0) atol = ATOL
	@test ∇F1c.counter == 0
	@test Fc.counter > 0
	@test Jc.counter == 0
	_reset_counter!(Fc)
	@test get_gradient( w, x0, 2 ) ≈ ∇F2(x0) atol = ATOL
	@test ∇F2c.counter == 0
	@test Fc.counter > 0
	@test Jc.counter == 0
	_reset_counter!(Fc)
	@test get_jacobian( w, x0 ) ≈ J(x0) atol = ATOL
	@test ∇F1c.counter == ∇F2c.counter == 0
	@test Fc.counter >0
	@test Jc.counter == 0
end

@testset "$(WType)Hessians" begin
	w = WType(;
		objf = Fc,
		hessians = [H1c, H2c]
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)
	@test get_hessian( w, x0, 1 ) ≈ H1(x0) atol = ATOL
	@test H1c.counter == 1
	@test get_hessian( w, x0, 2 ) ≈ H2(x0) atol = ATOL
	@test H2c.counter == 1
	@test Fc.counter == 0

	w = WType(;
		objf = Fc,
	) 
	_reset_counter!(Fc, ∇F1c, ∇F2c, Jc, H1c, H2c)
	@test get_hessian( w, x0, 1 ) ≈ H1(x0) atol = ATOL
	@test H1c.counter == 0
	@test get_hessian( w, x0, 2 ) ≈ H2(x0) atol = ATOL
	@test H2c.counter == 0
	@test Fc.counter > 0
end
end#for