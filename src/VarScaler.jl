struct NoVarScaling <: AbstractVarScaler end

# defaults (no scaling)
transform( x, :: AbstractVarScaler ) = copy(x) 
untransform( _x, :: AbstractVarScaler ) = copy(_x)

struct LinearScaling{DType, BType, DInvType} <: AbstractVarScaler
	D :: DType
	b :: BType

	Dinv :: DInvType

	function LinearScaling( D :: DType, b :: BType ) where{DType, BType}
		n, m = size(D)
		@assert n == m "Scaling Matrix `D` must be square and invertible." 
		@assert m == length(b) "Dimensions of scaling matrix `D` and translation vector `b` do not match."
		Dinv = inv(D)
		return new{DType,BType,typeof(Dinv)}(D,b,Dinv)
	end
end

function LinearScaling( d :: Vector{T}, b = nothing ) where T<:Real
	_b = isnothing(b) ? zeros(T, length(d)) : b
	return LinearScaling( LinearAlgebra.Diagonal(d), _b )
end

transform( x, scal :: LinearScaling) = scal.D * x .+ scal.b
untransform( _x, scal :: LinearScaling ) =  scal.Dinv * ( _x .- scal.b )

# derived functions, used by the algorithm:

# provided the m×n jacobian `J` of `f: ℝⁿ → ℝᵐ`, return a vector `c`
# of `n` scaling vectors so that the absolute values of `J' = J⋅diagm(c)` 
# are as close to one as possible. 
# `J'` is the jacobian of `f∘s` where `s` scales the variables with `c`.
# Loosely inspired by “Scaling nonlinear programs”, Lasdon & Beck
function _scaling_factors( J, RHS = nothing )
	M, num_cols = size(J)
	factors = ones( Base.promote_op( log, eltype(J) ), num_cols )
	
	!isnothing(RHS) && @assert size(J) == (M,num_cols)
	
	for (j, col) = enumerate(eachcol(J))
		NZ = findall( .!(iszero.(col)) )
		num_not_zero = sum( NZ )
		if num_not_zero > 0
			exp_arg = - sum( log.( abs.(col[NZ] ) ) )
			if !isnothing(RHS)
				exp_arg += sum( log.(abs.(RHS[NZ,j])) )
			end
			exp_arg /= num_not_zero
			
			factors[j] = exp( exp_arg )
		end
	end
	return factors
end
import Statistics: mean
function get_var_scaler(x0, mop :: AbstractMOP, ac :: AbstractConfig )
	lb, ub = full_bounds( mop )

	user_scaler = var_scaler( ac )
	if !isnothing( user_scaler )
		return user_scaler
	end

	if !(any( isinf.([lb;ub])) )
		# the problem vars are finitely box constraint
		# we scale every variable to the unit hypercube [0,1]^n 
		w = ub .- lb
		w_inv = 1 ./ w
		t = - lb ./ w
		return LinearScaling( w_inv, t )		
	else
		@warn """
		Doing some finite differencing to estimate variable scaling.
		You can disable this behavior with `var_scaler = NoVarScaling()`.
		"""

		any_inf_ind = .|(isinf.(lb), isinf.(ub))
		bound_ind = .!(any_inf_ind)

		fd_func = function( x )
			return hcat( [eval_vec_mop( mop, x, ind ) for ind = get_function_indices(mop)]... )
		end

		x0_pert = _project_into_box( x0 .+ rand( -.1:1e-4:1, length(x0) ), lb, ub )
		J = FD.finite_difference_jacobian(fd_func, x0_pert)

		if all( any_inf_ind )
			# problem vars are completely unconstrained
			var_factors = _scaling_factors(J)
			return LinearScaling( var_factors )
		else
			# some variables are not constrained
			w = ub .- lb 
			
			J_fin = J[:, bound_ind] ./ w[bound_ind]'	# scale each colum as if we applied unit scaling
			J_inf = J[:, any_inf_ind]
			target_val = mean( abs.(J_fin); dims = 2 )
			RHS = repeat( target_val, 1, sum(any_inf_ind) )

			var_factors_inf = _scaling_factors( J_inf, RHS )

			J_inf .* var_factors_inf'

			var_factors = similar(w)
			var_factors[any_inf_ind] .= var_factors_inf
			var_factors[bound_ind] .= (1 ./ w[bound_ind] )
			
			return LinearScaling( var_factors )
		end
	end
	
end