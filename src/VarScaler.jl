Base.broadcastable( scal :: AbstractVarScaler ) = Ref(scal)
# AbstractVarScaler represents linear transformations of the form 
# x̂ = Sx + b 	# S =̂ scaling matrix, b =̂ scaling_offset
# x = S⁻¹(x̂ - b)

# mandatory 
full_lower_bounds_internal( scal :: AbstractVarScaler ) = nothing
full_upper_bounds_internal( scal :: AbstractVarScaler ) = nothing

scaling_matrix( scal :: AbstractVarScaler ) = nothing 
function scaling_offset( scal :: AbstractVarScaler )
    nothing
end
unscaling_matrix( scal :: AbstractVarScaler ) = LinearAlgebra.inv( scaling_matrix(scal) )


# derived
function transform( x, scal :: AbstractVarScaler )
	return scaling_matrix(scal) * x .+ scaling_offset(scal)
end

function untransform( x_scaled, scal :: AbstractVarScaler )
	return unscaling_matrix(scal) * ( x_scaled .- scaling_offset(scal) )
end

function full_bounds_internal( scal :: AbstractVarScaler )
	return ( 
		full_lower_bounds_internal(scal), 
		full_upper_bounds_internal(scal)
	)
end

function jacobian_of_unscaling( scal :: AbstractVarScaler )
	return unscaling_matrix( scal )
end

function jacobian_of_unscaling_inv( scal :: AbstractVarScaler)
	return LinearAlgebra.inv( jacobian_of_unscaling(scal) )
end


# from two scalers 
# s(x) = Sx + a
# t(x) = Tx + b
# Return the linear scaler t ∘ s⁻¹, i.e., the 
# scaler that untransforms via s and then applies t
function combined_untransform_transform_scaler(
		scal1 :: AbstractVarScaler, scal2 :: AbstractVarScaler )

	scal_mat = scaling_matrix(scal2) * unscaling_matrix(scal1)
	scal_offset = scaling_offset(scal2) - unscaling_matrix(scal1) * scaling_offset(scal1)
	unscal_mat = scaling_matrix(scal1) * unscaling_matrix(scal2)

	lb_old, ub_old = full_bounds_internal( scal1 )
	lb = scal_mat * lb_old + scal_offset
	ub = scal_mat * ub_old + scal_offset
	return LinearScaling(lb, ub, scal_mat, scal_offset, unscal_mat)		
end

#####

struct NoVarScaling{ LBType, UBType } <: AbstractVarScaler 
	lb :: LBType
	ub :: UBType

	n_vars :: Int 
	function NoVarScaling( lb :: LBType, ub :: UBType ) where{LBType, UBType}
		n_vars = length(lb)
		@assert n_vars == length(ub)
		return new{LBType,UBType}(lb,ub,n_vars)
	end
end

function combined_untransform_transform_scaler( scal1 :: NoVarScaling, :: NoVarScaling )
	return scal1 
end


full_lower_bounds_internal( scal :: NoVarScaling ) = scal.lb 
full_upper_bounds_internal( scal :: NoVarScaling ) = scal.ub

scaling_matrix(scal :: NoVarScaling) = LinearAlgebra.I( scal.n_vars )
unscaling_matrix(scal :: NoVarScaling) = scaling_matrix(scal)
scaling_offset(scal :: NoVarScaling) = zeros(Bool, scal.n_vars)	# TODO `Bool` sensible here?

# overwrite defaults
transform( x, :: NoVarScaling ) = copy(x) 
untransform( _x, :: NoVarScaling ) = copy(_x)
 
jacobian_of_unscaling_inv(scal :: NoVarScaling) = jacobian_of_unscaling( scal )

struct LinearScaling{
		LBType, UBType, DType, BType, DInvType,
	} <: AbstractVarScaler
	

	lb_scaled :: LBType
	ub_scaled :: UBType
	
	D :: DType
	b :: BType

	Dinv :: DInvType

end

function LinearScaling( lb :: L, ub :: U, D :: DType, b :: BType ) where{L, U, DType, BType}
	n, m = size(D)
	@assert n == m "Scaling Matrix `D` must be square and invertible." 
	@assert m == length(b) "Dimensions of scaling matrix `D` and translation vector `b` do not match."
	Dinv = inv(D)

	lb_scaled = D * lb .+ b
	ub_scaled = D * ub .+ b

	return LinearScaling(lb_scaled,ub_scaled,D,b,Dinv)
end

function LinearScaling( lb, ub, d :: Vector{T}, b = nothing ) where T<:Real
	_b = isnothing(b) ? zeros(T, length(d)) : b
	return LinearScaling( lb, ub, LinearAlgebra.Diagonal(d), _b )
end

scaling_matrix( scal :: LinearScaling ) = scal.D
unscaling_matrix( scal :: LinearScaling ) = scal.Dinv
scaling_offset( scal :: LinearScaling ) = scal.b

full_lower_bounds_internal( scal :: LinearScaling ) = scal.lb_scaled
full_upper_bounds_internal( scal :: LinearScaling ) = scal.ub_scaled

# derived functions and helpers, used by the algorithm:

# provided the m×n jacobian `J` of `f: ℝⁿ → ℝᵐ`, return a vector `c`
# of `n` scaling vectors so that the absolute values of `J' = J⋅diagm(c)` 
# are as close to one as possible. 
# `J'` is the jacobian of `f∘s` where `s` scales the variables with `c`.
# Loosely inspired by “Scaling nonlinear programs”, Lasdon & Beck
# Works only for constant and linear objectives!!! deactivated for now
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

const MIN_SCALING_FACTOR = 1e-8
const MAX_SCALING_FACTOR = 1e8

function _estimate_linear_scaling( lb, ub, J )
	any_inf_ind = .|(isinf.(lb), isinf.(ub))
	bound_ind = .!(any_inf_ind)

	if all( any_inf_ind )
		# problem vars are completely unconstrained
		var_factors = _scaling_factors(J)
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
	end
	var_factors[:] .= max.( min.( var_factors, MAX_SCALING_FACTOR ), MIN_SCALING_FACTOR )
	
	return LinearScaling( lb, ub, var_factors )
end

function get_var_scaler(x0, mop :: AbstractMOP, ac :: AbstractConfig)
	lb, ub = full_vector_bounds( mop )
	n_vars = length(lb)

	user_scaler = var_scaler( ac )
	
	if user_scaler isa AbstractVarScaler
		return user_scaler
	end

	if !(any( isinf.([lb;ub])) )
		if user_scaler == :default || user_scaler == :auto
			# the problem vars are finitely box constraint
			# we scale every variable to the unit hypercube [0,1]^n 
			w = ub .- lb
			w_inv = 1 ./ w
			t = - lb ./ w
			return LinearScaling( lb, ub, w_inv, t )
		end	
	else
		if user_scaler == :auto
			@warn """
			Doing some finite differencing to estimate variable scaling.
			You can disable this behavior with `var_scaler = NoVarScaling()`.
			"""

			x0_pert = _project_into_box( x0 .+ rand( -.1:1e-4:1, length(x0) ), lb, ub )

			J = vcat( [ let _J = get_objf_jacobian( _get(mop, ind), x0_pert );
				J_ind = if isnothing(_J)
					FD.finite_difference_jacobian( 
					ξ -> eval_vec_mop_at_func_index_at_unscaled_site( mop, ξ, ind ), x0_pert ) 
				else
					_J
				end
				J_ind 
			end for ind = get_function_indices(mop) ]... )
			
			return _estimate_linear_scaling( lb, ub, J )
		end
	end

	return NoVarScaling(lb, ub)
end

function new_var_scaler( x_scaled, old_scal :: AbstractVarScaler, mop :: AbstractMOP, 
	sc :: AbstractSurrogateContainer, ac :: AbstractConfig  )

	# determine bounds for untransformed problem 
	lb, ub = full_vector_bounds(mop)
	if var_scaler_update( ac ) == :model
		@logmsg loglevel2 "Determining new scaling with surrogate derivatives."
		# jacobian of `( m_k ∘ s⁻¹ )( x̂ )` =̂ Jm(x) ⋅ Js⁻¹(x̂)  
		J_composite = vcat( 
			[ eval_container_jacobian_at_func_index_at_scaled_site( sc, old_scal, x_scaled, ind ) 
				for ind = get_function_indices(sc) ]... 
		)
		# approximation of Jf ≈ Jm_k
		J = J_composite * jacobian_of_unscaling_inv( old_scal)
		return _estimate_linear_scaling( lb, ub, J )
	end

	return old_scal
end