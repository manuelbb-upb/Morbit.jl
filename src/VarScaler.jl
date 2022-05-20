Base.broadcastable( scal :: AbstractAffineScaler ) = Ref(scal)
# AbstractAffineScaler represents affine linear transformations of the form 
# x̂ = Sx + b 	# S =̂ scaling matrix, b =̂ scaling_constants
# x = S⁻¹(x̂ - b)

# mandatory 
_variables( scal :: AbstractAffineScaler ) = VarInd[]

__init( Type{<:AbstractAffineScaler}; kwargs...) = nothing 

# either one of 
__get_lower_bound_internal( scal :: AbstractAffineScaler, vi :: VarInd ) = nothing
__full_lower_bounds_internal( scal :: AbstractAffineScaler ) = nothing

# one of 
__get_upper_bound_internal( scal :: AbstractAffineScaler, vi :: VarInd ) = nothing
__full_upper_bounds_internal( scal :: AbstractAffineScaler ) = nothing

# one of 
"To obtain the scaled value of variable `vi`, return a `Dictionary` of coefficients to apply to the unscaled variables."
__scaling_coefficients( scal :: AbstractAffineScaler, vi :: VarInd) :: Union{AbstractDictionary,Nothing} = nothing
__scaling_matrix( scal :: AbstractAffineScaler) = nothing

__scaling_constant( scal :: AbstractAffineScaler, vi :: VarInd) = nothing
__scaling_constants_vector( scal :: AbstractAffineScaler ) = nothing

# derived (callable) methods 
_var_pos( scal :: AbstractAffineScaler, vi :: VarInd ) = findfirst(vi, _variables(scal))

macro not_defined_fallback( hopefully_defined_return_expr, alternative_expr )
	return quote
		hopefully_defined_return = $(esc(hopefully_defined_return_expr));
		if isnothing( hopefully_defined_return )
			return $(esc(alternative_expr))
		else
			return hopefully_defined_return
		end
	end
end

function _scaled_lower_bounds_vector( scal :: AbstractAffineScaler )
	return @not_defined_fallback(
		__full_lower_bounds_internal( scal ),
		[ __get_lower_bound_internal( scal, vi ) for vi = _variables(scal) ]
	)
end

function _scaled_upper_bounds_vector( scal :: AbstractAffineScaler )
	return @not_defined_fallback(
		__full_upper_bounds_internal( scal ),
		[ __get_upper_bound_internal( scal, vi ) for vi = _variables(scal) ]
	)
end

_bound_vectors( scal :: AbstractAffineScaler ) = (_scaled_upper_bounds_vector(scal), _scaled_upper_bounds_vector(scal))

function _scaled_lower_bound( scal :: AbstractAffineScaler, vi :: VarInd)
	return @not_defined_fallback(
		__get_lower_bound_internal( scal, vi ),
		_scaled_lower_bounds_vector(scal)[ _var_pos( scal, vi) ]
	)
end

function _scaled_upper_bound( scal :: AbstractAffineScaler, vi :: VarInd)
	return @not_defined_fallback(
		__get_upper_bound_internal( scal, vi ),
		_scaled_upper_bounds_vector(scal)[ _var_pos( scal, vi) ]
	)
end

__unscaling_matrix( scal :: AbstractAffineScaler ) = LinearAlgebra.inv( _scaling_matrix(scal) )
__unscaling_coefficients( scal :: AbstractAffineScaler, vi :: VarInd ) = nothing

# (helpers)
function _coeff_dict_of_dicts(scal)
	return dictionary( vi => __scaling_coefficients( scal, vi ) for vi = _variables(scal) )
end

function _matrix_from_coeff_dict( vars , coeff_dict_of_dicts )
	return hcat(
		( collect(get_indices( coeff_dict_of_dicts[vi], vars ) for vi = vars ))...
	)
end

function _scaling_matrix( scal :: AbstractAffineScaler )
	return @not_defined_fallback(
		__scaling_matrix( scal ),
		_matrix_from_coeff_dict( _variables(scal), _coeff_dict_of_dicts(scal) )
	)
end

function _matrix_to_coeff_dict( mat, vi_pos )
	coeff_row = mat[:, vi_pos];
	dictionary( vj => coeff_row[ _var_pos(scal, vj) ] for vj = _variables(scal) ) 
end

function _scaling_coefficients( scal :: AbstractAffineScaler, vi :: VarInd )
	return @not_defined_fallback(
		__scaling_coefficients(scal, vi),
		_matrix_to_coeff_dict( __scaling_matrix( scal ), _var_pos(scal, vi) )
	)
end

function _scaling_constants_vector( scal :: AbstractAffineScaler )
    return @not_defined_fallback(
		__scaling_constants_vector( scal ),
		__scaling_constant.(scal, _variables(scal))
	)
end

function _scaling_constant( scal ::AbstractAffineScaler, vi :: VarInd )
	return @not_defined_fallback(
		__scaling_constant(scal, vi),
		__scaling_constants_vector( scal )[ _var_pos(scal,vi) ]
	)
end

function _unscaling_coefficients( scal :: AbstractAffineScaler, vi :: VarInd )
	return @not_defined_fallback(
		__unscaling_coefficients( scal, vi ),
		_matrix_to_coeff_dict( __unscaling_matrix(scal), _var_pos(scal, vi) )
	)
end

function _inv_coeff_dict_of_dicts(scal)
	return dictionary( vi => __unscaling_coefficients( scal, vi ) for vi = _variables(scal) )
end

function _unscaling_matrix( scal :: AbstractAffineScaler )
	return @not_defined_fallback(
		__unscaling_matrix(scal),
		_matrix_from_coeff_dict( _variables(scal), _inv_coeff_dict_of_dicts(scal) )
	)
end

function transform( x :: AbstractVector, scal :: AbstractAffineScaler )
	return _scaling_matrix(scal) * x .+ _scaling_constants_vector(scal)
end

function untransform( x_scaled :: AbstractVector, scal :: AbstractAffineScaler )
	return _unscaling_matrix(scal) * ( x_scaled .- _scaling_constants_vector(scal) )
end

function dot_prod( x :: AbstractDictionary, y :: AbstractDictionary )
	return sum( x .* getindices(y, x) )
end

function transform( x :: AbstractDictionary, scal :: AbstractAffineScaler )
	return map(
		vi -> dot_prod(x, _scaling_coefficients( scal, vi) ) + _scaling_constant(scal, vi),
		keys(x)
	)
end
 
function untransform( x :: AbstractDictionary, scal :: AbstractAffineScaler )
	_x = map( ((vi,xi),) -> xi - _scaling_constant(scal, vi), pairs(x) )
	return map(
		vi -> dot_prod( _x, _unscaling_coefficients(scal, vi) ),
		keys(x)
	)
end

function jacobian_of_unscaling( scal :: AbstractAffineScaler )
	return _unscaling_matrix( scal )
end

function jacobian_of_unscaling_inv( scal :: AbstractAffineScaler)
	return LinearAlgebra.inv( jacobian_of_unscaling(scal) )
end

macro set_if_needed(
	kwarg_name_expr,
	kwarg_symbols_expr,
	alternative_params_expr
)
	return quote
		local provided_params = $( esc( kwarg_name_expr ) )
		local var_val = if isnothing( provided_params ) && 
			$(Meta.quot(kwarg_name_expr)) in $(kwarg_symbols_expr)
			$(esc(alternative_params_expr))
		else
			provided_params
		end
		$(esc(Symbol("🐢", kwarg_name_expr))) = var_val
	end
end

function _init( scal_type :: Type{<:AbstractAffineScaler};
	variables :: Union{Nothing, AbstractVector{<:VarInd}} = nothing,
	scaled_lower_bounds_dict :: Union{Nothing, AbstractDictionary{VarInd, <:Real}} = nothing,
	scaled_upper_bounds_dict :: Union{Nothing, AbstractDictionary{VarInd, <:Real}} = nothing,
	scaled_lower_bounds_vector :: Union{Nothing, AbstractVector{<:Real}} = nothing,
	scaled_upper_bounds_vector :: Union{Nothing, AbstractVector{<:Real}} = nothing,
	scaling_coefficients_dict_of_dicts :: Union{Nothing, AbstractDictionary{VarInd, <:AbstractDictionary}} = nothing,
	scaling_matrix :: AbstractMatrix{<:Real} = nothing, 
	scaling_constants_dict :: Union{Nothing, AbstractDictionary{VarInd, <:Real}} = nothing,
	scaling_constants_vector :: AbstractVector{<:Real} = nothing,
	unscaling_coefficients_dict_of_dicts :: Union{Nothing, AbstractDictionary{VarInd, <:AbstractDictionary}} = nothing,
	unscaling_matrix :: AbstractMatrix = nothing
)
	@assert !isnothing(variables) && !isempty(variables)
	@assert !(isnothing(scaled_lower_bounds_dict) && isnothing(scaled_lower_bounds_vector))
	@assert !(isnothing(scaled_upper_bounds_dict) && isnothing(scaled_upper_bounds_vector))
	@assert !(isnothing(scaling_constants_dict) && isnothing(scaling_constants_vector))
	@assert !(isnothing(scaling_coefficients_dict_of_dicts) && isnothing(scaling_matrix))
	@assert !(isnothing(unscaling_coefficients_dict_of_dicts) && isnothing(unscaling_matrix))

	kw_arg_symbols = Base.kwarg_decl( @which( __init(T) ) )

	@set_if_needed(	scaled_lower_bounds_dict, kw_arg_symbols,	
		dictionary( v => b for (v,p) = zip(variables, scaled_lower_bounds_vector) ) 
	)
	@set_if_needed(	scaled_upper_bounds_dict, kw_arg_symbols,
		dictionary( v => b for (v,p) = zip(variables, scaled_upper_bounds_vector) )
	)
	
	@set_if_needed( scaled_lower_bounds_vector, kw_arg_symbols,
		collect( getindices( scaled_lower_bounds_dict, variables ) )
	)
	@set_if_needed( scaled_upper_bounds_vector, kw_arg_symbols,
		collect( getindices( scaled_upper_bounds_dict, variables ) )
	)

	@set_if_needed( scaling_coefficients_dict_of_dicts, kw_arg_symbols,
		dictionary( vi => _matrix_to_coeff_dict( scaling_matrix, vi_pos ) for (vi_pos,vi) = enumerate(variables) )
	)
	@set_if_needed( unscaling_coefficients_dict_of_dicts, kw_arg_symbols,
		dictionary( vi => _matrix_to_coeff_dict( unscaling_matrix, vi_pos ) for (vi_pos,vi) = enumerate(variables) )
	)

	@set_if_needed( scaling_matrix, kw_arg_symbols,
		_matrix_from_coeff_dict( variables, scaling_coefficients_dict_of_dicts )
	)	
	@set_if_needed( unscaling_matrix, kw_arg_symbols,
		_matrix_from_coeff_dict( variables, unscaling_coefficients_dict_of_dicts )
	)	

	@set_if_needed( scaling_constants_dict, kw_arg_symbols,
		dictionary( vi => b for (vi,b) = enumerate( scaling_constants_vector ) )
	)

	@set_if_needed( scaling_constants_vector, kw_arg_symbols,
		collect( getindices( scaling_constants_dict, variables) )
	)

	return __init( scal_type;
		variables, 
		scaled_lower_bounds_dict = 🐢scaled_lower_bounds_dict,
		scaled_upper_bounds_dict = 🐢scaled_upper_bounds_dict,
		scaled_lower_bounds_vector = 🐢scaled_lower_bounds_vector,
		scaled_upper_bounds_vector = 🐢scaled_upper_bounds_vector,
		scaling_coefficients_dict_of_dicts = 🐢scaling_coefficients_dict_of_dicts,
		scaling_matrix = 🐢scaling_matrix, 
		scaling_constants_dict = 🐢scaling_constants_dict,
		scaling_constants_vector = 🐢scaling_constants_vector, 
		unscaling_coefficients_dict_of_dicts = 🐢unscaling_coefficients_dict_of_dicts, 
		unscaling_matrix = 🐢unscaling_matrix
	)
end


# TODO: partial jacobian, gradients ?

# ## Implementations
struct SimpleScaler{
		V, LBType, UBType, DType, BType, DInvType,
	} <: AbstractAffineScaler

	variables :: V

	lb_scaled :: LBType
	ub_scaled :: UBType
	
	D :: DType
	b :: BType

	Dinv :: DInvType
end

function __init( :: Type{<:SimpleScaler};
	variables,
	scaled_lower_bounds_vector,
	scaled_upper_bounds_vector,
	scaling_matrix,
	scaling_constants_vector,
	unscaling_matrix,
	kwargs...
)
	return SimpleScaler(
		variables,
		scaled_lower_bounds_vector,
		scaled_upper_bounds_vector,
		scaling_matrix,
		scaling_constants_vector,
		unscaling_matrix,
	)
end

_variables( scal :: SimpleScaler ) = scal.variables

__scaling_matrix( scal :: SimpleScaler ) = scal.D
__unscaling_matrix( scal :: SimpleScaler ) = scal.Dinv
__scaling_constants_vector( scal :: SimpleScaler ) = scal.b

__full_lower_bounds_internal( scal :: SimpleScaler ) = scal.lb_scaled
__full_upper_bounds_internal( scal :: SimpleScaler ) = scal.ub_scaled

# TODO: FullScaler that implements all `__` methods

#####

struct NoVarScaling{ V, LBType, UBType } <: AbstractAffineScaler 
	variables :: V
	lb :: LBType
	ub :: UBType

	n_vars :: Int 
	function NoVarScaling( vars :: V, lb :: LBType, ub :: UBType ) where{V, LBType, UBType}
		n_vars = length(lb)
		@assert n_vars == length(ub) && n_vars == length(vars)
		return new{V,LBType,UBType}(variables,lb,ub,n_vars)
	end
end

__init( :: Type{<:NoVarScaling};
	variables,
	scaled_lower_bounds_vector,
	scaled_upper_bounds_vector,
	kwargs...
) = NoVarScaling(variables, scaled_lower_bounds_vector, scaled_upper_bounds_vector)

__scaled_lower_bounds_vector( scal :: NoVarScaling ) = scal.lb 
__scaled_upper_bounds_vector( scal :: NoVarScaling ) = scal.ub

__scaling_matrix(scal :: NoVarScaling) = LinearAlgebra.I( scal.n_vars )
__unscaling_matrix(scal :: NoVarScaling) = _scaling_matrix(scal)
__scaling_constants_vector(scal :: NoVarScaling) = zeros(Bool, scal.n_vars)	# TODO `Bool` sensible here?

# overwrite defaults
transform( x, :: NoVarScaling ) = copy(x) 
untransform( _x, :: NoVarScaling ) = copy(_x)
 
jacobian_of_unscaling_inv(scal :: NoVarScaling) = jacobian_of_unscaling( scal )

# derived functions and helpers, used by the algorithm:

# from two scalers 
# s(x) = Sx + a
# t(x) = Tx + b
# Return the linear scaler t ∘ s⁻¹, i.e., the 
# scaler that untransforms via s and then applies t
function combined_untransform_transform_scaler(
	scal1 :: AbstractAffineScaler, scal2 :: AbstractAffineScaler, 
	target_type = SimpleScaler 
)

	variables = _variables( scal1 )
	vars2 = _variables( scal2 )
	@assert variables == vars2 || isempty(setdiff( variables, vars2 ))

	# indices such that variables == vars2[ind2]
	ind2 = if variables != vars2
		[ findfirst(v, variables) for v = vars2 ]
	else
		eachindex( vars2 )
	end
	
	scaling_matrix = _scaling_matrix(scal2)[ind2, ind2] * _unscaling_matrix(scal1)
	scaling_constants_vector = _scaling_constants_vector(scal2)[ind2] - _unscaling_matrix(scal1) * _scaling_constants_vector(scal1)
	unscaling_matrix = _scaling_matrix(scal1) * _unscaling_matrix(scal2)[ind2, ind2]

	lb_old, ub_old = full_bounds_internal( scal1 )
	scaled_lower_bounds_vector = scaling_matrix * lb_old + scaling_constants_vector
	scaled_upper_bounds_vector = scaling_matrix * ub_old + scaling_constants_vector
	
	return _init( target_type;
		variables,
		scaled_lower_bounds_vector,
		scaled_upper_bounds_vector,
		scaling_matrix,
		scaling_constants_vector,
		unscaling_matrix,
		kwargs...
	)
end


function combined_untransform_transform_scaler( scal1 :: NoVarScaling, :: NoVarScaling )
	return scal1 
end

function initialize_var_scaler(x0, mop :: AbstractMOP, ac :: AbstractConfig)
	lb, ub = full_vector_bounds( mop )
	n_vars = length(lb)

	user_scaler = var_scaler( ac )
	variables = _variables(mop)
	
	if user_scaler isa AbstractAffineScaler 
		vars_scaler = _variables(user_scaler)
		if isempty(setdiff(vars_mop, vars_scaler))
			if variables == vars_scaler
				return user_scaler
			else
				@warn "Cannot use the provided variable scaler because the variable indices differ."
			end
		else
			@warn "Cannot use the provided variable scaler because the variable indices differ."
		end
	end

	if !(any( isinf.([lb;ub])) )
		if user_scaler == :default || user_scaler == :auto
			# the problem vars are finitely box constraint
			# we scale every variable to the unit hypercube [0,1]^n 
			w = ub .- lb
			w_inv = 1 ./ w
			scaling_constants_vector = - lb .* w_inv

			scaling_matrix = LinearAlgebra.Diagonal( w_inv )
			unscaling_matrix = LinearAlgebra.Diagonal( w )
			scaled_lower_bounds_vector = scaling_matrix * lb .+ scaling_constants_vector
			scaled_upper_bounds_vector = scaling_matrix * ub .+ scaling_constants_vector
			return _init( SimpleScaler;
				variables,
				scaled_lower_bounds_vector,
				scaled_upper_bounds_vector,
				scaling_matrix,
				scaling_constants_vector,
				unscaling_matrix,
			)
		end	
	end 

	return _init( NoVarScaling; 
		variables,
		scaled_lower_bounds_vector = lb,
		scaled_upper_bounds_vector = ub,
	)
end