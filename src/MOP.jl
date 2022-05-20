# this is NOT a MOI.ModelLike !!
# but I tried to design it with MOI in mind 
# to have an easy time writing a wrapper eventually

# I am now using Dictionaries.jl which should also preserve the order :)

# depends on `VecFun.jl` (implementing `VecFun`, `CompositeVecFun` and `RefVecFun`)
@with_kw struct MOP <: AbstractMOP{true}
	variables :: Vector{VarInd} = []

	lower_bounds :: Dict{VarInd, Real} = Dict()
	upper_bounds :: Dict{VarInd, Real} = Dict()
	
	functions :: Dictionary{AnyIndex, AbstractVecFun} = Dictionary()

	objective_functions :: Dictionary{ObjectiveIndex, AbstractVecFun} = Dictionary()
	nl_eq_constraints :: Dictionary{NLConstraintIndexEq, AbstractVecFun} = Dictionary()
	nl_ineq_constraints :: Dictionary{NLConstraintIndexIneq, AbstractVecFun} = Dictionary()

	eq_constraints :: Dictionary{ConstraintIndexEq, MOI.VectorAffineFunction} = Dictionary()
	ineq_constraints :: Dictionary{ConstraintIndexIneq, MOI.VectorAffineFunction} = Dictionary()

	optimized_evaluation :: Bool = true
end

# custom hash and == function for memoization
# else, empty matrices could be constructed for MOPTyped
# if the original mop is changed in-between iterations
_hash(x, h, ::Val{0}) = h

function _hash(x, h, ::Val{i}) where i
    _hash(x, hash(getfield(x, i), h), Val(i-1))
end

function Base.hash(mop :: MOP, h::UInt)
	_hash(mop, hash(MOP, h), Val(fieldcount(MOP)))
end

function Base.:(==)( a :: MOP, b :: MOP )
	r_val = true
	for fn in fieldnames(MOP)
		r_val *= getfield(a, fn) == getfield(b, fn)
		r_val == false && break
	end
	return r_val
end

struct MOPTyped{
		VarType <: Tuple{Vararg{<:VarInd}},
		LbType <: Union{AbstractDict, AbstractDictionary}, 
		UbType <: Union{AbstractDict, AbstractDictionary},
		FunType <: Union{AbstractDict, AbstractDictionary},
		ObjfType <: Union{AbstractDict, AbstractDictionary},
		NlEqType <: Union{AbstractDict, AbstractDictionary}, 
		NlIneqType <: Union{AbstractDict, AbstractDictionary},
		EqMatType, IneqMatType, EqVecType, IneqVecType,
		EqType <: Union{AbstractDict, AbstractDictionary},
		IneqType <: Union{AbstractDict, AbstractDictionary},
	} <: AbstractMOP{false}

	variables :: VarType
	lower_bounds :: LbType
	upper_bounds :: UbType 

	functions :: FunType

	objective_functions :: ObjfType
	nl_eq_constraints :: NlEqType
	nl_ineq_constraints :: NlIneqType

	eq_mat :: EqMatType
	ineq_mat :: IneqMatType
	eq_vec :: EqVecType
	ineq_vec :: IneqVecType

	# TODO 11.05.2022 use full constraint expressions (Dictionary instead of vectors)
	eq_constraints :: EqType 
	ineq_constraints :: IneqType

	optimized_evaluation :: Bool
end


function MOPTyped( mop :: AbstractMOP )
	variables = Tuple( var_indices(mop) )
	lower_bounds = Base.ImmutableDict( ( vi => ensure_precision( get_lower_bound( mop, vi ) ) for vi in variables )... )
	upper_bounds = Base.ImmutableDict( ( vi => ensure_precision( get_upper_bound( mop, vi ) ) for vi in variables )... )

	functions = _create_dict(mop, get_inner_function_indices(mop), InnerIndex )

	objective_functions = _create_dict(mop, 
		get_objective_indices(mop), ObjectiveIndex ) 
	nl_eq_constraints = _create_dict(mop, 
		get_nl_eq_constraint_indices(mop), NLConstraintIndexEq )
	nl_ineq_constraints = _create_dict(mop, 
		get_nl_ineq_constraint_indices(mop), NLConstraintIndexIneq ) 

	eq_constraints = _create_dict( mop, 
		get_eq_constraint_indices(mop), ConstraintIndexEq
	)
	ineq_constraints = _create_dict( mop, 
		get_ineq_constraint_indices(mop), ConstraintIndexIneq
	)

	eq_mat, eq_vec = get_eq_matrix_and_vector( mop )
	ineq_mat, ineq_vec = get_ineq_matrix_and_vector( mop )
	
	return MOPTyped( 
		variables,
		lower_bounds,
		upper_bounds,
		functions,
		objective_functions,
		nl_eq_constraints,
		nl_ineq_constraints,
		eq_mat, ineq_mat, eq_vec, ineq_vec,
		eq_constraints,
		ineq_constraints,
		mop.optimized_evaluation
	)
end

function _create_dict(mop, indices, index_type = FunctionIndex )
	if isempty( indices )
		return Base.ImmutableDict{index_type, Nothing}()
	else
		num_indices = length(indices)
		dict_vals = [ _get(mop, ind) for ind = indices ]
		return ArrayDictionary{ index_type, eltype(dict_vals) }( SVector{num_indices}(collect(indices)), dict_vals )
	end
end

const BothMOP = Union{MOP, MOPTyped}

# alternative constructors for convenience 
MOP( n_vars :: Int ) = MOP(; variables = [ VarInd(i) for i = 1 : n_vars] )
function MOP( lb :: Vector{<:Real}, ub :: Vector{<:Real} )
	n_vars = length(lb)
	@assert length(ub) == length(lb) "Variable bounds vectors must have same length."
	var_inds = [ VarInd(i) for i = 1 : n_vars ]
	return MOP(; 
		variables = var_inds,
		lower_bounds = Dict( var_ind => lb[i] for (i,var_ind) = enumerate(var_inds) ),
		upper_bounds = Dict( var_ind => ub[i] for (i,var_ind) = enumerate(var_inds) ),
	)
end

var_indices(mop :: BothMOP) = mop.variables 

get_lower_bound( mop :: BothMOP, vi :: VarInd) = get( mop.lower_bounds, vi, -MIN_PRECISION( Inf ))
get_upper_bound( mop :: BothMOP, vi :: VarInd) = get( mop.upper_bounds, vi, MIN_PRECISION( Inf ) )

num_vars( mop :: BothMOP ) = length( mop.variables )

get_objective_indices( mop :: BothMOP ) = keys( mop.objective_functions )
get_inner_function_indices(mop :: BothMOP) = keys(mop.functions)
get_nl_eq_constraint_indices( mop :: BothMOP ) = keys( mop.nl_eq_constraints )
get_eq_constraint_indices( mop :: BothMOP ) = keys( mop.eq_constraints )
get_nl_ineq_constraint_indices( mop :: BothMOP ) = keys( mop.nl_ineq_constraints )
get_ineq_constraint_indices( mop :: BothMOP ) = keys( mop.ineq_constraints )

get_eq_matrix_and_vector( mop :: MOPTyped ) = (mop.eq_mat, mop.eq_vec)
get_ineq_matrix_and_vector( mop :: MOPTyped ) = (mop.ineq_mat, mop.ineq_vec)

_get( mop :: BothMOP, ind :: InnerIndex ) = mop.functions[ind]

_get( mop :: BothMOP, ind::ObjectiveIndex ) = mop.objective_functions[ind]

_get( mop :: BothMOP, ind :: NLConstraintIndexEq ) = mop.nl_eq_constraints[ind]
_get( mop :: BothMOP, ind :: NLConstraintIndexIneq ) = mop.nl_ineq_constraints[ind]
_get( mop :: BothMOP, ind :: ConstraintIndexEq ) = mop.eq_constraints[ind]
_get( mop :: BothMOP, ind :: ConstraintIndexIneq ) = mop.eq_constraints[ind]

_next_val( indices ) = isempty(indices) ? 1 : maximum( ind.value for ind in indices ) + 1 
_next_val( dict :: Union{AbstractDict, AbstractDictionary} ) = _next_val( collect(keys(dict) ))

function add_lower_bound!(mop :: MOP, vi :: VarInd, val :: Real) 
	mop.lower_bounds[vi] = val
	return nothing 
end

function del_lower_bound!(mop :: MOP, vi :: VarInd)
	delete!(mop.lower_bounds, vi )
	return nothing 
end

function del_upper_bound!(mop :: MOP, vi :: VarInd)
	delete!(mop.upper_bounds, vi )
	return nothing 
end

function add_upper_bound!( mop :: MOP, vi :: VarInd, val :: Real) 
	mop.upper_bounds[vi] = val
	return nothing 
end

function add_variable!(mop :: MOP)
	var_ind = VarInd( _next_val(mop.variables) ) 
	push!(mop.variables, var_ind)
	return var_ind
end

function _add_function!(mop, fun :: AbstractVecFun )
	if !(fun isa VecFun)
		# verbose error and explanation
		error("Adding something else than `VecFun`s as inner functions is not supported yet.")
	end
	ind = InnerIndex( 
		_next_val( mop.functions ),
		num_outputs( fun ) 
	)
	insert!(mop.functions, ind, fun)
	return ind
end

# helper to get the right type of "outer" function 
# from `outer` and `n_out`
function _get_composite_vec_fun(mop, nl_ind, outer, n_vars, n_out)
	if outer isa AbstractString 
		if isempty(outer)
			# if there is no expr, simply use a `RefVecFun` that stores 
			# a reference to `fun` and forwards all relevant methods
			return RefVecFun( _get(mop,nl_ind), nl_ind )
		else
			_outer = make_outer_fun( outer; n_vars, n_out )
		end
	elseif outer isa AbstractVecFun
		if !(outer isa VecFun) || needs_gradients(model_cfg(outer)) == false
			# verbose error
			error("The outer function must be of type `VecFun` and support 
			first order derivatives as set up with an `ExactConfig()` or `TaylorCallbackConfig()`.")
		end
		_outer = outer
	end

	return CompositeVecFun( _outer, _get(mop,nl_ind), nl_ind )
end

function _add_objective!(
	mop :: MOP, nl_ind :: InnerIndex, 
	outer :: Union{String, AbstractVecFun} = ""; n_out = 0, n_vars = 0
)
	_fun = _get_composite_vec_fun(mop, nl_ind, outer, n_vars, n_out)
	# obtain next `ObjectiveIndex`
	ind = ObjectiveIndex( 
		_next_val( mop.objective_functions ),
		num_outputs( _fun ) 
	)
	# put `_fun` into `objective_functions` dict
	insert!(mop.objective_functions, ind, _fun)
	return ind
end

# this definiton allows for the required call `_add_objective!(mop, fun)`
function _add_objective!(mop :: MOP, fun :: AbstractVecFun, 
	outer :: Union{String, AbstractVecFun} = ""; n_vars = 0, n_out = 0 )
	# add the function to the `functions` dict in `MOP`
	nl_ind = _add_function!(mop, fun)
	return _add_objective!(mop, nl_ind, outer; n_vars, n_out)
end

# Similar for equality constraints …
function _add_nl_eq_constraint!(mop :: MOP, nl_ind :: InnerIndex,
	outer :: Union{String, AbstractVecFun} = ""; n_out = 0, n_vars = 0 )
	_fun = _get_composite_vec_fun(mop, nl_ind, outer, n_vars,n_out)
	ind = NLConstraintIndexEq( 
		_next_val( mop.nl_eq_constraints ),
		num_outputs( _fun ),
	)
	insert!(mop.nl_eq_constraints, ind, _fun)
	return ind
end

function _add_nl_eq_constraint!(mop :: MOP, fun :: AbstractVecFun, 
	outer :: Union{String, AbstractVecFun} = ""; n_vars = 0, n_out = 0)
	# add the function to the `functions` dict in `MOP`
	nl_ind = _add_function!(mop, fun)
	return _add_nl_eq_constraint!(mop, nl_ind, outer; n_vars, n_out)
end

# … and inequality constraints:
function _add_nl_ineq_constraint!(mop :: MOP, nl_ind :: InnerIndex, 
	outer :: Union{String, AbstractVecFun} = ""; n_vars = 0, n_out = 0)
	_fun = _get_composite_vec_fun(mop, nl_ind, outer, n_vars, n_out)
	ind = NLConstraintIndexIneq( 
		_next_val( mop.nl_ineq_constraints ),
		num_outputs( _fun ),
	)
	insert!(mop.nl_ineq_constraints, ind, _fun)
	return ind
end

function _add_nl_ineq_constraint!(mop :: MOP, fun :: AbstractVecFun, 
	outer :: Union{String, AbstractVecFun} = ""; n_vars = 0, n_out = 0)
	# add the function to the `functions` dict in `MOP`
	nl_ind = _add_function!(mop, fun)
	return _add_nl_ineq_constraint!(mop, nl_ind, outer; n_vars, n_out)
end 

# linear constraints:
function _add_eq_constraint!( mop :: MOP, aff_func :: MOI.VectorAffineFunction )
	ind = ConstraintIndexEq(
		_next_val( mop.eq_constraints ),
		MOI.output_dimension( aff_func ),
	)
	insert!( mop.eq_constraints, ind, aff_func )
	return ind 
end

function _add_ineq_constraint!( mop :: MOP, aff_func :: MOI.VectorAffineFunction )
	ind = ConstraintIndexIneq(
		_next_val( mop.ineq_constraints ),
		MOI.output_dimension( aff_func ),
	)
	insert!(mop.ineq_constraints,ind, aff_func )
	return ind 
end

function lazy_get!(dict, index, func, args...)
	if haskey(dict, index)
		return dict[index]
	else
		y = func(args...)
		insert!(dict, index, y)
		return y
	end
end

# improve evaluation by reusing "inner" evaluations
function _optimized_eval_at_unscaled_site(mop, fun, tmp_res, x, ind = nothing)

	if fun isa VecFun
		return lazy_get!( tmp_res, ind, eval_vfun, fun, x )
	else
		inner_fun = fun.inner_ref[]
		gx = _optimized_eval_at_unscaled_site(
			mop, inner_fun, tmp_res, x, fun.inner_index
		)
		if fun isa RefVecFun
			return gx
		else			
			return eval_vfun( fun.outer_ref[], [x; gx])
		end
	end
	#=
	if fun isa RefVecFun
		#return get!( tmp_res, fun.inner_index, eval_vfun(fun, x) )
		return lazy_get!( tmp_res, fun.inner_index, eval_vfun, fun, x )
	end

	if fun isa CompositeVecFun
		#gx = get!( tmp_res, fun.inner_index, eval_vfun(fun.inner_ref[], x))
		gx = lazy_get!( tmp_res, fun.inner_index, eval_vfun, fun.inner_ref[], x)
		return eval_vfun(fun.outer_ref[], [x; gx])
	end
	
	return eval_vfun( fun, x )
	=#
end

function _eval_at_indices_at_unscaled_site( mop :: BothMOP, indices, tmp_res, x )
    return Dictionary(
        indices,
        _optimized_eval_at_unscaled_site(mop, _get(mop, ind), tmp_res, x) 
			for ind = indices 
    )
end

function _eval_nl_functions_at_unscaled_site( mop, tmp_res, x )
    return _eval_at_indices_at_unscaled_site( mop, get_inner_function_indices(mop), tmp_res, x )
end

function _eval_objectives_at_unscaled_site( mop, tmp_res, x )
    return _eval_at_indices_at_unscaled_site( mop, get_objective_indices(mop), tmp_res, x )
end

function _eval_nl_eq_constraints_at_unscaled_site( mop, tmp_res, x )
    return _eval_at_indices_at_unscaled_site( mop, get_nl_eq_constraint_indices(mop), tmp_res, x )
end

function _eval_nl_ineq_constraints_at_unscaled_site( mop, tmp_res, x )
    return _eval_at_indices_at_unscaled_site( mop, get_nl_ineq_constraint_indices(mop), tmp_res, x )
end

function _optimized_evaluate_at_unscaled_site( mop, x)
	tmp_res = _eval_nl_functions_at_unscaled_site( mop, x )
	return (
        tmp_res,
        _eval_objectives_at_unscaled_site( mop, tmp_res, x ),
        _eval_nl_eq_constraints_at_unscaled_site( mop, tmp_res, x ),
        _eval_nl_ineq_constraints_at_unscaled_site( mop, tmp_res, x ),
    )
end

function evaluate_at_unscaled_site( mop :: BothMOP, x )
	if mop.optimized_evaluation
		return _optimized_evaluate_at_unscaled_site( mop, x )
	else
		return _evaluate_at_unscaled_site( mop, x )
	end
end