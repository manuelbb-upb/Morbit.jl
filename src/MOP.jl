# this is NOT a MOI.ModelLike !!
# but I tried to design it with MOI in mind 
# to have an easy time writing a wrapper eventually
using DataStructures: SortedDict

@with_kw struct MOP <: AbstractMOP{true}
	variables :: Vector{VarInd} = []

	lower_bounds :: Dict{VarInd, Real} = Dict()
	upper_bounds :: Dict{VarInd, Real} = Dict()

	objective_functions :: SortedDict{ObjectiveIndex, AbstractVecFun} = Dict()
	nl_eq_constraints :: SortedDict{ConstraintIndex, AbstractVecFun} = Dict()
	nl_ineq_constraints :: SortedDict{ConstraintIndex, AbstractVecFun} = Dict()

	eq_constraints :: SortedDict{ConstraintIndex, MOI.VectorAffineFunction} = Dict()
	ineq_constraints :: SortedDict{ConstraintIndex, MOI.VectorAffineFunction} = Dict()
end

struct MOPTyped{
		VarType <: Tuple{Vararg{<:VarInd}},
		LbType <: AbstractDict{VarInd,<:AbstractFloat},
		UbType <: AbstractDict{VarInd,<:AbstractFloat},
		ObjfType <: AbstractDict, #{ObjectiveIndex,<:Union{AbstractVecFun,Nothing}},
		NlEqType <: AbstractDict,   #{ConstraintIndex,<:Union{AbstractVecFun,Nothing}},
		NlIneqType <: AbstractDict, #{ConstraintIndex,<:Union{AbstractVecFun,Nothing}},
		EqMatType, IneqMatType, EqVecType, IneqVecType
	} <: AbstractMOP{false}

	variables :: VarType
	lower_bounds :: LbType
	upper_bounds :: UbType 
	objective_functions :: ObjfType
	nl_eq_constraints :: NlEqType
	nl_ineq_constraints :: NlIneqType

	eq_mat :: EqMatType
	ineq_mat :: IneqMatType
	eq_vec :: EqVecType
	ineq_vec :: IneqVecType
end

function MOPTyped( mop :: AbstractMOP )
	variables = Tuple( var_indices(mop) )
	lower_bounds = Base.ImmutableDict( ( vi => ensure_precision( get_lower_bound( mop, vi ) ) for vi in variables )... )
	upper_bounds = Base.ImmutableDict( ( vi => ensure_precision( get_upper_bound( mop, vi ) ) for vi in variables )... )
	objective_functions = let indices = get_objective_indices(mop);
		if isempty(indices)
			Base.ImmutableDict{ObjectiveIndex, Nothing}()
		else
			SortedDict( ind => _get(mop, ind) for ind = indices ) 
		end
	end
	nl_eq_constraints = let indices = get_nl_eq_constraint_indices(mop);
		if isempty(indices)
			Base.ImmutableDict{ConstraintIndex, Nothing}()
		else
			SortedDict( ind => _get(mop, ind) for ind = indices ) 
		end
	end
	nl_ineq_constraints = let indices = get_nl_ineq_constraint_indices(mop);
		if isempty(indices)
			Base.ImmutableDict{ConstraintIndex, Nothing}()
		else
			SortedDict( ind => _get(mop, ind) for ind = indices )
		end
	end

	eq_mat, eq_vec = get_eq_matrix_and_vector( mop )
	ineq_mat, ineq_vec = get_ineq_matrix_and_vector( mop )
	
	return MOPTyped( 
		variables,
		lower_bounds,
		upper_bounds,
		objective_functions,
		nl_eq_constraints,
		nl_ineq_constraints,
		eq_mat, ineq_mat, eq_vec, ineq_vec
	)
end

get_eq_matrix_and_vector( mop :: MOPTyped ) = (mop.eq_mat, mop.eq_vec)
get_ineq_matrix_and_vector( mop :: MOPTyped ) = (mop.ineq_mat, mop.ineq_vec)

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
get_nl_eq_constraint_indices( mop :: BothMOP ) = keys( mop.nl_eq_constraints )
get_eq_constraint_indices( mop :: MOP ) = keys( mop.eq_constraints )
get_nl_ineq_constraint_indices( mop :: BothMOP ) = keys( mop.nl_eq_constraints )
get_ineq_constraint_indices( mop :: MOP ) = keys( mop.ineq_constraints )

_get( mop :: BothMOP, ind::ObjectiveIndex ) = mop.objective_functions[ind]

function _get( mop :: BothMOP, ind :: ConstraintIndex )
	if ind.type == :nl_eq 
		return mop.nl_eq_constraints[ind]
	elseif ind.type == :nl_ineq 
		return mop.nl_ineq_constraints[ind]
	else
		return __get( mop, ind )
	end
end

function __get( mop :: MOP, ind :: ConstraintIndex )
	if ind.type == :eq
		return mop.eq_constraints[ind]
	elseif ind.type == :ineq 
		return mop.ineq_constraints[ind]
	end
end

_next_val( indices ) = isempty(indices) ? 1 : maximum( ind.value for ind in indices ) + 1 
_next_val( dict :: AbstractDict ) = _next_val( keys(dict) )

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

function _add_objective!(mop :: MOP, fun :: AbstractVecFun)
	ind = ObjectiveIndex( 
		_next_val( mop.objective_functions ),
		num_outputs( fun ) 
	)
	mop.objective_functions[ind] = fun 
	return ind
end

function _add_nl_eq_constraint!( mop :: MOP, fun :: AbstractVecFun )
	ind = ConstraintIndex( 
		_next_val( mop.nl_eq_constraints ),
		num_outputs( fun ),
		:nl_eq
	)
	mop.nl_eq_constraints[ind] = fun 
	return ind
end

function _add_nl_ineq_constraint!( mop :: MOP, fun :: AbstractVecFun )
	ind = ConstraintIndex( 
		_next_val( mop.nl_ineq_constraints ),
		num_outputs( fun ),
		:nl_ineq
	)
	mop.nl_ineq_constraints[ind] = fun 
	return ind
end

function add_eq_constraint!( mop :: MOP, aff_func :: MOI.VectorAffineFunction )
	ind = ConstraintIndex(
		_next_val( mop.eq_constraints ),
		MOI.output_dimension( aff_func ),
		:eq
	)
	mop.eq_constraints[ind] = aff_func
	return ind 
end

function add_ineq_constraint!( mop :: MOP, aff_func :: MOI.VectorAffineFunction )
	ind = ConstraintIndex(
		_next_val( mop.ineq_constraints ),
		MOI.output_dimension( aff_func ),
		:ineq
	)
	mop.ineq_constraints[ind] = aff_func
	return ind 
end
