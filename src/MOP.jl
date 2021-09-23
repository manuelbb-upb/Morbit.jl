# this is NOT a MOI.ModelLike !!
# but I tried to design it with MOI in mind 
# to have an easy time writing a wrapper eventually
using DataStructures: SortedDict

@with_kw struct MOP <: AbstractMOP{true}
	variables :: Vector{VarInd} = []

	lower_bounds :: Dict{VarInd, Real} = Dict()
	upper_bounds :: Dict{VarInd, Real} = Dict()

	objective_functions :: SortedDict{ObjectiveIndex, AbstractVecFun} = Dict()
	nl_eq_constraints :: SortedDict{EqConstraintIndex, AbstractVecFun} = Dict()
	nl_ineq_constraints :: SortedDict{IneqConstraintIndex, AbstractVecFun} = Dict()
end

struct MOPTyped{
		VarType <: Tuple{Vararg{<:VarInd}},
		LbType <: AbstractDict{VarInd,<:AbstractFloat},
		UbType <: AbstractDict{VarInd,<:AbstractFloat},
		ObjfType <: AbstractDict,#{ObjectiveIndex,<:Union{AbstractVecFun,Nothing}},
		EqType <: AbstractDict,#{EqConstraintIndex,<:Union{AbstractVecFun,Nothing}},
		IneqType <: AbstractDict,#{IneqConstraintIndex,<:Union{AbstractVecFun,Nothing}},
		JacType <: AbstractMatrix
	} <: AbstractMOP{false}

	variables :: VarType
	lower_bounds :: LbType
	upper_bounds :: UbType 
	objective_functions :: ObjfType
	nl_eq_constraints :: EqType
	nl_ineq_constraints :: IneqType
	unscaling_jacobian :: JacType
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
	nl_eq_constraints = let indices = get_eq_constraint_indices(mop);
		if isempty(indices)
			Base.ImmutableDict{EqConstraintIndex, Nothing}()
		else
			SortedDict( ind => _get(mop, ind) for ind = indices ) 
		end
	end
	nl_ineq_constraints = let indices = get_ineq_constraint_indices(mop);
		if isempty(indices)
			Base.ImmutableDict{IneqConstraintIndex, Nothing}()
		else
			SortedDict( ind => _get(mop, ind) for ind = indices )
		end
	end
	return MOPTyped( 
		variables,
		lower_bounds,
		upper_bounds,
		objective_functions,
		nl_eq_constraints,
		nl_ineq_constraints,
		jacobian_of_unscaling(rand(num_vars(mop)), mop)  # does only work, because scaling is constant
	)
end

function jacobian_of_unscaling( x_scaled :: Vec, mop :: MOPTyped)
	return mop.unscaling_jacobian
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
get_eq_constraint_indices( mop :: BothMOP ) = keys( mop.nl_eq_constraints )
get_ineq_constraint_indices( mop :: BothMOP ) = keys( mop.nl_ineq_constraints )

_get( mop :: BothMOP, ind::ObjectiveIndex ) = mop.objective_functions[ind]
_get( mop :: BothMOP, ind::EqConstraintIndex ) = mop.objective_functions[ind]
_get( mop :: BothMOP, ind::IneqConstraintIndex ) = mop.objective_functions[ind]

_next_val( indices ) = isempty(indices) ? 1 : maximum( ind.val for ind in indices ) + 1 
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

function _add_eq_constraint!( mop :: MOP, fun :: AbstractVecFun )
	ind = EqConstraintIndex( 
		_next_val( mop.objective_functions ),
		num_outputs( fun )
	)
	mop.eq_constraints[ind] = fun 
	return ind
end

function _add_ineq_constraint!( mop :: MOP, fun :: AbstractVecFun )
	ind = IneqConstraintIndex( 
		_next_val( mop.objective_functions ),
		num_outputs( fun )
	)
	mop.ineq_constraints[ind] = fun 
	return ind
end
