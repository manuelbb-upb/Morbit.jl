# needs `AbstractVecFun` (which in turn needs the Surrogate Interface)
Broadcast.broadcastable( mop :: AbstractMOP ) = Ref( mop );

const VarIndIterable = Union{AbstractVector{<:VarInd}, Tuple{Vararg{<:VarInd}}}

# MANDATORY methods 

"Return a vector of `VariableIndice`s used in the model."
var_indices(:: AbstractMOP) :: AbstractVector{<:VarInd} = nothing

"Return the lower bound for the variable with Index `VarInd`."
get_lower_bound( :: AbstractMOP, :: VarInd) :: Real = nothing 
"Return the upper bound for the variable with Index `VarInd`."
get_upper_bound( :: AbstractMOP, :: VarInd) :: Real = nothing

"Return a vector or tuple of all objective indices used in the model."
get_objective_indices( :: AbstractMOP ) = ObjectiveIndex[]

# nonlinear constraints
"Return the vector or tuple of the indices of nonlinear equality constraints used in the model."
get_nl_eq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]
"Return the vector or tuple of the indices of nonlinear inequality constraints used in the model."
get_nl_ineq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]

# linear constraints (optional - either define these or 
# `get_eq_matrix_and_vector` as well as `get_ineq_matrix_and_vector`)
"Return the vector or tuple of the indices of *linear* equality constraints used in the model."
get_eq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]
"Return the vector or tuple of the indices of *linear* inequality constraints used in the model."
get_ineq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]

# for objectives and nl_constraints (and linear constraints if the "getter"s are defined)
_get( :: AbstractMOP, :: FunctionIndex ) :: AbstractVecFun = nothing

# Methods for editable models, i.e., <:AbstractMOP{true}
# optional and only for user editable problems, i.e., <:AbstractMOP{true}
add_variable!(::AbstractMOP{true}) :: VarInd = nothing

add_lower_bound!(mop :: AbstractMOP{true}, vi :: VarInd, bound :: Real ) = nothing
add_upper_bound!(mop :: AbstractMOP{true}, vi :: VarInd, bound :: Real ) = nothing
del_lower_bound!(mop :: AbstractMOP{true}, vi :: VarInd ) = nothing
del_upper_bound!(mop :: AbstractMOP{true}, vi :: VarInd ) = nothing

# The functions that are to be called by the user are derived below.
# They do not use an underscore.
"Add an objective function to the model."
_add_objective!(::AbstractMOP{true}, ::AbstractVecFun) :: ObjectiveIndex = nothing

"Add a nonlinear equality constraint function to the model."
_add_nl_eq_constraint!(::AbstractMOP{true}, ::AbstractVecFun) :: ConstraintIndex = nothing
"Add a nonlinear inequality constraint function to the model."
_add_nl_ineq_constraint!(::AbstractMOP{true}, ::AbstractVecFun) :: ConstraintIndex = nothing

"Add a linear equality constraint function to the model."
_add_eq_constraint!(::AbstractMOP{true}, :: MOI.VectorAffineFunction) :: ConstraintIndex = nothing
"Add a linear inequality constraint function to the model."
_add_ineq_constraint!(::AbstractMOP{true}, :: MOI.VectorAffineFunction) :: ConstraintIndex = nothing


# not used anywhere yet
# not implemented yet by `MOP<:AbstractMOP{true}`
"Remove a function from the MOP."
_del!(::AbstractMOP{true}, ::FunctionIndex) :: Nothing = nothing
_del!(::AbstractMOP{true}, ::VarInd) :: Nothing = nothing

# DERIVED methods 
function get_lower_bounds( mop :: AbstractMOP, indices :: VarIndIterable )
    return get_lower_bound.(mop, indices)
end

function get_upper_bounds( mop :: AbstractMOP, indices :: VarIndIterable )
    return get_upper_bound.(mop, indices)
end

"Return full vector of lower variable vectors for original problem."
function full_lower_bounds( mop :: AbstractMOP )
    return get_lower_bounds( mop, var_indices(mop) )
end

"Return full vector of upper variable vectors for original problem."
function full_upper_bounds( mop :: AbstractMOP )
    return get_upper_bounds( mop, var_indices(mop) )
end

# can be improved
num_vars( mop :: AbstractMOP ) :: Int = length(var_indices(mop))

function add_variables!(mop::AbstractMOP{true}, num_new :: Int )
    return [add_variable!(mop) for _ = 1 : num_new]
end

function get_function_indices( mop :: AbstractMOP )
    return Iterators.flatten( (
        get_objective_indices( mop ),
        get_nl_eq_constraint_indices( mop ),
        get_nl_ineq_constraint_indices( mop )
    ) )
end

#=
# TODO use a dict with SMOP
# defines: `get_objective_positions(mop, func_ind)`, `get_nl_eq_constraint_positions(mop, func_ind)`, `get_nl_ineq_constraint_positions(mop, func_ind)`
# defines: `get_objective_positions(sc, func_ind)`, `get_nl_eq_constraint_positions(sc, func_ind)`, `get_nl_ineq_constraint_positions(sc, func_ind)`
for (out_type, T) in zip([:objective, :eq_constraint, :ineq_constraint], 
        [:ObjectiveIndex, :EqConstraintIndex, :IneqConstraintIndex ])
    func_name = Symbol("get_$(out_type)_positions")
    @eval begin 
        function $(func_name)( mop :: Union{AbstractMOP,AbstractSurrogateContainer}, func_ind :: $(T))
            first_pos = findfirst( ind -> ind == func_ind, get_objective_indices( mop ) )
            if isnothing(first_pos)
                return nothing
            end
            return CartesianIndex((first_pos : first_pos + num_outputs( func_ind ) - 1)...)
        end
    end
end
=#

function full_bounds( mop :: AbstractMOP )
    (full_lower_bounds(mop), full_upper_bounds(mop))
end

function full_vector_bounds( mop :: AbstractMOP )
    lb, ub = full_bounds(mop)
    return (collect(lb), collect(ub))
end
#=
function _width( mop :: AbstractMOP )
    lb, ub = full_bounds(mop)
    return ub .- lb
end
=#

"Return a list of `AbstractVectorObjective`s."
function list_of_objectives( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_objective_indices(mop) ]
end

function list_of_nl_eq_constraints( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_nl_eq_constraint_indices(mop) ]
end

function list_of_nl_ineq_constraints( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_nl_ineq_constraint_indices(mop) ]
end

function list_of_functions( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_function_indices(mop) ]
end

"Number of scalar-valued objectives of the problem."
function num_objectives( mop :: AbstractMOP )
    isempty(get_objective_indices(mop)) && return 0
    return sum( num_outputs(func_ind) for func_ind = get_objective_indices(mop))
end

function _sum_inds( inds )
    isempty(inds) && return 0
    return sum( num_outputs(func_ind) for func_ind = inds )
end

function num_eq_constraints( mop :: AbstractMOP )
    return _sum_inds( get_eq_constraint_indices( mop ) )
end

function num_ineq_constraints( mop :: AbstractMOP )
    _sum_inds(get_ineq_constraint_indices(mop))
end

function num_nl_eq_constraints( mop :: AbstractMOP )
    _sum_inds(get_nl_eq_constraint_indices(mop))
end

function num_nl_ineq_constraints( mop :: AbstractMOP )
    _sum_inds(get_nl_ineq_constraint_indices(mop)) 
end

num_nl_constraints( mop :: AbstractMOP ) = num_nl_eq_constraints(mop) + num_nl_ineq_constraints(mop)
num_lin_constraints( mop :: AbstractMOP ) = num_eq_constraints(mop) + num_ineq_constraints(mop)

# more convenient functions to add functions to an editable model
for func_name in [:add_objective!, :add_nl_eq_constraint!, :add_nl_ineq_constraint!]
    @eval begin
        function $(func_name)( 
                mop :: AbstractMOP{true},
                func :: Function; kwargs...) 

            objf = make_vec_fun( func; kwargs... )
            return $(Symbol("_$(func_name)"))(mop, objf)
        end
    end
end

# `eval_objf` respects custom batching

# ## Evaluation of an AbstractMOP,
# should be improved in implementations
# most important:
#     evaluate_at_unscaled_site


# helpers
function _eval_at_index_at_unscaled_site(mop, ind, x)
    return eval_objf( _get(mop, ind), x ) 
end

function _eval_at_indices_at_unscaled_site( mop, indices, x )
    return Dictionary(
        indices,
        _eval_at_index_at_unscaled_site(mop, ind, x) for ind = indices 
    )
end

function _flatten_mop_dict( eval_dict, _indices = nothing )
    indices = isnothing(_indices) ? keys(eval_dict) : _indices
    if isempty(indices) || isempty(eval_dict)
        return MIN_PRECISION[]
    end
    return ensure_precision(collect( Iterators.flatten( eval_dict[ind] for ind = indices )))
end

_flatten_mop_dicts( args... ) = _flatten_mop_dict.(args)

for fntype = [:nl_function, :objective, :nl_eq_constraints, :nl_ineq_constraints ]
    get_XXX_indices = Symbol("get_", fntype, "_indices")
    _eval_XXXs_at_unscaled_site = Symbol("_eval_$(fntype)s_at_unscaled_site")
    eval_XXXs_to_vec_at_unscaled_site = Symbol("eval_$(fntype)s_to_vec_at_unscaled_site")
    @eval begin
        function $(_eval_XXXs_at_unscaled_site)( mop, x )
            return _eval_at_indices_at_unscaled_site( mop, $(get_XXX_indices)(mop), x )
        end
        function $(eval_XXXs_to_vec_at_unscaled_site)(mop, x)
            return _flatten_mop_dict( _eval_XXXs_to_vec_at_unscaled_site(mop,x) )
        end
    end
end

function _evaluate_at_unscaled_site( mop, x )
    return (
        _eval_nl_functions_at_unscaled_site( mop, x ),
        _eval_objectives_at_unscaled_site( mop, x ),
        _eval_nl_eq_constraints_at_unscaled_site( mop, x ),
        _eval_nl_ineq_constraints_at_unscaled_site( mop, x ),
    )
end

# MAIN METHOD for complete evaluation of a problem
evaluate_at_unscaled_site(mop,x) = _evaluate_at_unscaled_site(mop,x)

function evaluate_to_vecs_at_unscaled_site( mop, x )
    return _flatten_mop_dict.( evaluate_at_unscaled_site( mop, x ) )
end

function evaluate_at_scaled_site( mop, scal, x_scaled )
    return evaluate_at_unscaled_site( mop, untransform(scal, x_scaled ) )
end

function evaluate_to_vecs_at_scaled_site( mop, scal, x_scaled ) 
    return evaluate_to_vecs_at_unscaled_site( mop, untransform( scal, x_scaled) )
end
#=
"""
    eval_dict_mop_at_func_indices_at_unscaled_sites(mop, sites, func_indices)

Return a Dict with keys `func_indices` and values that are Vector-of-Vectors 
of the corresponding objectives of `mop` evaluated on all `sites`."
"""
function eval_dict_mop_at_func_indices_at_unscaled_sites( 
        mop :: AbstractMOP, func_indices, X_unscaled :: VecVec
    )
    return Base.Dict( func_ind => eval_objf.( _get(mop, func_ind), X_unscaled ) for func_ind=func_indices )
end

"""
    eval_dict_mop_at_func_indices_at_unscaled_site(mop, x, func_indices)

Return a Dict with keys `func_indices` and values that are Vectors 
of the corresponding objectives of `mop` evaluated at `x`."
"""
function eval_dict_mop_at_func_indices_at_unscaled_site( 
        mop :: AbstractMOP, func_indices, x_unscaled :: Vec
    )
    return Base.Dict( func_ind => eval_objf( _get(mop, func_ind), x_unscaled ) for func_ind=func_indices )
end

function flatten_mop_dicts( eval_dicts, _func_indices = nothing )
   N = length(first(values(eval_dicts)))
   func_indices = isnothing(_func_indices) ? keys(eval_dicts) : _func_indices
   return flatten_vecs.( [eval_dicts[find][j] for find=func_indices] for j=1:N )
end

function flatten_mop_dict( eval_dict :: Union{AbstractDict{K,V}, AbstractDictionary{K,V}}, _func_indices = nothing ) where{K,V}
    func_indices = isnothing(_func_indices) ? keys(eval_dict) : _func_indices
    if isempty( func_indices )
        return eltype(V)[]
    else
        return flatten_vecs( [eval_dict[find] for find=func_indices] )
    end
end

function eval_vec_mop_at_func_indices_at_unscaled_sites( 
        mop :: AbstractMOP, func_indices, X_unscaled :: VecVec
    )
    return flatten_mop_dicts(
        eval_dict_mop_at_func_indices_at_unscaled_sites(mop, func_indices, X_unscaled), 
        func_indices
    )
end

function eval_vec_mop_at_func_indices_at_unscaled_site( 
        mop :: AbstractMOP, func_indices, X_unscaled :: Vec
    )
    return flatten_mop_dict(
        eval_dict_mop_at_func_indices_at_unscaled_site(mop, func_indices, X_unscaled), 
        func_indices
    )
end

function eval_dict_mop_at_unscaled_site( mop :: AbstractMOP, x :: Vec )
    all_func_inds = get_function_indices(mop)
    return eval_dict_mop_at_func_indices_at_unscaled_site(mop, all_func_inds, x)
end

function eval_result_to_all_vectors( 
        eval_dict :: Union{AbstractDict, AbstractDictionary},  mop :: Union{AbstractSurrogateContainer,AbstractMOP} 
    )
    return (
        flatten_mop_dict( eval_dict, get_objective_indices(mop) ),
        flatten_mop_dict( eval_dict, get_nl_eq_constraint_indices(mop) ),
        flatten_mop_dict( eval_dict, get_nl_ineq_constraint_indices(mop) )
    )
end    

for func_name = [:eval_dict_mop_at_func_indices, :eval_vec_mop_at_func_indices]
    fn_scaled_sites = Symbol(func_name, "_at_scaled_sites")
    fn_scaled_site = Symbol(func_name, "_at_scaled_site")
    fn_unscaled_sites = Symbol(func_name, "_at_unscaled_sites")
    fn_unscaled_site = Symbol(func_name, "_at_unscaled_site")
    @eval begin 
        function $(fn_scaled_sites)(mop :: AbstractMOP, func_indices, X_scaled :: VecVec, scal :: AbstractVarScaler )
            X_unscaled = untransform.(X_scaled, scal)
            return $(fn_unscaled_sites)( mop, func_indices, X_unscaled )
        end

        function $(fn_scaled_site)(mop :: AbstractMOP, func_indices, X_scaled :: Vec, scal :: AbstractVarScaler )
            X_unscaled = untransform(X_scaled, scal)
            return $(fn_unscaled_site)( mop, func_indices, X_unscaled )
        end
    end
end
=#

#=
# defined below:
# eval_vec_mop_objectives_at_scaled_site(s)
# eval_vec_mop_nl_eq_constraints_at_scaled_site(s)
# eval_vec_mop_nl_ineq_constraints_at_scaled_site(s)
for eval_type in [:objective, :nl_eq_constraint, :nl_ineq_constraint]
    for suffix1 in [:scaled_site, :unscaled_site]
        for suffix2 in ["", "s"]
            getter = Symbol("get_$(eval_type)_indices")
            base_fn = Symbol("eval_vec_mop_at_func_indices_at_",suffix1, suffix2)
            fn_new = Symbol( "eval_vec_mop_", eval_type, "s_at_", suffix1, suffix2)
            @eval begin
                function $(fn_new)(mop, x, args...)
                    func_indices = $(getter)(mop)
                    return $(base_fn)(mop, func_indices, x, args...)
                end
            end
        end
    end
end
=#

function eval_linear_constraints_at_unscaled_site( x, mop )
    #A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints( scal, mop )
    A_eq, b_eq = get_eq_matrix_and_vector( mop )
    A_ineq, b_ineq = get_ineq_matrix_and_vector( mop )
    return (A_eq * x .+ b_eq, A_ineq * x + b_ineq)
end

function eval_linear_constraints_at_scaled_site( x_scaled, mop, scal )
    A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints( scal, mop )
    return (A_eq * x_scaled .+ b_eq, A_ineq * x_scaled + b_ineq)
end

# Helper functions …
function num_evals( mop :: AbstractMOP ) :: Vector{Int}
    [ num_evals(objf) for objf ∈ list_of_functions(mop) ]
end

@doc "Set evaluation counter to 0 for each VecFun in `m.vector_of_objectives`."
function reset_evals!(mop :: AbstractMOP) :: Nothing
    for objf ∈ list_of_functions( mop )
        wrapped_function(objf).counter[] = 0
    end
    return nothing
end

####### helpers for linear constraints

function _scalar_to_vector_aff_func( aff_func :: MOI.ScalarAffineFunction )
    vec_terms = [ MOI.VectorAffineTerm( 1, term) for term = aff_func.terms ]
    consts = [ aff_func.constant, ]
    return MOI.VectorAffineFunction( vec_terms, consts )
end

function add_eq_constraint!( mop :: AbstractMOP{true}, aff_func :: MOI.ScalarAffineFunction )
    return add_eq_econstraint!(mop, _scalar_to_vector_aff_func(aff_func) )
end

function add_ineq_constraint!( mop :: AbstractMOP{true}, aff_func :: MOI.ScalarAffineFunction )
    return add_ineq_econstraint!(mop, _scalar_to_vector_aff_func(aff_func) )
end

# for easier user input:

function _matrix_to_vector_affine_function( A :: AbstractMatrix{F}, b :: AbstractVector{T}, vars :: AbstractVector{<:VarInd} ) where{F<:Number, T<:Number}
	m, n = size(A)
	@assert n == length(vars) "`A` must have the same number of columns as there are `vars`."
	@assert m == length(b) "`A` must have the same number of rows as there are entries in `b`."

	S = Base.promote_type(F, T)
	terms = collect(Iterators.flatten(
		[ [ MOI.VectorAffineTerm( i, MOI.ScalarAffineTerm( S(row[j]), vars[j] ) ) for j = 1:n ] for (i,row) = enumerate( eachrow(A) ) ] ))
	constants = S.(b)
	return MOI.VectorAffineFunction( terms, constants )
end

function add_ineq_constraint!(mop :: AbstractMOP{true}, A :: AbstractMatrix, b :: AbstractVector = [], vars :: Union{Nothing,AbstractVector{<:VarInd}} = nothing)
	_vars = isnothing( vars ) ? var_indices(mop) : vars
    _b = isempty( b ) ? zeros( Bool, size(A,1) ) : b
	return add_ineq_constraint!(mop, 
		_matrix_to_vector_affine_function( A, _b, _vars )
	)
end

function add_eq_constraint!(mop :: AbstractMOP{true}, A :: AbstractMatrix, b :: AbstractVector, vars :: Union{Nothing,AbstractVector{<:VarInd}} = nothing)
	_vars = isnothing( vars ) ? var_indices(mop) : vars
    _b = isempty( b ) ? zeros( Bool, size(A,1) ) : b
	return add_ineq_constraint!(mop, 
		_matrix_to_vector_affine_function( A, _b, _vars )
	)
end

import SparseArrays: sparse

function _num_type_of_any_array( arr :: Vector )
    n = length(arr)
    
    if isempty(arr)
        return Any
    else
        if n == 1
            return typeof(arr[1])
        else
            return Base.promote_type( typeof(arr[1]), _num_type_of_any_array(arr[2:end]) )
        end
    end
end

# TODO: MathOptInterface v0.10 and higher will need 
# `term.scalar_term.variable` instead of `term.scalar_term.variable_index`
function _construct_constraint_matrix_and_vector( vec_affine_funcs, vars )
    ## vars = sort( vars, lt = (x,y) -> Base.isless(x.value, y.value ) )
    var_pos_dict = Dict( v => i for (i,v) = enumerate( vars) )
    row_inds = Int[]
    col_inds = Int[]
    vals = Any[]
    offset = 0
    b_parts = Any[]
    for vaf in vec_affine_funcs
        for term in vaf.terms
            push!( row_inds, term.output_index + offset )
            push!( col_inds, var_pos_dict[ term.scalar_term.variable_index ] )
            push!( vals, term.scalar_term.coefficient )
        end
		offset += MOI.output_dimension(vaf)
        push!(b_parts, vaf.constants ) 
    end

    coeff_type = _num_type_of_any_array( vals )
    coeff = Vector{coeff_type}( vals )

    return sparse( row_inds, col_inds, coeff ), vcat( b_parts... )
end

# These matrices are constant and we should assume that the
# functions are called late enough
@memoize ThreadSafeDict function get_eq_matrix_and_vector( mop :: AbstractMOP )
    eq_indices = get_eq_constraint_indices(mop)
    if isempty(eq_indices)
        return Matrix{Bool}(undef,0, num_vars(mop) ), Vector{Bool}(undef,0)
    else
        return _construct_constraint_matrix_and_vector( 
            [ _get( mop, ind ) for ind = eq_indices ],
            var_indices(mop)
        )
    end
end

@memoize ThreadSafeDict function get_ineq_matrix_and_vector( mop :: AbstractMOP )
	ineq_indices = get_ineq_constraint_indices(mop)
    if isempty(ineq_indices)
        return Matrix{Bool}(undef,0, num_vars(mop) ), Vector{Bool}(undef,0)
    else
        return _construct_constraint_matrix_and_vector( 
            [ _get( mop, ind ) for ind = ineq_indices ],
            var_indices(mop)
        )
    end
end

## Scaled constraints 

function _transform_linear_constraints( A, b, Tinv, offset )
    _A = A*Tinv
    return _A, b - _A * offset  
end

function transformed_linear_eq_constraints(scal, mop)
    A, b = get_eq_matrix_and_vector( mop )

    Tinv = unscaling_matrix(scal)
    offset = scaling_offset(scal)

    return _transform_linear_constraints( A, b, Tinv, offset)
end

function transformed_linear_ineq_constraints(scal, mop)
    A, b = get_ineq_matrix_and_vector( mop )

    Tinv = unscaling_matrix(scal)
    offset = scaling_offset(scal)

    return _transform_linear_constraints( A, b, Tinv, offset)
end

@memoize ThreadSafeDict function transformed_linear_constraints( scal, mop )
    return (transformed_linear_eq_constraints(scal, mop)..., transformed_linear_ineq_constraints(scal,mop)...)
end

# Function handles for NLopt
function _get_optim_handle( mat_row, offset )
    opt_fun = function( x, g )
        if !isempty(g)
            g[:] .= mat_row[:]
        end
        return mat_row'x .+ offset
    end
    return opt_fun
end

function get_eq_constraints_optim_handles( mop, scal )
    A, b = transformed_linear_eq_constraints( scal, mop )
    return [_get_optim_handle(A[i,:], b[i]) for i = 1:length(b)]
end
 
function get_ineq_constraints_optim_handles( mop, scal )
    A, b = transformed_linear_ineq_constraints( scal, mop )
    return [_get_optim_handle(A[i,:], b[i]) for i = 1:length(b)]
end

# pretty printing
_is_editable(::Type{<:AbstractMOP{T}}) where T = T 

function Base.show(io::IO, mop :: M) where M<:AbstractMOP
    str = "$( _is_editable(M) ? "Editable" : "Non-editable") MOP of Type $(_typename(M)). "
    if !get(io, :compact, false)
        str *= """There are 
        * $(num_vars(mop)) variables and $(num_objectives(mop)) objectives,
        * $(num_nl_eq_constraints(mop)) nonlinear equality and $(num_nl_ineq_constraints(mop)) nonlinear inequality constraints,
        * $(num_eq_constraints(mop)) linear equality and $(num_ineq_constraints(mop)) linear inequality constraints.
        The lower bounds and upper variable bounds are 
        $(_prettify(full_lower_bounds(mop), 5))
        $(_prettify(full_upper_bounds(mop), 5))"""
    end
    print(io, str)
end