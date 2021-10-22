# needs `AbstractVecFun` (which in turn needs the Surrogate Interface)
Broadcast.broadcastable( mop :: AbstractMOP ) = Ref( mop );

# MANDATORY methods
const VarIndIterable = Union{AbstractVector{<:VarInd}, Tuple{Vararg{<:VarInd}}}

var_indices(:: AbstractMOP) :: AbstractVector{<:VarInd} = nothing

get_lower_bound( :: AbstractMOP, :: VarInd) :: Real = nothing 
get_upper_bound( :: AbstractMOP, :: VarInd) :: Real = nothing

get_objective_indices( :: AbstractMOP ) = ObjectiveIndex[]

# nonlinear constraints
get_nl_eq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]
get_nl_ineq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]

# linear constraints (optional - either define these or 
# `get_eq_matrix_and_vector` as well as `get_ineq_matrix_and_vector`)
get_eq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]
get_ineq_constraint_indices( :: AbstractMOP ) = ConstraintIndex[]

# for objectives and nl_constraints (and linear constraints if the "getter"s are defined)
_get( :: AbstractMOP, :: FunctionIndex ) :: AbstractVecFun = nothing

# optional and only for user editable problems, i.e. <:AbstractMOP{true}
add_variable!(::AbstractMOP{true}) :: VarInd = nothing

add_lower_bound!(mop :: AbstractMOP{true}, vi :: VarInd, bound :: Real ) = nothing
add_upper_bound!(mop :: AbstractMOP{true}, vi :: VarInd, bound :: Real ) = nothing
del_lower_bound!(mop :: AbstractMOP{true}, vi :: VarInd ) = nothing
del_upper_bound!(mop :: AbstractMOP{true}, vi :: VarInd ) = nothing

"Add an objective function to MOP"
_add_objective!(::AbstractMOP{true}, ::AbstractVecFun) :: ObjectiveIndex = nothing

_add_nl_eq_constraint!(::AbstractMOP{true}, ::AbstractVecFun) :: ConstraintIndex = nothing
_add_nl_ineq_constraint!(::AbstractMOP{true}, ::AbstractVecFun) :: ConstraintIndex = nothing

add_eq_constraint!(::AbstractMOP{true}, :: MOI.VectorAffineFunction) :: ConstraintIndex = nothing
add_ineq_constraint!(::AbstractMOP{true}, :: MOI.VectorAffineFunction) :: ConstraintIndex = nothing

# not used anywhere yet
# not implemented yet by `MOP<:AbstractMOP{true}`
"Remove a function from the MOP."
_del!(::AbstractMOP{true}, ::FunctionIndex) :: Nothing = nothing
_del!(::AbstractMOP{true}, ::VarInd) :: Nothing = nothing

# DERIVED methods 

get_lower_bounds( mop :: AbstractMOP, indices :: VarIndIterable ) = get_lower_bound.(mop, indices)
get_upper_bounds( mop :: AbstractMOP, indices :: VarIndIterable ) = get_upper_bound.(mop, indices)

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

function _width( mop :: AbstractMOP )
    lb, ub = full_bounds(mop)
    return ub .- lb
end

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

for func_name in [:add_objective!, :add_nl_eq_constraint!, :add_nl_ineq_constraint!]
    @eval begin
        function $(func_name)( 
                mop :: AbstractMOP{true}, T :: Type{<:AbstractVecFun},
                func :: Function; kwargs...) 

            objf = _wrap_func( T, func; kwargs... )
            return $(Symbol("_$(func_name)"))(mop, objf)
        end
    end
end

function eval_vec_mop_at_func_index_at_unscaled_site( 
        mop :: AbstractMOP, x_unscaled :: Vec, func_ind :: FunctionIndex 
    )
    objf = _get(mop, func_ind)
    y = eval_objf( objf, x_unscaled )
    return y
end

function eval_vec_mop_at_func_index_at_scaled_site( 
        mop :: AbstractMOP, scal :: AbstractVarScaler, x_scaled :: Vec, 
        func_ind :: FunctionIndex 
    )
    x = untransform( x_scaled, scal )
    return eval_vec_mop_at_func_index_at_unscaled_site(mop, x, func_ind )
end

function eval_mop_at_func_indices_at_unscaled_site( 
        mop :: AbstractMOP, x :: Vec, func_indices  
    )
    isempty(func_indices) && return Base.ImmutableDict{FunctionIndexTuple,MIN_PRECISION}()
    return Base.ImmutableDict( 
        (func_ind => eval_vec_mop_at_func_index_at_unscaled_site( mop, x, func_ind ) for func_ind in func_indices)...
    )
end 

function eval_mop_at_func_indices_at_scaled_site( 
        mop :: AbstractMOP, scal :: AbstractVarScaler, x_scaled :: Vec, 
        func_indices  
    )
    x = untransform( x_scaled, scal )
    return eval_mop_at_func_indices_at_unscaled_site( mop, x, func_indices )
end

function eval_vec_mop_at_func_indices_at_scaled_site(
        mop :: AbstractMOP, scal :: AbstractVarScaler, x_scaled :: Vec, 
        func_indices  
    )
    return eval_result_to_vector( 
        mop, 
        eval_mop_at_func_indices_at_scaled_site( mop, scal, x_scaled, func_indices ) 
    )
end

function eval_mop_at_unscaled_site(
        mop :: AbstractMOP, x :: Vec
    ) 
    return eval_mop_at_func_indices_at_unscaled_site(mop, x, get_function_indices(mop))
end

#=
function eval_vec_mop_at_func_indices_at_unscaled_site( 
        mop :: AbstractMOP, x_scaled :: Vec, 
        func_indices )
    tmp = eval_mop_at_func_indices_at_unscaled_site( mop, x_scaled, func_indices )
    return flatten_vecs(values(tmp))
end
=#

function eval_result_to_vector( mop :: AbstractMOP, eval_result :: AbstractDict )
    return eval_result_to_vector(eval_result, get_function_indices(mop))
end

function eval_result_to_vector( eval_result :: AbstractDict, func_indices  )
    return flatten_vecs( 
        get( eval_result, func_ind, MIN_PRECISION[] ) for func_ind in func_indices
    )  
end

function eval_result_to_all_vectors( 
        eval_result :: AbstractDict,  mop :: Union{AbstractSurrogateContainer,AbstractMOP} 
    )
    return (
        eval_result_to_vector( eval_result, get_objective_indices(mop) ),
        eval_result_to_vector( eval_result, get_nl_eq_constraint_indices(mop) ),
        eval_result_to_vector( eval_result, get_nl_ineq_constraint_indices(mop) )
    )
end    

# custom broadcasting to exploit inner batching in `eval_missing!`
function Broadcast.broadcasted( ::typeof( eval_vec_mop_at_func_index_at_unscaled_site ), 
        mop :: AbstractMOP, X_unscaled :: VecVec, func_ind :: FunctionIndex )
    objf = _get(mop, func_ind)
    return eval_objf.( objf, X_unscaled )
end

function Broadcast.broadcasted( ::typeof(eval_mop_at_func_indices_at_scaled_site), 
        mop :: AbstractMOP, scal :: AbstractVarScaler, X_scaled :: VecVec, func_ind :: FunctionIndex )
    X = untransform.( X_scaled, scal)
    return eval_vec_mop_at_func_index_at_unscaled_site.(mop, X, Ref(func_ind ))
end

function Broadcast.broadcasted( ::typeof(eval_vec_mop_at_func_indices_at_scaled_site), 
        mop :: AbstractMOP, scal :: AbstractVarScaler, X_scaled :: VecVec, func_ind :: FunctionIndex )
    return eval_result_to_vector.( 
        mop,
        eval_vec_mop_at_func_index_at_unscaled_site.(mop, X, Ref(func_ind) )
    )
end

# defined below:
# eval_mop_objectives_at_scaled_site
# eval_mop_nl_eq_constraints_at_scaled_site
# eval_mop_nl_ineq_constraints_at_scaled_site
# eval_vec_mop_objectives_at_scaled_site
# eval_vec_mop_nl_eq_constraints_at_scaled_site
# eval_vec_mop_nl_ineq_constraints_at_scaled_site

for eval_type in [:objective, :nl_eq_constraint, :nl_ineq_constraint]
    fn1 = Symbol("eval_mop_", eval_type, "s_at_scaled_site")
    fn2 = Symbol("eval_vec_mop_", eval_type, "s_at_scaled_site")

    fn3 =  Symbol("eval_mop_", eval_type, "s_at_unscaled_site")
    fn4 = Symbol("eval_vec_mop_", eval_type, "s_at_unscaled_site")
    getter_fun = Symbol("get_$(eval_type)_indices")
    @eval begin
        function $(fn1)(
                mop :: AbstractMOP, scal :: AbstractVarScaler, x_scaled :: Vec )
            return eval_mop_at_func_indices_at_scaled_site( 
                mop, scal, x_scaled, $(getter_fun)(mop) 
            )
        end

        function $(fn2)( 
                mop :: AbstractMOP, scal :: AbstractVarScaler, x_scaled :: Vec )
            tmp_res = $(fn1)( mop, scal, x_scaled )
            return eval_result_to_vector( mop, tmp_res )
        end

        function Base.broadcastable( :: typeof($(fn1)), 
            mop :: AbstractMOP, scal :: AbstractVarScaler, X_scaled :: VecVec)
            X_unscaled = untransform.( X_scaled, scal )
            return eval_mop_at_func_indices_at_unscaled_site.(
                mop, X_unscaled, Ref($(getter_fun)(mop))
            )
        end

        function Base.broadcastable( ::typeof($(fn2)), 
            mop :: AbstractMOP, scal :: AbstractVarScaler, X_scaled :: VecVec )
            tmp_res = $(fn1).(mop, scal, X_scaled)
            return eval_result_to_vector.(mop, tmp_res)
        end

        # unscaled 
        function $(fn3)( mop :: AbstractMOP, x :: Vec )
            return eval_mop_at_func_indices_at_unscaled_site( 
                mop, x, $(getter_fun)(mop) 
            )
        end

        function $(fn4)( mop :: AbstractMOP, x :: Vec )
            tmp_res = $(fn3)( mop, x )
            return eval_result_to_vector( mop, tmp_res )
        end

        function Base.broadcastable( :: typeof($(fn3)), mop :: AbstractMOP, X :: VecVec)
            return eval_mop_at_func_indices_at_unscaled_site.(
                mop, X, Ref($(getter_fun)(mop))
            )
        end

        function Base.broadcastable( ::typeof($(fn4)), mop :: AbstractMOP, X :: VecVec )
            tmp_res = $(fn3).(mop, X)
            return eval_result_to_vector.(mop, tmp_res)
        end

    end
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