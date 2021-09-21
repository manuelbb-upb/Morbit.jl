# needs `AbstractVecFun` (which in turn needs the Surrogate Interface)
Broadcast.broadcastable( mop :: AbstractMOP ) = Ref( mop );

# MANDATORY methods
const VarIndIterable = Union{AbstractVector{<:VarInd}, Tuple{Vararg{<:VarInd}}}

var_indices(:: AbstractMOP) :: AbstractVector{<:VarInd} = nothing

get_lower_bound( :: AbstractMOP, :: VarInd) :: Real = nothing 
get_upper_bound( :: AbstractMOP, :: VarInd) :: Real = nothing 

get_objective_indices( :: AbstractMOP ) = ObjectiveIndex[]
get_eq_constraint_indices( :: AbstractMOP ) = EqConstraintIndex[]
get_ineq_constraint_indices( :: AbstractMOP ) = IneqConstraintIndex[]

_get( :: AbstractMOP, :: FunctionIndex ) :: AbstractVecFun = nothing

# optional and only for user editable problems, i.e. <:AbstractMOP{true}
add_lower_bound!(mop :: AbstractMOP{true}, vi :: VarInd, bound :: Real ) = nothing
add_upper_bound!(mop :: AbstractMOP{true}, vi :: VarInd, bound :: Real ) = nothing
del_lower_bound!(mop :: AbstractMOP{true}, vi :: VarInd ) = nothing
del_upper_bound!(mop :: AbstractMOP{true}, vi :: VarInd ) = nothing

"Add an objective function to MOP"
_add_objective!(::AbstractMOP{true}, ::AbstractVecFun) :: ObjectiveIndex = nothing
_add_eq_constraint!(::AbstractMOP{true}, ::AbstractVecFun) :: EqConstraintIndex = nothing
_add_ineq_constraint!(::AbstractMOP{true}, ::AbstractVecFun) :: IneqConstraintIndex = nothing

add_variable!(::AbstractMOP{true}) :: VarInd = nothing

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
        get_eq_constraint_indices( mop ),
        get_ineq_constraint_indices( mop )
    ) )
end

#=
# TODO use a dict with SMOP
# defines: `get_objective_positions(mop, func_ind)`, `get_eq_constraint_positions(mop, func_ind)`, `get_ineq_constraint_positions(mop, func_ind)`
# defines: `get_objective_positions(sc, func_ind)`, `get_eq_constraint_positions(sc, func_ind)`, `get_ineq_constraint_positions(sc, func_ind)`
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
"Return lower variable bounds for scaled variables."
function full_lower_bounds_internal( mop :: AbstractMOP )
    [ isinf(l) ? l : 0 for l ∈ full_lower_bounds(mop) ];
end

"Return upper variable bounds for scaled variables."
function full_upper_bounds_internal( mop :: AbstractMOP )
    [ isinf(u) ? u : 1 for u ∈ full_upper_bounds(mop) ];
end

function full_bounds( mop :: AbstractMOP )
    (full_lower_bounds(mop), full_upper_bounds(mop))
end

function full_bounds_internal( mop :: AbstractMOP )
    (full_lower_bounds_internal(mop), full_upper_bounds_internal(mop))
end


"Return a list of `AbstractVectorObjective`s."
function list_of_objectives( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_objective_indices(mop) ]
end

function list_of_eq_constraints( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_eq_constraint_indices(mop) ]
end

function list_of_ineq_constraints( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_ineq_constraint_indices(mop) ]
end

function list_of_functions( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = get_function_indices(mop) ]
end

"Number of scalar-valued objectives of the problem."
function num_objectives( mop :: AbstractMOP )
    isempty(get_objective_indices(mop)) && return 0
    return sum( num_outputs(func_ind) for func_ind = get_objective_indices(mop))
end

function num_eq_constraints( mop :: AbstractMOP )
    isempty(get_eq_constraint_indices(mop)) && return 0
    return sum( num_outputs(func_ind) for func_ind = get_eq_constraint_indices(mop))
end

function num_ineq_constraints( mop :: AbstractMOP )
    isempty(get_ineq_indices(mop)) && return 0
    return sum( num_outputs(func_ind) for func_ind = get_ineq_constraint_indices(mop))
end

"Scale variables fully constrained to a closed interval to [0,1] internally."
function scale( x :: Vec, mop :: AbstractMOP )
    x̂ = copy(x);
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _scale!(x̂, lb, ub);
    return x̂
end

"Reverse scaling for fully constrained variables from [0,1] to their former domain."
function unscale( x̂ :: Vec, mop :: AbstractMOP )
    x = copy(x̂);
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _unscale!(x, lb, ub);
    return x
end

function scale!( x :: Vec, mop :: AbstractMOP )
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _scale!(x, lb, ub);    
end

function unscale!( x̂ :: Vec, mop :: AbstractMOP )
    lb, ub = full_lower_bounds(mop), full_upper_bounds(mop);
    _unscale!( x̂, lb, ub);
end

"Local bounds vectors `lb_eff` and `ub_eff` using scaled variable constraints from `mop`."
function local_bounds( mop :: AbstractMOP, x :: Vec, Δ :: Union{Real, Vec} )
    lb, ub = full_lower_bounds_internal( mop ), full_upper_bounds_internal( mop );
    return _local_bounds( x, Δ, lb, ub );
end

function intersect_bounds( mop :: AbstractMOP, x :: Vec, Δ :: Union{Real, Vec}, 
    d :: Vec; return_vals :: Symbol = :both )
    lb_eff, ub_eff = local_bounds( mop, x, Δ );
    return intersect_bounds( x, d, lb_eff, ub_eff; return_vals )
end

for func_name in [:add_objective!, :add_eq_constraint!, :add_ineq_constraint!]
    @eval begin
        function $(func_name)( 
                mop :: AbstractMOP{true}, T :: Type{<:AbstractVecFun},
                func :: Function; kwargs...) 

            objf = _wrap_func( T, func; kwargs... )
            return $(Symbol("_$(func_name)"))(mop, objf)
        end
    end
end

function eval_vec_mop( mop :: AbstractMOP, x_unscaled :: Vec, func_ind :: FunctionIndex )
    objf = _get(mop, func_ind)
    return eval_objf( objf, x_unscaled )
end

function eval_vec_mop_at_scaled_site( mop :: AbstractMOP, x_scaled :: Vec, func_ind :: FunctionIndex )
    x = unscale( x_scaled, mop )
    return eval_vec_mop(mop, x, func_ind )
end

function eval_mop_at_scaled_site( mop :: AbstractMOP, x_scaled :: Vec, 
        func_indices  )
    x = unscale( x_scaled, mop )
    return Base.ImmutableDict( 
        (func_ind => eval_vec_mop_at_scaled_site( mop, x, func_ind ) for func_ind in func_indices)...
    )
end

eval_mop_at_scaled_site(mop :: AbstractMOP, x::Vec) = eval_mop_at_scaled_site(mop, x, get_function_indices(mop))

function eval_vec_mop_at_scaled_site( mop :: AbstractMOP, x_scaled :: Vec, 
        func_indices  )
    tmp = eval_mop_at_scaled_site( mop, x_scaled, func_indices )
    return flatten_vecs(values(tmp))
end

function eval_result_to_vector( mop :: AbstractMOP, eval_result :: AbstractDict )
    return eval_result_to_vector(eval_result, get_function_indices(mop))
end

function eval_result_to_vector( eval_result :: AbstractDict, func_indices  )
    return flatten_vecs( get( eval_result, func_ind, MIN_PRECISION[] ) for func_ind in func_indices)  
end

function eval_result_to_all_vectors( eval_result :: AbstractDict,  mop :: Union{AbstractSurrogateContainer,AbstractMOP} )
    return (
        eval_result_to_vector( eval_result, get_objective_indices(mop) ),
        eval_result_to_vector( eval_result, get_eq_constraint_indices(mop) ),
        eval_result_to_vector( eval_result, get_ineq_constraint_indices(mop) )
    )
end    

# defines:
# eval_mop_objectives_at_scaled_site
# eval_mop_eq_constraints_at_scaled_site
# eval_mop_ineq_constraints_at_scaled_site
# eval_vec_mop_objectives_at_scaled_site
# eval_vec_mop_eq_constraints_at_scaled_site
# eval_vec_mop_ineq_constraints_at_scaled_site

for eval_type in [:objective, :eq_constraint, :ineq_constraint]
    @eval begin
        function $(Symbol("eval_mop_", eval_type, "s_at_scaled_site"))( mop :: AbstractMOP, x_scaled :: Vec )
            return eval_mop_at_scaled_site( mop, x_scaled, $(Symbol("get_$(eval_type)_indices"))(mop) )
        end

        function $(Symbol("eval_vec_mop_", eval_type, "s_at_scaled_site"))( mop :: AbstractMOP, x_scaled :: Vec )
            tmp_res = $(Symbol("eval_mop_", eval_type, "s_at_scaled_site"))( mop, x_scaled )
            return eval_result_to_vector( mop, tmp_res )
        end
    end
end

# custom broadcasting to exploit inner batching in `eval_missing!`
function Broadcast.broadcasted( ::typeof( eval_vec_mop ), mop :: AbstractMOP, X_unscaled :: VecVec, func_ind :: FunctionIndex )
    objf = _get(mop, func_ind)
    return eval_objf.( objf, X_unscaled )
end

function Broadcast.broadcasted( ::typeof(eval_vec_mop_at_scaled_site), mop :: AbstractMOP, X_scaled :: VecVec, func_ind :: FunctionIndex )
    X = unscale.( X_scaled, mop )
    return eval_vec_mop.(mop, X, func_ind )
end

function Broadcast.broadcasted( ::typeof(eval_vec_mop_at_scaled_site), mop :: AbstractMOP, X_scaled :: VecVec, 
        func_indices  )
    X = unscale.( X_scaled, mop )
    partial_vecs = [eval_vec_mop_at_scaled_site.(mop, X, func_ind) for func_ind in func_indices]
    return reduce.(vcat, zip(partial_vecs...))
    
end

# Helper functions …
function num_evals( mop :: AbstractMOP ) :: Vector{Int}
    [ num_evals(objf) for objf ∈ list_of_functions(mop) ]
end

@doc "Set evaluation counter to 0 for each VectorObjectiveFunction in `m.vector_of_objectives`."
function reset_evals!(mop :: AbstractMOP) :: Nothing
    for objf ∈ list_of_functions( mop )
        wrapped_function(objf).counter[] = 0
    end
    return nothing
end
