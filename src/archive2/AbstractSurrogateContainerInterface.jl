# INFO 
# `sw` ~ of type SurrogateWrapper ~ `wrapper`
# `sc` ~ of type SurrogateContainer ~ `container`
# `sw_index` = position of a wrapper in the `list_of_wrappers` of an sc
# `sw_output`, `sw_outputs` ~ position in the return vector of a wrapper 
# … (same as the wrapped model), evaluated in order "objectives" "eq_constraints", "ineq_constraints"
# `sc_output`, `sc_output` ~ position in the return vector of a container

# AbstractSurrogateWrapper
Base.broadcastable( sw :: AbstractSurrogateWrapper ) = Ref(sw)

# constructor
function init_wrapper( :: Type{<:AbstractSurrogateWrapper}, 
    cfg :: SurrogateConfig, mod :: SurrogateModel, meta :: SurrogateMeta,
    objective_indices :: Vector{<:ObjectiveIndex}, 
    eq_constraint_indices :: Vector{<:EqConstraintIndex}, 
    ineq_constraint_indices :: Vector{<:IneqConstraintIndex},
    x_cache :: VecF, vals_cache :: VecF, jacobian_cache :: MatF; kwargs... ) :: AbstractSurrogateWrapper
    return nothing
end

# ... automalically make caches:
function _init_caches( model :: SurrogateModel, x :: VecF )
    model_output = eval_models(model, x)
    no_out = length(model_output)
    x_cache = MVector{n_vars}(x)
    y_cache = MVector{no_out}(model_output)
    jacobian_cache = MMatrix{no_out,n_vars}( get_jacobian(model, x) )
    return x_cache, y_cache, jacobian_cache
end

function init_wrapper( sw_type :: Type{<:AbstractSurrogateWrapper},
    cfg :: SurrogateConfig, mod :: SurrogateModel, meta :: SurrogateMeta,
    objective_indices :: Vector{<:ObjectiveIndex}, 
    eq_constraint_indices :: Vector{<:EqConstraintIndex}, 
    ineq_constraint_indices :: Vector{<:IneqConstraintIndex},
    x :: VecF; kwargs... ) :: AbstractSurrogateWrapper
    x_cache, vals_cache, jacobian_cache = _init_caches( model, x )
    return init_wrapper( sw_type, cfg, mod, meta, objective_indices, 
        eq_constraint_indices, ineq_constraint_indices, x_cache, 
        vals_cache, jacobian_cache; kwargs... )
end

# mandatory methods
get_config( sw :: AbstractSurrogateWrapper ) :: SurrogateConfig = nothing
get_model( sw :: AbstractSurrogateWrapper ) :: SurrogateModel = nothing 
get_meta( sw :: AbstractSurrogateWrapper ) :: SurrogateMeta = nothing

function get_objective_indices( sw :: AbstractSurrogateWrapper ) :: Vector{ObjectiveIndex}
    return nothing
end

function get_eq_constraint_indices( sw :: AbstractSurrogateWrapper ) :: Vector{ConstraintIndex}
    return nothing 
end

function get_ineq_constraint_indices( sw :: AbstractSurrogateWrapper ) :: Vector{ConstraintIndex}
    return nothing 
end

get_x_cache!( sw :: AbstractSurrogateWrapper ) = nothing
set_x_cache!( sw :: AbstractSurrogateWrapper, x̂ :: Vec ) = nothing
get_val_cache( sw :: AbstractSurrogateWrapper ) = nothing 
set_val_cache!( sw :: AbstractSurrogateWarpper, vals :: Vec ) = nothing
get_jacobian_cache( sw :: AbstractSurrogateWrapper ) = nothing
set_jacobian_cache!( sw :: AbstractSurrogateWrapper, J :: Mat ) = nothing

# derived
function get_function_indices( sw :: AbstractSurrogateWrapper )
    return Iterators.flatten(( 
        get_objective_indices(sw),
        get_eq_constraint_indices( sw ),
        get_ineq_constraint_indices( sw ),
    ))
end

function get_function_index_tuple( sw :: AbstractSurrogateWrapper )
    return Tuple( get_function_indices(sw) )
end

function num_outputs( sw :: AbstractSurrogateWrapper )
    return sum( num_outputs(ind) for ind in get_function_indices( sw ) )
end

function wrapper_vals_cached( sw :: AbstractSurrogateWrapper, x̂ :: Vec )
    if get_x_cache(sw) == x̂
        vals = get_val_cache( sw )
        if _is_valid_vector(vals)
            return vals
        end
    end 

    vals = eval_models( get_model(sw), x̂ )
    set_val_cache!( sw, vals )
    set_x_cache!( sw, x̂ )
    return vals 
end

function wrapper_jacobian_cached( sw :: AbstractSurrogateWrapper, x̂ :: Vec )
    if get_x_cache(sw) == x̂
        J = get_jacobian_cache( sw )
        if _is_valid_vector( J )
            return J
        end
    end
    J = get_jacobian( get_model(sw), x̂ )
    set_jacobian_cache!( sw, J )
    set_x_cache!( sw, x̂ )
    return J
end

function wrapper_vals_of_outputs( sw :: AbstractSurrogateWrapper, x̂ :: Vec, 
        sw_outputs :: Vector{Int}, :: Val{:cached} )
    vals = wrapper_vals_cached( sw, x̂ )
    return vals[sw_outputs]
end

function wrapper_vals_of_outputs( sw :: AbstractSurrogateWrapper, x̂ :: Vec,
        sw_outputs :: Vector{Int}, :: Val{:uncached} )
    return flatten_vecs( eval_models( get_model(sw), x̂, l ) for l in sw_outputs )
end

function wrapper_jacobian_of_outputs( sw :: AbstractSurrogateWrapper, x̂ :: Vec, 
        sw_outputs :: Vector{Int},:: Val{:cached} )
    vals = wrapper_jacobian_cached( sw, x̂ )
    return vals[sw_outputs, :]
end

function wrapper_jacobian_of_outputs( sw :: AbstractSurrogateWrapper, x̂ :: Vec,
        sw_outputs :: Vector{Int}, :: Val{:uncached} )
    return mat_from_row_vecs( get_gradient( get_model(sw), x̂, l ) for l in sw_outputs  )
end

# AbstractSurrogateContainer
Base.broadcastable( sc :: AbstractSurrogateContainer ) = Ref(sc)

# constructor 
function init_surrogates( :: Type{<:AbstractSurrogateContainer}, :: AbstractMOP, :: AbstractIterData, 
	:: AbstractDB, :: AbstractConfig )
	return nothing 
end

# update methods
function update_surrogates!( :: AbstractSurrogateContainer, :: AbstractMOP, :: AbstractIterData,
	:: AbstractDB; ensure_fully_linear = false ) 
	return nothing
end

function improve_surrogates!( :: AbstractSurrogateContainer, :: AbstractMOP, :: AbstractIterData,
	:: AbstractDB; ensure_fully_linear = false, kwargs... ) 
	return nothing
end

# mandatory functions
list_of_wrappers( sc :: AbstractSurrogateContainer ) :: AbstractVector{<:AbstractSurrogateWrapper } = nothing
get_objective_indices( sc :: AbstractSurrogateContainer ) :: Vector{ObjectiveIndex} = nothing
get_eq_constraint_indices( sc :: AbstractSurrogateContainer ) :: Vector{ConstraintIndex} = nothing
get_ineq_constraint_indices( sc :: AbstractSurrogateContainer ) :: Vector{ConstraintIndex} = nothing

# derived
function get_function_indices( sc :: AbstractSurrogateContainer )
    return Iterators.flatten( ( 
        get_objective_indices( sc ),
        get_eq_constraint_indices( sc ),
        get_ineq_constraint_indices( sc ),
    ))
end

# TODO get_wrapper can be improved
get_wrapper(sc :: AbstractSurrogateContainer, i :: Int) = list_of_wrappers(sc)[i]

function eval_container_at_scaled_site( sc :: SurrogateContainer, x_scaled :: Vec )
    return Base.ImmutableDict(
    Iterators.flatte( 
        wrapper_vals_of_outputs( )
    )
    )
end

#=
# TODO implement a dict in `sc` to make this more performant
function _sw_index_and_sw_outputs_from_func_index( wrapper_list, func_ind :: FunctionIndex)
    for (sw_ind, sw) in enumerate(wrapper_list)
        sw_output = 1
        for sw_func_ind in enumerate(get_function_indices( sw ))
            next_sw_output = sw_output + num_outputs( sw_func_ind )
            if func_ind == sw_func_ind
                return sw_ind, sw_output : next_sw_output - 1
            end
            sw_output = next_sw_output
        end
    end
    return nothing
end

function sw_index_and_sw_outputs_from_func_index( sc :: AbstractSurrogateContainer, func_ind :: FunctionIndex)
    return _sw_index_and_sw_outputs_from_func_index( list_of_wrappers(sc), func_ind )
end

function sw_and_sw_outputs_from_func_index( sc :: AbstractSurrogateContainer, func_ind :: FunctionIndex )
    sw_ind, sw_outputs = sw_index_and_sw_outputs_from_func_index( sc, func_ind )
    return get_wrapper( sc, sw_ind ), sw_outputs 
end    

function get_wrapper_from_func_index( sc :: AbstractSurrogateContainer, func_ind :: FunctionIndex )
    return sw_and_sw_outputs_from_func_index(sc, func_index )[1]
end

function container_vals_at_func_index( sc :: AbstractSurrogateContainer, x̂ :: Vec, func_ind :: FunctionIndex )
    sw, sw_outputs = sw_and_sw_outputs_from_func_index( sc, func_ind )
    return wrapper_vals_of_outputs( sw, x̂, sw_outputs, Val(:cached))
end

function container_jacobian_at_func_index(  sc :: AbstractSurrogateContainer, x̂ :: Vec, func_ind :: FunctionIndex )
    sw, sw_outputs = sw_and_sw_outputs_from_func_index( sc, func_ind )
    return wrapper_jacobian_of_outputs( sw, x̂, sw_outputs, Val(:cached))
end

# TODO: use a dict in `sc` to make this faster
function _sw_index_and_sw_output_from_sc_index_list( wrapper_list, 
    sc_index_list :: Vector{<:FunctionIndex}, l :: Int )

    sc_output_counter = 1
    for objf_ind in sc_index_list
        objf_ind_out = num_output( objf_ind )
        next_sc_output_counter = sc_output_counter + objf_ind_out
        if sc_output_counter <= l < next_sc_output_counter 
            sw_index, sw_outputs = _sw_index_and_sw_outputs_from_func_index( wrapper_list, objf_ind )
            return sw_index, sw_outputs[ l - sc_output_counter + 1 ]
        end
        sc_output_counter = next_sc_output_counter
    end
    return nothing
end

# Defined here:
#=
fully_linear_objectives eval_objectives_surrogates eval_objectives_surrogates_jacobian 
sw_index_and_sw_output_from_sc_objectives_output sw_and_sw_output_from_sc_objectives_output 
get_objectives_surrogate_optim_handle get_objectives_indices eval_objectives_surrogates_handle 
get_objectives_surrogates_gradient get_objectives_surrogates_gradient

fully_linear_eq_constraints eval_eq_constraints_surrogates eval_eq_constraints_surrogates_jacobian
sw_index_and_sw_output_from_sc_eq_constraints_output sw_and_sw_output_from_sc_eq_constraints_output 
get_eq_constraints_surrogate_optim_handle get_eq_constraints_indices eval_eq_constraints_surrogates_handle 
get_eq_constraints_surrogates_gradient get_eq_constraints_surrogates_gradient

fully_linear_ineq_constraints eval_ineq_constraints_surrogates eval_ineq_constraints_surrogates_jacobian 
sw_index_and_sw_output_from_sc_ineq_constraints_output sw_and_sw_output_from_sc_ineq_constraints_output 
get_ineq_constraints_surrogate_optim_handle get_ineq_constraints_indices eval_ineq_constraints_surrogates_handle 
get_ineq_constraints_surrogates_gradient get_ineq_constraints_surrogates_gradient
=#
for mod_type in [:objectives, :eq_constraints, :ineq_constraints]
    fully_linear_func_name = Symbol("fully_linear_", mod_type )
    
    eval_func_name = Symbol("eval_", mod_type, "_surrogates")
    eval_jac_name = Symbol("eval_", mod_type, "_surrogates_jacobian")
    
    index_func_name = Symbol("sw_index_and_sw_output_from_sc_", mod_type, "_output")
    wrapper_func_name = Symbol("sw_and_sw_output_from_sc_", mod_type, "_output")

    optim_handle_func_name = Symbol("get_", mod_type, "_surrogate_optim_handle")
    get_name = Symbol("get_", mod_type, "_indices")

    eval_handle_func_name = Symbol("eval_", mod_type, "_surrogates_handle")

    gradient_func_name = Symbol("get_", mod_type, "_surrogates_gradient")
    gradient_handle_func_name = Symbol("get_", mod_type, "_surrogates_gradient")

    @eval begin
        function $(fully_linear_func_name)( sc :: AbstractSurrogateContainer )
            wrappers = unique( get_wrapper_from_func_index( sc, ind) for ind in $(get_name)(sc) )
            return all( fully_linear(get_model(sw)) for sw in wrappers )
        end
                
        function container_vals( sc :: AbstractSurrogateContainer, x̂ :: Vec, ::Val{:($mod_type)})
            return flatten_vecs( container_vals_at_func_index( sc, x̂, ind ) for ind in $(get_name)( sc ) ) 
        end    

        function container_jacobian( sc :: AbstractSurrogateContainer, x̂ :: Vec, ::Val{:($mod_type)})
            return vcat( (container_jacobian_at_func_index( sc, x̂, ind ) for ind in $(get_name)( sc ) )... )
        end

        "Evaluate all surrogate models (of the $(mod_type)) stored in `sc` at scaled site `x̂`."
        function $(eval_func_name)( sc :: AbstractSurrogateContainer, x̂ :: Vec )
            return container_vals( sc, x̂, Val(:($mod_type)))
        end

        "Evaluate Jacobian of surrogate models (of the objectives) stored in `sc` at scaled site `x̂`."
        function $(eval_jac_name)( sc :: AbstractSurrogateContainer, x̂ :: Vec )
            return container_jacobian( sc, x̂, Val(:($mod_type)))
        end

        function $(index_func_name)( sc :: AbstractSurrogateContainer, l :: Int )
            return _sw_index_and_sw_output_from_sc_index_list( list_of_wrappers(sc), $(get_name)(sc), l)
        end

        function $(wrapper_func_name)( sc :: AbstractSurrogateContainer, l :: Int )
            sw_ind, sw_out = $(index_func_name)( sc, l)
            return get_wrapper( sc, sw_ind ), sw_out 
        end

        """
            $(optim_handle_func_name)(sc, l)
        Return a function handle to be used with `NLopt` for $(mod_type) output `l` of `sc`.
        """
        function $(optim_handle_func_name)( sc :: AbstractSurrogateContainer, l :: Int )
            sw, sw_out = $(wrapper_func_name)( sc, l )
            return _get_optim_handle( get_model(sw), sw_out )
        end

        """
            $(eval_func_name)( sc, x̂, l )

        Return model value for $(mod_type) output `l` of `sc` at `x̂`.
        """
        function $(eval_func_name)( sc :: AbstractSurrogateContainer, x̂ :: Vec, l :: Int )
            sw, sw_out = $(wrapper_func_name)( sc, l )
            return eval_models( get_model(sw), x̂, sw_out )
        end

        function $(eval_handle_func_name)( sc :: AbstractSurrogateContainer, l :: Int )
            return x -> $(eval_func_name)( sc, x, l )
        end

        """
            $(gradient_func_name)( sc, x̂, l )

        Return a gradient for $(mod_type) output `l` of `sc` at `x̂`.
        """
        function $(gradient_func_name)( sc :: AbstractSurrogateContainer, x̂ :: Vec, l :: Int )
            sw, sw_out = sw_and_sw_output_from_sc_objective_output( sc, l )
            return get_gradient( get_model(sw), x̂, sw_out )
        end

        function $(gradient_handle_func_name)( sc :: AbstractSurrogateContainer, l :: Int)
            return x -> $(gradient_func_name)( sc, x, l )
        end

    end
end
=#