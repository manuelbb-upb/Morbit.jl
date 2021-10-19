# INFO 
# `sw` ~ of type SurrogateWrapper ~ `wrapper`
# `sc` ~ of type SurrogateContainer ~ `container`
# `sw_index` = position of a wrapper in the `list_of_wrappers` of an sc
# `sw_output`, `sw_outputs` ~ position in the return vector of a wrapper 
# … (same as the wrapped model), evaluated in order "objectives" "eq_constraints", "ineq_constraints"
# `sc_output`, `sc_output` ~ position in the return vector of a container

################################################################################

struct ModelGrouping{T}
    indices :: Vector{FunctionIndex}
    cfg :: T
end


# AbstractSurrogateWrapper
Base.broadcastable( sw :: AbstractSurrogateWrapper ) = Ref(sw)

# constructor
function init_wrapper( :: Type{<:AbstractSurrogateWrapper}, 
    cfg :: SurrogateConfig, mod :: SurrogateModel, meta :: SurrogateMeta,
    objective_indices, eq_constraint_indices, 
    ineq_constraint_indices; kwargs... ) :: AbstractSurrogateWrapper
    return nothing
end

# mandatory methods
get_config( sw :: AbstractSurrogateWrapper ) :: SurrogateConfig = nothing
get_model( sw :: AbstractSurrogateWrapper ) :: SurrogateModel = nothing 
get_meta( sw :: AbstractSurrogateWrapper ) :: SurrogateMeta = nothing

function get_objective_indices( sw :: AbstractSurrogateWrapper ) :: Vector{ObjectiveIndex}
    return nothing
end

function get_nl_eq_constraint_indices( sw :: AbstractSurrogateWrapper ) :: Vector{ConstraintIndex}
    return nothing 
end

function get_nl_ineq_constraint_indices( sw :: AbstractSurrogateWrapper ) :: Vector{ConstraintIndex}
    return nothing 
end

# derived
function get_function_indices( sw :: AbstractSurrogateWrapper )
    return Iterators.flatten(( 
        get_objective_indices(sw),
        get_nl_eq_constraint_indices( sw ),
        get_nl_ineq_constraint_indices( sw ),
    ))
end

# TODO use Dict here?
@memoize ThreadSafeDict function sw_outputs_from_func_index( sw :: AbstractSurrogateWrapper, func_index :: FunctionIndex )
    offset = 1
    for ind in get_function_indices(sw)
        next_offset = offset + num_outputs( ind )
        if ind == func_index
            return offset : next_offset - 1
        end
        offset = next_offset
    end
    return -1
end

@memoize ThreadSafeDict function _sortperm( sc :: AbstractSurrogateContainer )
    @show sc_indices = collect( get_function_indices( sc ) )
    @show sw_indices = collect( Iterators.flatten( 
        get_function_indices(sw) for sw in list_of_wrappers(sc)
    ) )
    return sortperm( sw_indices; order = Base.Order.Perm( Base.Order.Forward, sc_indices ) )
end

function get_function_index_tuple( sw :: AbstractSurrogateWrapper )
    return Tuple( get_function_indices(sw) )
end

function num_outputs( sw :: AbstractSurrogateWrapper )
    return sum( num_outputs(ind) for ind in get_function_indices( sw ) )
end

function eval_wrapper_at_scaled_site( sw :: AbstractSurrogateWrapper, scal :: AbstractVarScaler, x_scaled )
    return eval_models( get_model(sw), scal, x_scaled )
end

function eval_wrapper_jacobian_at_scaled_site( sw :: AbstractSurrogateWrapper, 
        scal :: AbstractVarScaler, x_scaled :: Vec 
    )
    return get_jacobian( get_model(sw), scal, x_scaled )
end

function eval_output_of_wrapper_at_scaled_site( sw :: AbstractSurrogateWrapper,
        scal :: AbstractVarScaler, x_scaled :: Vec, sw_output :: Int 
    )
    return eval_models( get_model(sw), scal, x_scaled, sw_output )
end

function eval_gradient_of_wrapper_output_at_scaled_site( sw :: AbstractSurrogateWrapper,
        scal :: AbstractVarScaler, x_scaled :: Vec, sw_output :: Int 
    )
    return get_gradient( get_model(sw), scal, x_scaled, sw_output )
end

# AbstractSurrogateContainer
Base.broadcastable( sc :: AbstractSurrogateContainer ) = Ref(sc)

# constructor 
function init_surrogates( :: Type{<:AbstractSurrogateContainer}, :: AbstractMOP, :: AbstractVarScaler, :: AbstractIterData, 
	:: AbstractConfig, groupings :: Vector{<:ModelGrouping}, :: AbstractSuperDB )
	return nothing 
end

# update methods
function update_surrogates!( :: AbstractSurrogateContainer, :: AbstractMOP, :: AbstractVarScaler, :: AbstractIterData,
	:: AbstractSuperDB, :: AbstractConfig; ensure_fully_linear = false, kwargs... ) 
	return nothing
end

function improve_surrogates!( :: AbstractSurrogateContainer, :: AbstractMOP, :: AbstractVarScaler, :: AbstractIterData,
	:: AbstractSuperDB, :: AbstractConfig; ensure_fully_linear = false, kwargs... ) 
	return nothing
end

# mandatory functions
list_of_wrappers( sc :: AbstractSurrogateContainer ) :: AbstractVector{<:AbstractSurrogateWrapper } = nothing
get_objective_indices( sc :: AbstractSurrogateContainer ) :: Vector{ObjectiveIndex} = nothing
get_nl_eq_constraint_indices( sc :: AbstractSurrogateContainer ) :: Vector{ConstraintIndex} = nothing
get_nl_ineq_constraint_indices( sc :: AbstractSurrogateContainer ) :: Vector{ConstraintIndex} = nothing

replace_wrapper!(sc :: AbstractSurrogateContainer, i :: Int, sw :: AbstractSurrogateWrapper) = nothing

# derived
function fully_linear(sc :: AbstractSurrogateContainer)
    return prod(fully_linear( get_model(sw)) for sw in list_of_wrappers(sc))
end

function get_function_indices( sc :: AbstractSurrogateContainer )
    return Iterators.flatten( ( 
        get_objective_indices( sc ),
        get_nl_eq_constraint_indices( sc ),
        get_nl_ineq_constraint_indices( sc ),
    ))
end

# TODO get_wrapper can be improved
get_wrapper(sc :: AbstractSurrogateContainer, i :: Int) = list_of_wrappers(sc)[i]

function eval_container_at_scaled_site( sc :: AbstractSurrogateContainer, scal :: AbstractVarScaler, x_scaled :: Vec )
    return Base.ImmutableDict( Iterators.flatten( 
            let sw_res = eval_wrapper_at_scaled_site(sw, scal, x_scaled); 
                ( f_ind => sw_res[ sw_outputs_from_func_index(sw, f_ind)] for 
                   f_ind in get_function_indices(sw) )
            end 
            for sw in list_of_wrappers(sc)
        )...
    )
end

function eval_vec_container_at_scaled_site(sc :: AbstractSurrogateContainer, scal :: AbstractVarScaler, x_scaled :: Vec )
    tmp = eval_container_at_scaled_site( sc, scal, x_scaled )
    return eval_result_to_all_vectors( tmp, sc )
    # TODO benchmark against using `_sortperm`
end

#=
function eval_container_jacobian_at_scaled_site( sc :: AbstractSurrogateContainer, scal :: AbstractVarScaler, x_scaled :: Vec )
    row_perm = _sortperm(sc)
    return vcat(
        ( eval_wrapper_jacobian_at_scaled_site( sw, scal, x_scaled ) for sw in list_of_wrappers(sc)
    )... )[row_perm, :]
end
=#

# more granular evaluation:

function _sw_index_and_sw_outputs_from_func_index( wrapper_list, func_ind :: FunctionIndex)
    for (sw_ind, sw) in enumerate(wrapper_list)
        sw_output = 1
        for sw_func_ind in get_function_indices( sw )
            next_sw_output = sw_output + num_outputs( sw_func_ind )
            if func_ind == sw_func_ind
                return sw_ind, sw_output : next_sw_output - 1
            end
            sw_output = next_sw_output
        end
    end
    return nothing
end

# TODO use dict in `sc`
@memoize ThreadSafeDict function sw_index_and_sw_outputs_from_func_index( sc :: AbstractSurrogateContainer, func_ind :: FunctionIndex)
    return _sw_index_and_sw_outputs_from_func_index( list_of_wrappers(sc), func_ind )
end

function sw_and_sw_outputs_from_func_index( sc :: AbstractSurrogateContainer, func_ind :: FunctionIndex )
    sw_ind, sw_outputs = sw_index_and_sw_outputs_from_func_index( sc, func_ind )
    return get_wrapper( sc, sw_ind ), sw_outputs 
end    

function get_wrapper_from_func_index( sc :: AbstractSurrogateContainer, func_ind :: FunctionIndex )
    return sw_and_sw_outputs_from_func_index(sc, func_ind )[1]
end

function eval_vec_container_at_func_index_at_scaled_site( sc :: AbstractSurrogateContainer, 
        scal :: AbstractVarScaler, x_scaled :: Vec, func_ind :: FunctionIndex )
    sw, sw_outputs = sw_and_sw_outputs_from_func_index( sc, func_ind )
    return [ eval_output_of_wrapper_at_scaled_site( sw, scal, x_scaled, l ) for l = sw_outputs ]
end

function eval_container_jacobian_at_func_index_at_scaled_site(  sc :: AbstractSurrogateContainer,
        scal :: AbstractVarScaler, x_scaled :: Vec, func_ind :: FunctionIndex 
    )
    sw, sw_outputs = sw_and_sw_outputs_from_func_index( sc, func_ind )
    return mat_from_row_vecs( 
        eval_gradient_of_wrapper_output_at_scaled_site( sw, scal, x_scaled, l ) for l = sw_outputs 
    )
end

#=
# this is used in sw_index_and_sw_output_from_XXXX_output
# l ∈ [1, …, length(sc_index_list)]
function _sw_index_and_sw_output_from_func_indices_and_position( wrapper_list, 
    func_indices, l :: Int )

    sc_output_counter = 1
    for objf_ind in func_indices
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
=#

for mod_type in [:objectives, :nl_eq_constraints, :nl_ineq_constraints]
    fully_linear_func_name = Symbol("fully_linear_", mod_type )
    
    eval_container_XXX_at_scaled_site = Symbol("eval_container_", mod_type, "_at_scaled_site")
    eval_container_XXX_jacobian_at_scaled_site = Symbol("eval_container_", mod_type, "_jacobian_at_scaled_site")
    
    sw_index_and_sw_output_from_sc_XXX_output = Symbol("sw_index_and_sw_output_from_sc_", mod_type, "_output")
    sw_and_sw_output_from_sc_XXX_output = Symbol("sw_and_sw_output_from_sc_", mod_type, "_output")

    get_indices_func = Symbol("get_", String(mod_type)[1:end-1], "_indices")
    #=
    optim_handle_func_name = Symbol("get_", mod_type, "_surrogate_optim_handle")
  
    eval_handle_func_name = Symbol("eval_", mod_type, "_surrogates_handle")

    gradient_func_name = Symbol("get_", mod_type, "_surrogates_gradient")
    gradient_handle_func_name = Symbol("get_", mod_type, "_surrogates_gradient")
    =#
    @eval begin
        function $(fully_linear_func_name)( sc :: AbstractSurrogateContainer )
            wrappers = unique( get_wrapper_from_func_index( sc, ind) for ind in $(get_indices_func)(sc) )
            return all( fully_linear(get_model(sw)) for sw in wrappers )
        end

        "Evaluate all surrogate models stored in `sc` at scaled site `x_scaled`."
        function $(eval_container_XXX_at_scaled_site)( sc :: AbstractSurrogateContainer, 
                scal :: AbstractVarScaler, x_scaled :: Vec )
            indices = $(get_indices_func)(sc)
            isempty(indices) && return Base.promote_type( eltype(x_scaled), MIN_PRECISION)[]
            return flatten_vecs( 
                eval_vec_container_at_func_index_at_scaled_site(sc, scal, x_scaled, ind ) for ind in indices
            )
        end

        "Evaluate Jacobian of surrogate models stored in `sc` at scaled site `x_scaled`."
        function $(eval_container_XXX_jacobian_at_scaled_site)( sc :: AbstractSurrogateContainer, 
                scal :: AbstractVarScaler, x_scaled :: Vec )
            indices = $(get_indices_func)(sc)
            isempty(indices) && return Matrix{Base.promote_type( eltype(x_scaled), MIN_PRECISION)}(undef, 0, length(x_scaled))
            return vcat( 
                (eval_container_jacobian_at_func_index_at_scaled_site(sc, scal, x_scaled, ind) for ind in indices)...
            )
        end

        #=
        function $(sw_index_and_sw_output_from_sc_XXX_output)( sc :: AbstractSurrogateContainer, l :: Int )
            return _sw_index_and_sw_output_from_sc_index_list( list_of_wrappers(sc), $(get_indices_func)(sc), l)
        end

        function $(sw_and_sw_output_from_sc_XXX_output)( sc :: AbstractSurrogateContainer, l :: Int )
            sw_ind, sw_out = $(sw_index_and_sw_output_from_sc_XXX_output)( sc, l)
            return get_wrapper( sc, sw_ind ), sw_out 
        end

        """
            $(optim_handle_func_name)(sc, l)
        Return a function handle to be used with `NLopt` for $(mod_type) output `l` of `sc`.
        """
        function $(optim_handle_func_name)( sc :: AbstractSurrogateContainer, l :: Int )
            sw, sw_out = $(sw_and_sw_output_from_sc_XXX_output)( sc, l )
            return _get_optim_handle( get_model(sw), sw_out )
        end

        """
            $(eval_container_XXX_at_scaled_site)( sc, x̂, l )

        Return model value for $(mod_type) output `l` of `sc` at `x̂`.
        """
        function $(eval_container_XXX_at_scaled_site)( sc :: AbstractSurrogateContainer, x̂ :: Vec, l :: Int )
            sw, sw_out = $(sw_and_sw_output_from_sc_XXX_output)( sc, l )
            return eval_models( get_model(sw), x̂, sw_out )
        end

        function $(eval_handle_func_name)( sc :: AbstractSurrogateContainer, l :: Int )
            return x -> $(eval_container_XXX_at_scaled_site)( sc, x, l )
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
        =#
    end
end