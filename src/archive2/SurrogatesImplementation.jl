
include("RbfModel.jl")
#include("ExactModel.jl")
#include("TaylorModel.jl")
#include("LagrangeModel.jl")

struct SurrogateWrapper{
        C <: SurrogateConfig,
        M <: SurrogateModel,
        I <: SurrogateMeta, 
        XT <: VecF, YT <: VecF, JT <: MatF} <: AbstractSurrogateWrapper
    cfg :: C
    model :: M
    meta :: I
    objective_indices :: Tuple{Vararg{ObjectiveIndex}}
    eq_constraint_indices :: Tuple{Vararg{EqConstraintIndex}}
    ineq_constraint_indices :: Tuple{Vararg{IneqConstraintIndex}}

    x_cache :: XT 
    val_cache :: YT 
    jacobian_cache :: JT

    num_outputs :: Int

end

function init_wrapper( :: Type{<:SurrogateWrapper}, 
    cfg :: SurrogateConfig, model :: SurrogateModel, meta :: SurrogateMeta,
    objective_indices = ObjectiveIndex[], 
    eq_constraint_indices = EqConstraintIndex[], 
    ineq_constraint_indices = IneqConstraintIndex[],
    x_cache, vals_cache, jacobian_cache; no_out = -1, kwargs... )

    if no_out < 0 
        no_out = sum( num_outputs(ind) for ind in [objective_indices; eq_constraint_indices; ineq_constraint_indices] )
    end

    return SurrogateWrapper( cfg, model, meta, 
        Tuple(objective_indices), Tuple(eq_constraint_indices), 
        Tuple(ineq_constraint_indices),
        x_cache, vals_cache, jacobian_cache, no_out )
end

get_config( sw :: SurrogateWrapper ) = sw.cfg
get_model( sw :: SurrogateWrapper ) = sw.model
get_meta( sw :: SurrogateWrapper ) = sw.meta
get_objective_indices( sw :: SurrogateWrapper ) = sw.objective_indices 
get_eq_constraints_indices( sw :: SurrogateWrapper ) = sw.eq_constraint_indices
get_ineq_constraints_indices( sw :: SurrogateWrapper ) = sw.ineq_constraint_indices

num_outputs(sw :: SurrogateWrapper) = sw.num_outputs

get_x_cache( sw :: SurrogateWrapper ) = sw.x_cache
get_val_cache(sw :: SurrogateWrapper ) = sw.y_cache
get_jacobian_cache( sw :: SurrogateWrapper ) = sw.jacobian_cache

function set_x_cache!( sw :: SurrogateWrapper, x̂ :: Vec )
    @assert length(sw.x_cache) == length(x̂)
    sw.x_cache[:] .= x̂[:]
    return nothing
end

function set_val_cache!( sw :: SurrogateWrapper, vals :: Vec )
    @assert length(sw.val_cache) == length(vals)
    sw.val_cache[:] .= vals[:]
    return nothing
end

function set_jacobian_cache!( sw :: SurrogateWrapper, J :: Mat )
    @assert size(sw.jacobian_cache) == size(J)
    sw.jacobian_cache[:] .= J[:]
    return nothing
end

struct SurrogateContainer{
        T <: Tuple{Vararg{<:AbstractSurrogateWrapper}} } <: AbstractSurrogateContainer
    
    wrappers :: T
    #=
    func_index_to_sw_index_and_sw_outputs :: Base.ImmutableDict{ 
        FunctionIndex, Tuple{Int, Tuple{Vararg{Int}}} }

    sc_objective_output_to_sw_index_and_sw_output :: Base.ImmutableDict{
        Int, Tuple{<:FunctionIndex, Int}
    }

    sc_eq_constraint_output_to_sw_index_and_sw_output :: Base.ImmutableDict{
        Int, Tuple{<:FunctionIndex, Int}
    }

    sc_ineq_constraint_output_to_sw_index_and_sw_output :: Base.ImmutableDict{
        Int, Tuple{<:FunctionIndex, Int}
    }=#

    # as defined in the MOP - for evaluation
    objective_indices :: Tuple{ObjectiveIndex}
    eq_constraint_indices :: Tuple{EqConstraintIndex}
    ineq_constraint_indices :: Tuple{IneqConstraintIndex}
end

# A handy constructor that takes care of sorting and 
# setting up the dicts
function SurrogateContainer( wrappers, objective_indices, eq_constraint_indices,
    ineq_constraint_indices )

    #=
    all_indices = [ objective_indices; eq_constraint_indices; ineq_constraint_indices ]

    func_ind_sw_index_sw_outputs_dict = Dict( 
        func_ind => _sw_index_and_sw_outputs_from_func_index( wrappers, func_ind ) for func_ind in all_indices
    )

    num_objective_outputs = sum( num_outputs(func_ind) for func_ind in objective_indices)
    sc_objective_to_sw_index_and_sw_output = Dict( 
        l => _sw_index_and_sw_output_from_sc_index_list( wrappers, objective_indices, l) for l = 1 : num_objective_outputs
    )
    
    num_eq_constraint_outputs = sum( num_outputs(func_ind) for func_ind in eq_constraint_indices)
    sc_eq_constraint_to_sw_index_and_sw_output = Dict( 
        l => _sw_index_and_sw_output_from_sc_index_list( wrappers, eq_constraint_indices, l) for l = 1 : num_eq_constraint_outputs
    )
    
    num_ineq_constraint_outputs = sum( num_outputs(func_ind) for func_ind in ineq_constraint_indices)
    sc_ineq_constraint_to_sw_index_and_sw_output = Dict( 
        l => _sw_index_and_sw_output_from_sc_index_list( wrappers, ineq_constraint_indices, l) for l = 1 : num_ineq_constraint_outputs
    )
    =#
    
    return SurrogateContainer( Tuple(wrappers), 
        #func_ind_sw_index_sw_outputs_dict, 
        #sc_objective_to_sw_index_and_sw_output,
        #sc_eq_constraint_to_sw_index_and_sw_output,
        #sc_ineq_constraint_to_sw_index_and_sw_output,
        Tuple(objective_indices), Tuple(eq_constraint_indices),
        Tuple(ineq_constraint_indices) 
    )    
end

list_of_wrappers( sc :: SurrogateContainer ) = sc.wrappers
get_wrapper( sc :: SurrogateContainer, i :: Int ) = sc.wrappers[i]

get_objective_indices( sc :: SurrogateContainer ) = sc.objective_indices
get_eq_constraint_indices( sc :: SurrogateContainer ) = sc.eq_constraint_indices
get_ineq_constraint_indices( sc :: SurrogateContainer ) = sc.ineq_constraint_indices

# exploit the dicts:
#=
function sw_index_and_sw_outputs_from_func_index( sc :: SurrogateContainer, func_ind :: FunctionIndex)
    return sc.func_index_to_sw_index_and_sw_outputs[func_ind]
end

function sw_index_and_sw_output_from_sc_objectives_output( sc :: SurrogateContainer, l :: Int)
    return sc.sc_objective_output_to_sw_index_and_sw_output[l]
end

function sw_index_and_sw_output_from_sc_eq_constraints_output( sc :: SurrogateContainer, l :: Int)
    return sc.sc_objective_output_to_sw_index_and_sw_output[l]
end

function sw_index_and_sw_output_from_sc_ineq_constraints_output( sc :: SurrogateContainer, l :: Int)
    return sc.sc_objective_output_to_sw_index_and_sw_output[l]
end 
=#
@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( ::Type{<:SurrogateContainer}, mop :: AbstractMOP, 
    id :: AbstractIterData{XT,YT,ET,IT,DT}, ac :: AbstractConfig, 
    groupings :: Vector{ModelGrouping}, sdb :: AbstractSuperDB ) where{XT,YT,ET,IT,DT}
    
    @logmsg loglevel2 "Initializing surrogate models."
    
    # init fields for SurrogateContainer
    meta_array = SurrogateMeta[]

    for group in groupings
        sub_db = get_sub_db( sdb, group.indices )
        meta = prepare_init_model( group.cfg, group.indices, mop, id, sub_db, ac; meta_array )
        push!(meta_array, meta)
    end
    
    @logmsg loglevel2 "Evaluation of unevaluated results."
    eval_missing!(sdb, mop)
    
    x = get_x(id)
    wrapper_array = SurrogateWrapper[]
    for (i,group) in enumerate(groupings)
        _meta = meta_array[i]
        sub_db = sub_dbs[i]
        model, meta = init_model( _meta, group.cfg, group.indices, mop, id, sub_db, ac )
        sw = init_wrapper( SurrogateWrapper, group.cfg, model, meta, _split( group.indices )..., x )
        push!(wrapper_array, sw)
    end

    return SurrogateContainer( 
        wrapper_array,
        get_objective_indices(mop),
        get_eq_constraint_indices(mop),
        get_ineq_constraint_indices(mop)
    )
end

function update_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    @logmsg loglevel2 "Updating surrogate models."

    meta_array = SurrogateMeta[]

    for sw in list_of_wrappers( sw )
        func_indices = get_function_index_tuple(sw)
        push!(meta_array, 
            prepare_update_model(
                get_model(sw), get_meta(sw), get_config(sw), func_indices,
                mop, id, get_sub_db(sdb, func_indices), ac; ensure_fully_linear, meta_array ) 
        )
    end

    eval_missing!(db, mop)

    for (si,sw) ∈ enumerate(sc.surrogates)
        _meta = meta_array[si]
        func_indices = get_function_index_tuple(sw)
        new_model, new_meta = update_model( 
            get_model(sw), _meta, get_config(sw), func_indices, 
            mop, id, get_sub_db(sdb, func_indices), ac; ensure_fully_linear )
        
        new_sw = init_wrapper(
            SurrogateWrapper, 
            get_cfg( sw ), new_model, new_meta,
            get_objective_indices( sw ), get_eq_constraint_indices(sw),
            get_ineq_constraint_indices(sw), get_x(id)
        )
        sc.surrogates[si] = new_sw
    end
    return nothing
end

function improve_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; 
    ensure_fully_linear :: Bool = false, kwargs... ) :: Nothing 
    @logmsg loglevel2 "Improving surrogate models."

    meta_array = SurrogateMeta[]

    for sw in list_of_wrappers( sw )
        func_indices = get_function_index_tuple(sw)
        push!(meta_array, 
            prepare_improve_model(
                get_model(sw), get_meta(sw), get_config(sw), func_indices,
                mop, id, get_sub_db(sdb, func_indices), ac; ensure_fully_linear, meta_array ) 
        )
    end

    eval_missing!(db, mop)

    for (si,sw) ∈ enumerate(sc.surrogates)
        _meta = meta_array[si]
        func_indices = get_function_index_tuple(sw)
        new_model, new_meta = improve_model( 
            get_model(sw), _meta, get_config(sw), func_indices, 
            mop, id, get_sub_db(sdb, func_indices), ac; ensure_fully_linear )
        
        new_sw = init_wrapper(
            SurrogateWrapper, 
            get_cfg( sw ), new_model, new_meta,
            get_objective_indices( sw ), get_eq_constraint_indices(sw),
            get_ineq_constraint_indices(sw), get_x(id)
        )
        sc.surrogates[si] = new_sw
    end
    return nothing
end

