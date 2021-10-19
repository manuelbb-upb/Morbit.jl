struct SurrogateWrapper{
        C <: SurrogateConfig,
        M <: SurrogateModel,
        I <: SurrogateMeta, 
        OIndType <: Tuple{Vararg{ObjectiveIndex}}, 
        EIndType <: Tuple{Vararg{ConstraintIndex}}, 
        IIndType <: Tuple{Vararg{ConstraintIndex}} 
    } <: AbstractSurrogateWrapper
    cfg :: C
    model :: M
    meta :: I
    objective_indices :: OIndType
    nl_eq_constraint_indices :: EIndType
    nl_ineq_constraint_indices :: IIndType

    num_outputs :: Int

    function SurrogateWrapper( cfg :: C, model :: M, meta :: I, objective_indices, 
        eq_constraint_indices, ineq_constraint_indices, num_outputs :: Int) where {C,M,I}
        t_objective_indices = Tuple( objective_indices )
        t_nl_eq_constraint_indices = Tuple( eq_constraint_indices )
        t_nl_ineq_constraint_indices = Tuple( ineq_constraint_indices )
        return new{C,M,I,typeof(t_objective_indices), typeof(t_nl_eq_constraint_indices), typeof(t_nl_ineq_constraint_indices)}(cfg,model,meta,t_objective_indices,t_nl_eq_constraint_indices,t_nl_ineq_constraint_indices,num_outputs)
    end
end

function init_wrapper( :: Type{<:SurrogateWrapper}, 
    cfg :: SurrogateConfig, model :: SurrogateModel, meta :: SurrogateMeta,
    objective_indices = ObjectiveIndex[],
    eq_constraint_indices = ConstraintIndex[], 
    ineq_constraint_indices = ConstraintIndex[];
    no_out = -1, kwargs... )

    if no_out < 0 
        no_out = sum( num_outputs(ind) for ind in [objective_indices; eq_constraint_indices; ineq_constraint_indices] )
    end

    return SurrogateWrapper( cfg, model, meta, 
        objective_indices, eq_constraint_indices, ineq_constraint_indices,
        no_out 
    )
end

get_config( sw :: SurrogateWrapper ) = sw.cfg
get_model( sw :: SurrogateWrapper ) = sw.model
get_meta( sw :: SurrogateWrapper ) = sw.meta
get_objective_indices( sw :: SurrogateWrapper ) = sw.objective_indices 
get_nl_eq_constraint_indices( sw :: SurrogateWrapper ) = sw.nl_eq_constraint_indices
get_nl_ineq_constraint_indices( sw :: SurrogateWrapper ) = sw.nl_ineq_constraint_indices

num_outputs(sw :: SurrogateWrapper) = sw.num_outputs

struct SurrogateContainer{
        T <: Vector{<:AbstractSurrogateWrapper},
        OIndType <: Tuple{Vararg{ObjectiveIndex}}, 
        EIndType <: Tuple{Vararg{ConstraintIndex}}, 
        IIndType <: Tuple{Vararg{ConstraintIndex}}
    } <: AbstractSurrogateContainer
    
    wrappers :: T
    
    # as defined in the MOP - for evaluation
    objective_indices :: OIndType
    nl_eq_constraint_indices :: EIndType
    nl_ineq_constraint_indices :: IIndType

    function SurrogateContainer(wrappers :: T, objective_indices, eq_constraint_indices, ineq_constraint_indices) where T
        t_objective_indices = Tuple( objective_indices )
        t_nl_eq_constraint_indices = Tuple( eq_constraint_indices )
        t_nl_ineq_constraint_indices = Tuple( ineq_constraint_indices )
        return new{T,typeof(t_objective_indices), typeof(t_nl_eq_constraint_indices), typeof(t_nl_ineq_constraint_indices)}(wrappers, t_objective_indices,t_nl_eq_constraint_indices,t_nl_ineq_constraint_indices)
    end
end

list_of_wrappers( sc :: SurrogateContainer ) = sc.wrappers
get_wrapper( sc :: SurrogateContainer, i :: Int ) = sc.wrappers[i]
function replace_wrapper!(sc :: SurrogateContainer, i :: Int, sw :: AbstractSurrogateWrapper) 
    sc.wrappers[i] = sw
end
get_objective_indices( sc :: SurrogateContainer ) = sc.objective_indices
get_nl_eq_constraint_indices( sc :: SurrogateContainer ) = sc.nl_eq_constraint_indices
get_nl_ineq_constraint_indices( sc :: SurrogateContainer ) = sc.nl_ineq_constraint_indices

@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( ::Type{<:SurrogateContainer}, mop :: AbstractMOP,
    scal :: AbstractVarScaler, 
    id :: AbstractIterData, ac :: AbstractConfig, 
    groupings :: Vector{ModelGrouping}, sdb :: AbstractSuperDB )
    
    @logmsg loglevel2 "Initializing surrogate models."
    
    # init fields for SurrogateContainer
    meta_array = SurrogateMeta[]

    for group in groupings
        meta = prepare_init_model( group.cfg, group.indices, mop, scal, id, sdb, ac; meta_array )
        push!(meta_array, meta)
    end
    
    @logmsg loglevel2 "Evaluation of unevaluated results."
    eval_missing!(sdb, mop, scal)
    
    wrapper_array = SurrogateWrapper[]
    for (i,group) in enumerate(groupings)
        _meta = meta_array[i]
        model, meta = init_model( _meta, group.cfg, group.indices, mop, scal, id, sdb, ac )
        sw = init_wrapper( SurrogateWrapper, group.cfg, model, meta, _split( group.indices )... )
        push!(wrapper_array, sw)
    end

    return SurrogateContainer( 
        wrapper_array,
        get_objective_indices(mop),
        get_nl_eq_constraint_indices(mop),
        get_nl_ineq_constraint_indices(mop)
    )
end

function update_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP,
    scal :: AbstractVarScaler, 
    id :: AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; 
    ensure_fully_linear :: Bool = false, kwargs... ) :: Nothing 
    
    @logmsg loglevel2 "Updating surrogate models."

    meta_array = SurrogateMeta[]

    for sw in list_of_wrappers( sc )
        func_indices = get_function_index_tuple(sw)
        meta = prepare_update_model(
            get_model(sw), get_meta(sw), get_config(sw), func_indices,
            mop, scal, id, sdb, ac; ensure_fully_linear, meta_array ) 
        push!(meta_array, meta)
    end

    eval_missing!(sdb, mop, scal)

    for (si,sw) ∈ enumerate( list_of_wrappers(sc ) )
        _meta = meta_array[si]
        func_indices = get_function_index_tuple(sw)
        new_model, new_meta = update_model( 
            get_model(sw), _meta, get_config(sw), func_indices, 
            mop, scal, id, sdb, ac; ensure_fully_linear )
        
        new_sw = init_wrapper(
            SurrogateWrapper, 
            get_config( sw ), new_model, new_meta,
            get_objective_indices( sw ), get_nl_eq_constraint_indices(sw),
            get_nl_ineq_constraint_indices(sw)
        )
        replace_wrapper!(sc, si, new_sw) #sc.surrogates[si] = new_sw
    end
    return nothing
end

function improve_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    scal :: AbstractVarScaler,
    id :: AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; 
    ensure_fully_linear :: Bool = false, kwargs... ) :: Nothing 
    @logmsg loglevel2 "Improving surrogate models."

    meta_array = SurrogateMeta[]

    for sw in list_of_wrappers( sc )
        func_indices = get_function_index_tuple(sw)
        push!(meta_array, 
            prepare_improve_model(
                get_model(sw), get_meta(sw), get_config(sw), func_indices,
                mop, scal, id, sdb, ac; ensure_fully_linear, meta_array ) 
        )
    end

    eval_missing!(sdb, mop, scal)

    for (si,sw) ∈ enumerate( list_of_wrappers(sc ) )
        _meta = meta_array[si]
        func_indices = get_function_index_tuple(sw)
        new_model, new_meta = improve_model( 
            get_model(sw), _meta, get_config(sw), func_indices, 
            mop, scal, id, sdb, ac; ensure_fully_linear )
        
        new_sw = init_wrapper(
            SurrogateWrapper, 
            get_config( sw ), new_model, new_meta,
            get_objective_indices( sw ), get_nl_eq_constraint_indices(sw),
            get_nl_ineq_constraint_indices(sw)
        )
        replace_wrapper!(sc, si, new_sw) #sc.surrogates[si] = new_sw
    end
    return nothing
end

