
include("RbfModel.jl")
include("ExactModel.jl")
include("TaylorModel.jl")
include("LagrangeModel.jl")

struct SurrogateWrapper{
        O <: AbstractObjective,
        M <: SurrogateModel,
        I <: SurrogateMeta} <: AbstractSurrogateWrapper
    objf :: O
    model :: M
    meta :: I
end

function init_wrapper( :: Type{<:SurrogateWrapper}, objf :: AbstractObjective, 
    model :: SurrogateModel, meta :: SurrogateMeta)
    return SurrogateWrapper( objf, model, meta )
end

num_outputs( sw :: SurrogateWrapper ) :: Int = num_outputs(get_objf(sw))
get_objf( sw :: SurrogateWrapper ) = sw.objf
get_model( sw :: SurrogateWrapper ) = sw.model
get_meta( sw :: SurrogateWrapper ) = sw.meta

struct SurrogateContainer{
        T <: AbstractVector{<:AbstractSurrogateWrapper} } <: AbstractSurrogateContainer
    surrogates :: T
    
    "A dict of tuples where the key is a MOP output index and the 
    value is a tuple where the first entry is the index of the 
    corresponding surrogate in `surrogates` and the second entry is the 
    index of the surrogate model output (or 1 if it's a scalar models)."
    output_model_mapping :: Base.ImmutableDict{Int,Tuple{Int,Int}}

    "A dict of tuples where the key is an output index of the 
    SurrogateContainer (i.e., corresponding to the internal sorting) and the
    value is a tuple where the first entry is the index of the 
    corresponding surrogate in `surrogates` and the second entry is the 
    index of the surrogate model output (or 1 if it's a scalar models)."
    sc_output_model_mapping :: Base.ImmutableDict{Int,Tuple{Int,Int}}
end

list_of_wrappers( sc :: SurrogateContainer ) = sc.surrogates
get_wrapper( sc, i ) = sc.surrogates[i]

_output_model_mapping( sc :: SurrogateContainer, l ) = sc.output_model_mapping[l]
_sc_output_model_mapping( sc :: SurrogateContainer, l ) = sc.sc_output_model_mapping[l]

@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( ::Type{<:SurrogateContainer}, mop :: AbstractMOP, 
    id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig )
    
    @logmsg loglevel2 "Initializing surrogate models."
    
    # init fields for SurrogateContainer
    sw_array = SurrogateWrapper[]   # TODO make AbstractSurrogateWrapper type configurable?
    output_model_mapping = Dict{Int,Tuple{Int,Int}}()
    sc_output_model_mapping = Dict{Int,Tuple{Int,Int}}()

    meta_array = []

    for objf ∈ list_of_objectives(mop)
        push!( meta_array, prepare_init_model( model_cfg( objf ), objf, mop, id, db, ac; meta_array ))
    end

    @logmsg loglevel2 "Evaluation of unevaluated results."
    eval_missing!(db, mop)
    
    L = 1   # counter of scalarized outputs of `mop`
    for (i, objf) ∈ enumerate(list_of_objectives(mop))
        meta_dat = meta_array[i]
        model, meta = _init_model( model_cfg(objf), objf, mop, id, db, ac, meta_dat )
        push!( sw_array, init_wrapper(SurrogateWrapper, objf, model, meta) )

        for (j, ℓ) in enumerate( output_indices( objf, mop ) )
            output_model_mapping[ℓ] = (i,j)
            sc_output_model_mapping[L] = (i,j)
            L += 1
        end
    end

    return SurrogateContainer( 
        sw_array, 
        Base.ImmutableDict(output_model_mapping...), 
        Base.ImmutableDict(sc_output_model_mapping...)
    )
end

function update_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    @logmsg loglevel2 "Updating surrogate models."

    meta_array = SurrogateMeta[]
    for sw ∈ sc.surrogates
        push!(meta_array, prepare_update_model(get_model(sw), get_objf(sw), get_meta(sw), mop, id, db, ac; ensure_fully_linear, meta_array ) )
    end
    eval_missing!(db, mop)

    for (si,sw) ∈ enumerate(sc.surrogates)
        meta_dat = meta_array[si]
        new_model, new_meta = update_model( get_model(sw), get_objf(sw), meta_dat, mop, id, db, ac; ensure_fully_linear )
        new_sw = init_wrapper(
            SurrogateWrapper, 
            get_objf(sw),
            new_model, 
            new_meta
        )
        sc.surrogates[si] = new_sw
    end
    nothing
end

function improve_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; 
    ensure_fully_linear :: Bool = false, kwargs... ) :: Nothing 
    @logmsg loglevel2 "Improving surrogate models."

    meta_array = SurrogateMeta[]
    for sw ∈ sc.surrogates
        push!(meta_array, prepare_improve_model(get_model(sw), get_objf(sw), get_meta(sw), mop, id, db, ac; ensure_fully_linear ) )
    end
    eval_missing!(db, mop)

    for (si,sw) ∈ enumerate(sc.surrogates)
        meta_dat = meta_array[si]
        new_model, new_meta = improve_model( get_model(sw), get_objf(sw), meta_dat, mop, id, db, ac; ensure_fully_linear, kwargs... )
        new_sw = SurrogateWrapper( 
            get_objf(sw),
            new_model, 
            new_meta
        )
        sc.surrogates[si] = new_sw
    end
    nothing
end

