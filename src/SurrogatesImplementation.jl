
include("RbfModel.jl")
include("ExactModel.jl")
include("TaylorModel.jl")
include("LagrangeModel.jl")
#include("newRBF.jl");

struct SurrogateWrapper{
        O <: AbstractObjective,
        M <: SurrogateModel,
        I <: SurrogateMeta}
    objf :: O
    model :: M
    meta :: I
end

num_outputs( sw :: SurrogateWrapper ) :: Int = num_outputs(sw.objf);

# this immutable wrapper makes memoized functions trigger only once
struct SurrogateContainer{T <: AbstractVector{<:SurrogateWrapper} }
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

function num_outputs( sc :: SurrogateContainer ) :: Int
    return sum( num_outputs(sw) for sw ∈ sc.surrogates )
end

function fully_linear( sc :: SurrogateContainer )
    return all( fully_linear(sw.model) for sw ∈ sc.surrogates )
end

#=
function get_surrogate_from_output_index( sc :: SurrogateContainer, ℓ :: Int, 
    mop :: AbstractMOP ) :: Union{Nothing,Tuple{SurrogateWrapper,Int}}
    for sw ∈ sc.surrogates
        objf_out_indices = output_indices( sw.objf, mop );
        l = findfirst( x -> x == ℓ, objf_out_indices );
        if !isnothing(l)
            return sw, l 
        end
    end
    return nothing
end
=#

@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( mop :: AbstractMOP, id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig ) :: SurrogateContainer
    @logmsg loglevel2 "Initializing surrogate models."
    
    sw_array = SurrogateWrapper[]
    output_model_mapping = Dict{Int,Tuple{Int,Int}}()
    sc_output_model_mapping = Dict{Int,Tuple{Int,Int}}()

    meta_array = SurrogateMeta[]

    for objf ∈ list_of_objectives(mop)
        push!( meta_array, prepare_init_model( model_cfg( objf ), objf, mop, id, db, ac; meta_array ))
    end

    @logmsg loglevel2 "Evaluation of unevaluated results."
    eval_missing!(db, mop)
    
    L = 1
    for (i, objf) ∈ enumerate(list_of_objectives(mop))
        meta_dat = meta_array[i]
        model, meta = _init_model( model_cfg(objf), objf, mop, id, db, ac, meta_dat )
        push!( sw_array, SurrogateWrapper(objf, model, meta) )

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
    for (si,sw) ∈ enumerate(sc.surrogates)
        push!(meta_array, prepare_update_model(sw.model, sw.objf, sw.meta, mop, id, db, ac; ensure_fully_linear, meta_array ) )
    end
    eval_missing!(db, mop)

    for (si,sw) ∈ enumerate(sc.surrogates)
        meta_dat = meta_array[si]
        new_model, new_meta = update_model( sw.model, sw.objf, meta_dat, mop, id, db, ac; ensure_fully_linear )
        new_sw = SurrogateWrapper( 
            sw.objf,
            new_model, 
            new_meta
        )
        sc.surrogates[si] = new_sw
    end
    nothing
end

function improve_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    @logmsg loglevel2 "Improving surrogate models."

    meta_array = SurrogateMeta[]
    for (si,sw) ∈ enumerate(sc.surrogates)
        push!(meta_array, prepare_improve_model(sw.model, sw.objf, sw.meta, mop, id, db, ac; ensure_fully_linear ) )
    end
    eval_missing!(db, mop)

    for (si,sw) ∈ enumerate(sc.surrogates)
        meta_dat = meta_array[si]
        new_model, new_meta = improve_model( sw.model, sw.objf, meta_dat, mop, id, ac; ensure_fully_linear )
        new_sw = SurrogateWrapper( 
            sw.objf,
            new_model, 
            new_meta
        )
        sc.surrogates[si] = new_sw
    end
    nothing
end

function eval_models( sc :: SurrogateContainer, x̂ :: Vec ) :: Vec
    vcat( (eval_models(sw.model , x̂) for sw ∈ sc.surrogates )...)
end

function get_jacobian( sc :: SurrogateContainer, x̂ :: Vec) :: Mat
    model_jacobians = [ get_jacobian(sw.model, x̂) for sw ∈ sc.surrogates ]
    vcat( model_jacobians... )
end

@doc """
Return a function handle to be used with `NLopt` for output `l` of `sc`.
Index `l` is assumed to be an *internal* index in the range of `1, …, n_objfs`,
where `n_objfs` is the total number of (scalarized) objectives stored in `sc`.
"""
function get_optim_handle( sc :: SurrogateContainer, l :: Int )
    i, ℓ = sc.sc_output_model_mapping[l]
    sw = sc.surrogates[i]
    return _get_optim_handle( sw.model, ℓ )
end

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `model`.
That is, if `model` is a surrogate for two scalar objectives, then `ℓ` must 
be either 1 or 2.
"""
function _get_optim_handle( model :: SurrogateModel, ℓ :: Int )
    # Return an anonymous function that modifies the gradient if present
    function (x :: Vec, g :: Vec)
        if !isempty(g)
            g[:] = get_gradient( model, x, ℓ)
        end
        return eval_models( model, x, ℓ)
    end
end

# These are used for PS constraints
# one can in theory provide vector constraints but most solvers fail then
@doc """
Return model value for output `l` of `sc` at `x̂`.
Index `l` is assumed to be an *internal* index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function eval_models( sc :: SurrogateContainer, x̂ :: Vec, l :: Int )
    i, ℓ = sc.sc_output_model_mapping[l]
    sw = sc.surrogates[i]
    return eval_models( sw.model, x̂, ℓ)
end

@doc """
Return a gradient for output `l` of `sc` at `x̂`.
Index `l` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_gradient( sc :: SurrogateContainer, x̂ :: Vec, l :: Int )
    i, ℓ = sc.sc_output_model_mapping[l]
    sw = sc.surrogates[i]
    return get_gradient( sw.model, x̂, ℓ );
end


#= 

MOP = [
(1)     A (rbf)
(2)     B (exact)
(3)     C (rbf)
]

becomes

SC = [
(1)     [
            A;
            C
        ] (rbf)
(2)     B  (exact)
]

hence 
sc.output_model_mapping =    Dict( 1 => (1,1), 2 => (3,1), 3 => (1,2) )
sc.sc_output_model_mapping = Dict( 1 => (1,1), 2 => (1,2), 3 => (2,1) )

TODO make (better/new) tests for this
=#