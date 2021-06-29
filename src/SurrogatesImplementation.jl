
#include("RBFModel.jl")
include("ExactModel.jl")
#include("TaylorModel.jl")
#include("LagrangeModel.jl")
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
struct SurrogateContainer 
    surrogates :: Vector{<:SurrogateWrapper} 
end

@memoize ThreadSafeDict function num_outputs( sc :: SurrogateContainer ) :: Int
    return sum( num_outputs(sw) for sw ∈ sc.surrogates )
end

function fully_linear( sc :: SurrogateContainer )
    return all( fully_linear(sw.model) for sw ∈ sc.surrogates )
end

# should only be called once `sc` is fully initialized
@memoize ThreadSafeDict function get_surrogate_from_output_index( sc :: SurrogateContainer, ℓ :: Int, 
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

@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( mop :: AbstractMOP, id :: AbstractIterData, ac :: AbstractConfig ) :: SurrogateContainer
    @logmsg loglevel2 "Initializing surrogate models."
    sw_array = SurrogateWrapper[]
    for objf ∈ list_of_objectives(mop)
        model, meta = init_model( objf, mop, id, ac );
        append!( sw_array, SurrogateWrapper(objf,model,meta))
    end
    return SurrogateContainer(sw_array)
end

function update_surrogates!( sc :: SurrogateContainer, mop :: AbstractMOP, 
    id :: AbstractIterData, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    @logmsg loglevel2 "Updating surrogate models."
    for (si,sw) ∈ enumerate(sc.surrogates)
        new_model, new_meta = update_model( sw.model, sw.objf, sw.meta, mop, id, ac; ensure_fully_linear )
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
    id :: AbstractIterData, ac :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Nothing 
    @logmsg loglevel2 "Improving surrogate models."
    for (si,sw) ∈ enumerate(sc.surrogates)
        new_model, new_meta = improve_model( sw.model, sw.objf, sw.meta, mop, id, ac; ensure_fully_linear )
        new_sw = SurrogateWrapper( 
            sw.objf,
            new_model, 
            new_meta
        );
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
Index `l` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_optim_handle( sc :: SurrogateContainer, mop :: AbstractMOP, l :: Int )
    sw, ℓ = get_surrogate_from_output_index( sc, l, mop )
    return get_optim_handle( sw.model, ℓ )
end

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `model`.
That is, if `model` is a surrogate for two scalar objectives, then `ℓ` must 
be either 1 or 2.
"""
function get_optim_handle( model :: SurrogateModel, ℓ :: Int )
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
Index `l` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function eval_models( sc :: SurrogateContainer, mop :: AbstractMOP, x̂ :: Vec, l :: Int )
    sw, ℓ = get_surrogate_from_output_index( sc, l, mop );
    return eval_models( sw.model, x̂, ℓ)
end

@doc """
Return a gradient for output `l` of `sc` at `x̂`.
Index `4` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_gradient( sc :: SurrogateContainer, mop :: AbstractMOP, x̂ :: Vec, l :: Int )
    sw, ℓ = get_surrogate_from_output_index( sc, l, mop );
    return get_gradient( sw.model, x̂, ℓ );
end