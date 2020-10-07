# This file defines the required data structures and methods for vector-valued
# Taylor models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.

@with_kw struct ExactModel <: SurrogateModel
    objf_obj :: Union{Nothing, VectorObjectiveFunction} = nothing;
end
broadcastable( em :: ExactModel ) = Ref(em);
fully_linear( em :: ExactModel ) = true;

struct ExactMeta end;

@doc "Return an ExactModel build from a VectorObjectiveFunction `objf` and corresponding
`cfg::ExactConfig` (ideally with `objf.model_config == cfg`).
Model is the same inside and outside of criticality round."
function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction, cfg :: ExactConfig, crit_flag :: Bool = true )
    em = ExactModel(
        objf_obj = objf,
    )
    return em, ExactMeta
end

@doc "Evaluate the ExactModel `em` at scaled site `ξ`."
function eval_models( em :: ExactModel, ξ :: Vector{Float64} )
    em.objf_obj( unscale(ξ) )
end

@doc "Evaluate output `ℓ` of the ExactModel `em` at scaled site `ξ`."
function eval_models( em :: ExactModel, ξ :: Vector{Float64}, ℓ :: Int64)
    return em.objf_obj( unscale(ξ) )[ℓ]
end

@doc "Gradient vector of output `ℓ` of `em` at scaled site `ξ`."
function get_gradient( em :: ExactModel, ξ :: Vector{Float64}, ℓ :: Int64)
    return em.objf_obj.gradient_handles[ℓ]( unscale(ξ) )
end

@doc "Jacobian Matrix of ExactModel `em` at scaled site `ξ`"
function get_jacobian( em :: ExactModel, ξ :: Vector{Float64} )
    get_jacobian( em.objf_obj, unscale(ξ) )   # simply delegate work to method of VectorObjectiveFunction
end
