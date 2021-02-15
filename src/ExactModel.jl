# This file defines the required data structures and methods for vector-valued
# Taylor models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.

@with_kw struct ExactModel <: SurrogateModel
    objf_obj :: Union{Nothing, VectorObjectiveFunction} = nothing;
    unscale_function :: Union{Nothing, F where F<:Function} = nothing;
end

@with_kw mutable struct ExactConfig <: SurrogateConfig
    gradients :: Union{Symbol, Nothing, Vector{T} where T, F where F<:Function } = :autodiff

    # alternative keyword, usage discouraged...
    jacobian :: Union{Symbol, Nothing, F where F<:Function} = nothing

    max_evals :: Int64 = typemax(Int64)
end

struct ExactMeta <: SurrogateMeta end   # no construction meta data needed

fully_linear( em :: ExactModel ) = true;
max_evals( emc :: ExactConfig ) = emc.max_evals

function prepare!(objf :: VectorObjectiveFunction, cfg :: ExactConfig, ::AlgoConfig )
    @info("Preparing Exact Models")
    set_gradients!( cfg, objf )
end

@doc "Return an ExactModel build from a VectorObjectiveFunction `objf` and corresponding
`cfg::ExactConfig` (ideally with `objf.model_config == cfg`).
Model is the same inside and outside of criticality round."
function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: ExactConfig, crit_flag :: Bool = true )
    @unpack problem = ac;
    em = ExactModel(
        objf_obj = objf,
        unscale_function = x -> unscale(problem, x)
    )
    return em, ExactMeta()
end

@doc "Evaluate the ExactModel `em` at scaled site `ξ`."
function eval_models( em :: ExactModel, ξ :: Vector{Float64} )
    em.objf_obj( em.unscale_function(ξ) )
end

@doc "Evaluate output `ℓ` of the ExactModel `em` at scaled site `ξ`."
function eval_models( em :: ExactModel, ξ :: Vector{Float64}, ℓ :: Int64)
    return em.objf_obj( em.unscale_function(ξ) )[ℓ]
end

@doc "Gradient vector of output `ℓ` of `em` at scaled site `ξ`."
function get_gradient( em :: ExactModel, ξ :: Vector{Float64}, ℓ :: Int64)
    return em.objf_obj.model_config.gradients[ℓ]( em.unscale_function(ξ) )
end

@doc "Jacobian Matrix of ExactModel `em` at scaled site `ξ`"
function get_jacobian( em :: ExactModel, ξ :: Vector{Float64} )
    get_jacobian( em.objf_obj.model_config, em.unscale_function(ξ) )   # simply delegate work to method of VectorObjectiveFunction
end
