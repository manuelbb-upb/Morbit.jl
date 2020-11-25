# This file defines the required data structures and methods for vector-valued
# Taylor models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.
# NOTE Scaling of training SITES is provided by functions
# (un)scale( mop :: MixedMop, x )
# imported from "Objectives.jl" in "Surrogates.jl"

@with_kw mutable struct TaylorConfig <: ModelConfig
    n_out :: Int64 = 1; # used internally when setting hessians
    degree :: Int64 = 1;

    gradients :: Union{Symbol, Nothing, Vector{T} where T, F where F<:Function } = :fdm
    hessians ::  Union{Symbol, Nothing, Vector{T} where T, F where F<:Function } = gradients

    # alternative to specifying individual gradients
    jacobian :: Union{Symbol, Nothing, F where F<:Function} = nothing

    max_evals :: Int64 = typemax(Int64);
end

@with_kw mutable struct TaylorModel <: SurrogateModel
    n_out :: Int64 = -1;
    degree :: Int64 = 2;
    x :: Vector{R} where{R<:Real} = Float64[];
    f_x :: Vector{R} where{R<:Real} = Float64[];
    g :: Vector{Vector{R}} where{R<:Real} = Array{Float64,1}[];
    H :: Vector{Array{R,2}} where{R<:Real} = Array{Float64,2}[];
    unscale_function :: Union{Nothing, F where F<:Function} = nothing;
    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
end
Broadcast.broadcastable( tm :: TaylorModel ) = Ref(tm);

struct TaylorMeta <: SurrogateMeta end   # no construction meta data needed

fully_linear( tm :: TaylorModel ) = true;

max_evals( cfg :: TaylorConfig ) = cfg.max_evals;

function prepare!(objf :: VectorObjectiveFunction, cfg :: TaylorConfig, ::AlgoConfig )
    @info("Preparing Taylor Models")
    set_gradients!( cfg, objf )
    if cfg.degree == 2
        set_hessians!(cfg, objf)
    end
end

@doc "Return a TaylorModel build from a VectorObjectiveFunction `objf` and corresponding
`cfg::TaylorConfig` (ideally with `objf.model_config == cfg`).
Model is the same inside and outside of criticality round."
function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: TaylorConfig, crit_flag :: Bool = true)
    @unpack problem = ac;
    @info "BUILDING TAYLOR MODEL"
    tm = TaylorModel(
        n_out = objf.n_out,
        degree = cfg.degree,
        unscale_function = x -> unscale(problem, x),
        x = unscale(problem, ac.iter_data.x),
        f_x = ac.iter_data.f_x[ objf.internal_indices ]  # hopefully == objf(unscale(x))
    )

    do_hessians = tm.degree == 2;
    for ℓ = 1 : tm.n_out
        push!( tm.g, objf.model_config.gradients[ℓ](tm.x) )
        if do_hessians
            push!( tm.H, objf.model_config.hessians[ℓ](tm.x) )
        end
    end
    # Note: to save the evaluation sites, we could either have `objf` store the evals
    # or maybe retrieve them from the Gradient Cache of FiniteDiff
    return tm, TaylorMeta()
end

@doc "Return vector of evaluations for all outputs of a (vector) Taylor model
`tm` at scaled site `ξ`."
function eval_models( tm :: TaylorModel, ξ :: Vector{Float64} )
    h = tm.unscale_function(ξ) .- tm.x;
    return eval_models( tm :: TaylorModel, h, Val( tm.degree ) )
end
function eval_models( tm :: TaylorModel, h :: Vector{Float64}, ::Val{1})
    return vcat( [ tm.f_x[ℓ] + h'tm.g[ℓ] for ℓ=1:tm.n_out ]... )
end
function eval_models( tm :: TaylorModel, h :: Vector{Float64}, ::Val{2})
    return vcat( [ tm.f_x[ℓ] + h'tm.g[ℓ] + .5 * h'tm.H[ℓ]*h for ℓ=1:tm.n_out ]...)
end

@doc "Return vector of evaluations for output `ℓ` of a (vector) Taylor model
`tm` at scaled site `ξ`."
function eval_models( tm :: TaylorModel, ξ :: Vector{Float64} , ℓ :: Int64 )
    h = tm.unscale_function(ξ) .- tm.x;
    return eval_models( tm, h, ℓ, Val(tm.degree))
end
function eval_models( tm :: TaylorModel, h :: Vector{Float64}, ℓ :: Int64, ::Val{1} )
    tm.f_x[ℓ] + h'tm.g[ℓ]
end
function eval_models( tm :: TaylorModel, h :: Vector{Float64}, ℓ :: Int64, ::Val{2} )
    tm.f_x[ℓ] + h'tm.g[ℓ] + .5 * h'tm.H[ℓ]*h
end

@doc "Gradient vector of output `ℓ` of `tm` at scaled site `ξ`."
function get_gradient( tm :: TaylorModel, ξ :: Vector{Float64}, ℓ :: Int64)
    if tm.degree == 1
        return tm.g[ℓ]
    else
        h = tm.unscale_function(ξ) .- tm.x;
        return tm.g[ℓ] + .5 * ( tm.H[ℓ]' + tm.H[ℓ] ) * h
    end
end

@doc "Return Jacobian matrix of (vector-valued) Taylor Model `tm` at scaled site `ξ`."
function get_jacobian( tm :: TaylorModel, ξ :: Vector{Float64})
    grad_list = [get_gradient(tm, ξ, ℓ) for ℓ=1:tm.n_out ]
    return transpose( hcat( grad_list... ) )
end

# The following functions are a bit redundant becase `tm` is always fully linear.
#=
make_linear!( ::TaylorModel, ::TaylorMeta, ::AlgoConfig, ::VectorObjectiveFunction, ::TaylorConfig, ::Bool ) = false;
improve!( ::TaylorModel, ::TaylorMeta, ::AlgoConfig, ::VectorObjectiveFunction, ::TaylorConfig )  = false;
=#
