#module Surrogate

import Base.Broadcast: broadcastable, broadcasted
using Parameters: @with_kw, @pack!, @unpack

max_evals( m :: M where M <: ModelConfig ) = typemax(Int64);
prepare!(::VectorObjectiveFunction, :: M where M<:ModelConfig , ::AlgoConfig) = nothing

include("RBFModel.jl")
include("TaylorModel.jl")
include("ExactModel.jl")
include("LagrangeModel.jl")

include("objectives.jl");

include("misc.jl")
include("build_derivatives.jl")

@doc """
A container for all (possibly vector-valued) surrogate models used during
optimization.

# Fields
* `objf_list`: A list of `VectorObjectiveFunction`s to be used during surrogate
  training.
* `model_list`: The list of surragte models, all subtypes of `SurrogateModel`.
* `model_meta`: Meta data collected during training for debugging and plotting.
"""
@with_kw mutable struct SurrogateContainer
    objf_list :: Union{Nothing, Vector{VectorObjectiveFunction}} = nothing;
    model_list :: Vector{Any} = [] ;
    model_meta :: Vector{Any} = [] ;
    n_objfs :: Int64 = 0;   # number of scalar(ized) objectives
    #lb :: Union{Nothing, Vector{Float64}} = nothing;
    #width :: Union{Nothing, Vector{Float64}} = nothing; # ub - lb
    output_to_objf_index :: Union{Nothing, Vector{Tuple{Int64,Int64}}} = nothing;
end

function unscale( x :: Vector{Float64}, sc :: SurrogateContainer )
    if isnothing(sc.lb)
        return x
    else
        return sc .+ ( x .* ( sc.width ) )
    end
end

@doc "Return a SurrogateContainer initialized from the information provided in `mop`."
function init_surrogates( ac :: AlgoConfig )
    mop = ac.problem;
    sc = SurrogateContainer(
        objf_list = mop.vector_of_objectives,
        n_objfs = mop.n_objfs,
    )

    set_output_objf_list!(sc)
    for objf in sc.objf_list
        prepare!(objf, objf.model_config, ac)
    end
    return sc
end

function set_output_objf_list!(sc)
    sc.output_to_objf_index = Vector{NTuple{2,Int64}}();
    for (objf_index, objf) ∈ enumerate( sc.objf_list )
        for int_index ∈ eachindex(objf.internal_indices)
            push!(sc.output_to_objf_index, (objf_index, int_index))
        end
    end
end

function build_models!(sc :: SurrogateContainer, ac :: AlgoConfig, crit_flag :: Bool = false )
    sc.model_list = Any[];
    sc.model_meta = Any[]
    for (ℓ,objf) ∈ enumerate(sc.objf_list)
        new_model, new_meta = build_model( ac, objf, objf.model_config, crit_flag )
        push!(sc.model_list, new_model)
        push!(sc.model_meta, new_meta)
    end
end

function improve!(sc :: SurrogateContainer, non_linear_indices :: Vector{Int64},
        ac :: AlgoConfig )
    for ℓ ∈ non_linear_indices
        model = sc.model_list[ℓ]
        meta = sc.model_meta[ℓ]
        objf = sc.objf_list[ℓ]
        improve!(model, meta, ac, objf, objf.model_config)
    end
end

function make_linear!(sc :: SurrogateContainer, non_linear_indices :: Vector{Int64},
        ac :: AlgoConfig )
    has_changed = false
    for ℓ ∈ non_linear_indices
        model = sc.model_list[ℓ]
        meta = sc.model_meta[ℓ]
        objf = sc.objf_list[ℓ]
        has_changed *= make_linear!(model, meta, ac, objf, objf.model_config)
    end
    return has_changed
end

function eval_models( sc :: SurrogateContainer, x :: Vector{Float64} )
    vcat( ( eval_models(model , x) for model ∈ sc.model_list )...)
end

function get_jacobian( sc :: SurrogateContainer, x :: Vector{Float64})
    model_jacobians = [ get_jacobian(model , x) for model ∈ sc.model_list ]
    vcat( model_jacobians... )
end

# let each surrogate handle its broadcasting behavior itself...
# # (ExactModel might be BatchObjectiveFunction)
function broadcasted( f :: typeof(eval), sc::SurrogateContainer, X :: Vector{Vector{Float64}} )
    [ vcat(z...) for z ∈ zip( [ eval.(model, X) for model ∈ sc.model_list ]... ) ]
end

@doc """
Return `true` (and `[]`) if all models in `sc.model_list` qualify as fully linear.
Return `false` (and a list of the violating indices wrt `sc.model_list`) elsewise.
"""
function fully_linear( sc :: SurrogateContainer )
    linear_flag = true;
    non_linear_indices = Int64[];
    for (model_index, model) ∈ enumerate(sc.model_list)
        if !fully_linear(model)
            linear_flag = false
            push!(non_linear_indices, model_index);
        end
    end
    return linear_flag, non_linear_indices
end

@doc """
Return a function handle to be used with `NLopt` for output `l` of `model`.
Assume `l` to be in [1, …, `model.n_out`].
"""
function get_optim_handle( model :: M where{ M <: SurrogateModel}, l :: Int64 )
    function (x :: Vector{Float64}, g :: Vector{Float64})
        if !isempty(g)
            g[:] = get_gradient( model, x, l)
        end
        return eval_models( model, x, l)
    end
end

@doc """
Return a function handle to be used with `NLopt` for output `ℓ` of `sc`.
Index `ℓ` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_optim_handle( sc :: SurrogateContainer, ℓ :: Int64 )
    #for (objf_index, objf) ∈ enumerate(sc.objf_list)
        #l = findfirst( objf.internal_indices == ℓ )
        #if !isnothing(l)
        i,l = sc.output_to_objf_index[ℓ]
            return get_optim_handle( sc.model_list[i], l )
        #end
    #end
end

@doc """
Return a  gradient handle to be used with `NLopt` for output `ℓ` of `sc`.
Index `ℓ` is assumed to be an internal index in the range of 1,…,n_objfs,
where n_objfs is the total number of (scalarized) objectives stored in `sc`.
"""
function get_gradient( sc :: SurrogateContainer, ξ :: Vector{Float64}, ℓ :: Int64 )
    i,l = sc.output_to_objf_index[ℓ]
    return get_gradient( sc.model_list[i], ξ, l )
end

@doc "Output ℓ of model container at site x."
function eval_models( sc :: SurrogateContainer, x :: Vector{Float64}, ℓ :: Int64 )
    i,l = sc.output_to_objf_index[ℓ]
    return eval_models( sc.model_list[i], x, l )
end
#end#module
