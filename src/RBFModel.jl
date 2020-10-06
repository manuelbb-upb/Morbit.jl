# This file defines the required data structures and methods for vector-valued
# Taylor models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.
# NOTE Scaling of training SITES is provided by functions
# (un)scale( mop :: MixedMop, x )
# imported from "Objectives.jl" in "Surrogates.jl"

# import basic definitions from standalone module
using Lazy: @forward
include("RBFBase.jl")
import .RBF: RBFModel, train!, is_valid, get_Π, get_Φ, φ, Π_col, as_second!, min_num_sites

# wrapper to make RBFModel a subtype of SurrogateModel interface
# # NOTE the distinction:
# # • RBFModel from module .RBF
# # • RbfModel the actual SurrogateModel used in the algorithm
struct RbfModel <: SurrogateModel
    model :: RBFModel
end
broadcastable( M :: RbfModel ) = Ref(M);
fully_linear(m :: RBFModel) = m.fully_linear

# Redefine imported methods for our own Model type
#@forward RbfModel.model train!, is_valid, get_Π, get_Φ, φ, Π_col, as_second!

# meta data object to be used during sophisticated sampling
@with_kw mutable struct RBFMeta
    center_index :: Int64 = 1;
    round1_indices :: Vector{Int64} = [];
    round2_indices :: Vector{Int64} = [];
    round3_indices :: Vector{Int64} = [];
    round4_indices :: Vector{Int64} = [];
    fully_linear :: Bool = false;
    Y :: Array{Float64,2} = Matrix{Float64}(undef, 0, 0);
    Z :: Array{Float64,2} = Matrix{Float64}(undef, 0, 0);
end

@doc "Modify first meta data object to equal second."
function as_second!(destination :: RBFMeta, source :: RBFMeta )
    @unpack_RBFMeta source;
    @pack_RBFMeta! destination;
    return nothing
end

# use functions from base module for evaluation
# (assumes that models are trained on the unit hypercube)
eval_models( m :: RBFModel, x :: Vector{Float64}) = RBF.output(m, x)
get_gradient( m :: RBFModel, x :: Vector{Float64}, ℓ :: Int64) = RBF.grad( m, ℓ, x )
get_jacobian( m :: RBFModel, x :: Vector{Float64}) = RBF.jac( m, x )

include("rbf_sampling.jl")

# Redefine interface methods for our own Model type
@forward RbfModel.model fully_linear, eval_models, get_gradient, get_jacobian
@forward RbfModel.model improve!, make_linear!

function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: RbfConfig, criticality_round :: Bool = false )
    model, meta = build_rbf_model( ac, objf, cfg, criticality_round)
    return RbfModel(model), meta
end
