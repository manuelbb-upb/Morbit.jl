# This file defines the required data structures and methods for vector-valued
# RBF models.
#
# This file is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.
# NOTE Scaling of training SITES is provided by functions
# (un)scale( mop :: MixedMop, x )
# imported from "Objectives.jl" in "Surrogates.jl"
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
Broadcast.broadcastable( M :: RbfModel ) = Ref(M);

@with_kw mutable struct RbfConfig <: SurrogateConfig
    kernel :: Symbol = :cubic;
    shape_parameter :: Union{F where {F<:Function}, R where R<:Real} = 1;
    polynomial_degree :: Int64 = 1;

    θ_enlarge_1 :: Real = 2;
    θ_enlarge_2 :: Real = 5;  # reset
    θ_pivot :: Real = 1 / (2 * θ_enlarge_1);
    θ_pivot_cholesky :: Real = Float16(1e-7);

    require_linear :: Bool = false;

    max_model_points :: Int64 = -1; # is probably reset in the algorithm
    use_max_points :: Bool = false;

    sampling_algorithm :: Symbol = :orthogonal # :orthogonal or :monte_carlo

    constrained :: Bool = false;    # restrict sampling of new sites

    max_evals :: Int64 = typemax(Int64);

    @assert sampling_algorithm ∈ [:orthogonal, :monte_carlo] "Sampling algorithm must be either `:orthogonal` or `:monte_carlo`."
    @assert kernel ∈ Symbol.(["exp", "multiquadric", "cubic", "thin_plate_spline"]) "Kernel '$kernel' not supported yet."
    @assert kernel != :thin_plate_spline || shape_parameter isa Int && shape_parameter >= 1
    #@assert θ_enlarge_1 >=1 && θ_enlarge_2 >=1 "θ's must be >= 1."
end

# meta data object to be used during sophisticated sampling
@with_kw mutable struct RBFMeta <: SurrogateMeta
    center_index :: Int64 = 1;
    round1_indices :: Vector{Int64} = [];
    round2_indices :: Vector{Int64} = [];
    round3_indices :: Vector{Int64} = [];
    round4_indices :: Vector{Int64} = [];
    fully_linear :: Bool = false;
    Y :: RMat = Matrix{Real}(undef, 0, 0);
    Z :: RMat = Matrix{Real}(undef, 0, 0);
end

fully_linear(m :: RBFModel) = m.fully_linear

max_evals( cfg :: RbfConfig ) = cfg.max_evals;

@doc "Modify first meta data object to equal second."
function as_second!(destination :: RBFMeta, source :: RBFMeta )
    @unpack_RBFMeta source;
    @pack_RBFMeta! destination;
    return nothing
end

# use functions from base module for evaluation
# (assumes that models are trained on the unit hypercube)
eval_models( m :: RBFModel, x :: RVec) = RBF.output(m, x)
eval_models( m :: RBFModel, x :: RVec, ℓ :: Int64) = RBF.output(m, ℓ, x)
get_gradient( m :: RBFModel, x :: RVec, ℓ :: Int64) = RBF.grad( m, ℓ, x )
get_jacobian( m :: RBFModel, x :: RVec ) = RBF.jac( m, x )

include("rbf_sampling.jl")

# Redefine interface methods for our own Model type
@forward RbfModel.model fully_linear, eval_models, get_gradient, get_jacobian
@forward RbfModel.model improve!, make_linear!

function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: RbfConfig, criticality_round :: Bool = false )
    model, meta = build_rbf_model( ac, objf, cfg, criticality_round)
    return RbfModel(model), meta
end
