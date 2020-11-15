####################### Surrogate Stuff Abstract #################
abstract type ModelConfig end
abstract type SurrogateModel end
abstract type SurrogateMeta end

max_evals( m :: M where M <: ModelConfig ) = typemax(Int64);

############## CUSTOM FUNCTION SUB TYPES ##############

# … to allow for batch evaluation outside of julia
# or to exploit parallelized objective functions
@with_kw struct BatchObjectiveFunction <: Function
    function_handle :: Union{T, Nothing} where{T<:Function} = nothing
end

# Main type of function used internally
@with_kw mutable struct VectorObjectiveFunction <: Function
    n_out :: Int64 = 0;
    n_evals :: Int64 = 0;   # true function evaluations (also counts fdm evaluations)

    max_evals :: Int64 = typemax(Int64);

    model_config :: Union{ Nothing, C } where {C <: ModelConfig } = nothing;

    function_handle :: Union{T, Nothing} where{T <: Function, F <: Function} = nothing

    internal_indices :: Vector{Int64} = [];
    problem_position :: Int64 = 0;
end

############# RBF Model Types ##############
# wrapper to make RBFModel a subtype of SurrogateModel interface
# # NOTE the distinction:
# # • RBFModel from module .RBF
# # • RbfModel the actual SurrogateModel used in the algorithm
struct RbfModel <: SurrogateModel
    model :: RBFModel
end
Broadcast.broadcastable( M :: RbfModel ) = Ref(M);

@with_kw mutable struct RbfConfig <: ModelConfig
    kernel :: Symbol = :multiquadric;
    shape_parameter :: Union{F where {F<:Function}, R where R<:Real} = 1;
    polynomial_degree :: Int64 = 1;

    θ_enlarge_1 :: Float64 = 2.0;
    θ_enlarge_2 :: Float64 = 5.0;  # reset
    θ_pivot :: Float64 = 1.0 / (2 * θ_enlarge_1);
    θ_pivot_cholesky :: Float64 = 1e-7;

    require_linear = false;

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
    Y :: Array{Float64,2} = Matrix{Float64}(undef, 0, 0);
    Z :: Array{Float64,2} = Matrix{Float64}(undef, 0, 0);
end

############# Exact Model Types ##############

@with_kw struct ExactModel <: SurrogateModel
    objf_obj :: Union{Nothing, VectorObjectiveFunction} = nothing;
    unscale_function :: Union{Nothing, F where F<:Function} = nothing;
end
Broadcast.broadcastable( em :: ExactModel ) = Ref(em);

@with_kw mutable struct ExactConfig <: ModelConfig
    gradients :: Union{Symbol, Nothing, Vector{T} where T, F where F<:Function } = :autodiff

    # alternative keyword, usage discouraged...
    jacobian :: Union{Symbol, Nothing, F where F<:Function} = nothing

    max_evals :: Int64 = typemax(Int64)
end

struct ExactMeta <: SurrogateMeta end   # no construction meta data needed

############# Taylor Model Types ##############
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

############# Lagrange Model Types ##############
@with_kw struct LagrangeModel <: SurrogateModel
    n_out :: Int64 = -1;
    degree :: Int64 = 1;

    lagrange_basis :: Vector{Polynomial{N} where N} = [];
    coefficients :: Vector{Vector{Float64}} = [];
    fully_linear :: Bool = false;
end
Broadcast.broadcastable( lm :: LagrangeModel ) = Ref(lm);

@with_kw mutable struct LagrangeConfig <: ModelConfig
    degree :: Int64 = 1;
    θ_enlarge :: Float64 = 2;

    ε_accept :: Float64 = 1e-6;
    Λ :: Float64 = 1.5;

    allow_not_linear :: Bool = true;

    # the basis is set by `prepare!`
    canonical_basis :: Union{ Nothing, Vector{Polynomial} } = nothing

    max_evals :: Int64 = typemax(Int64);
end

@with_kw mutable struct LagrangeMeta <: SurrogateMeta
    interpolation_indices :: Vector{Int64} = [];
end

###############  MULTIOBJECTIVE OPTIMIZATION PROBLEM DEFINITION ##############

@with_kw mutable struct MixedMOP
    #vector_of_objectives :: Vector{ Union{ ObjectiveFunction, VectorObjectiveFunction } } = [];
    vector_of_objectives :: Vector{ VectorObjectiveFunction } = [];
    n_objfs :: Int64 = 0;

    internal_sorting :: Vector{Int64} = [];
    reverse_sorting :: Vector{Int64} = [];
    non_exact_indices ::Vector{Int64} = [];     # indices of vector returned by `eval_all_objectives` corresponding to non-exact objectives

    x_0 :: Vector{Float64} = [];
    lb :: Union{Nothing,Vector{Float64}} = nothing;
    ub :: Union{Nothing,Vector{Float64}} = nothing;
    is_constrained = !( isnothing(lb) || isnothing(ub) )

    #store functions and n_out for research & debugging 
    original_functions :: Vector{Tuple{F where F<:Function, Int}} = [];
end
Broadcast.broadcastable(m::MixedMOP) = Ref(m);

##################### INNER SETTINGS #########################################

# collectible data during iterations (used for plotting and analysis)
@with_kw mutable struct IterData
    # "global" data (used in each iteration)
    x :: Vector{Float64} = []  # current iteration site
    f_x :: Vector{Float64} = []  # true objective values at current iterate,
    x_index :: Int64 = 0;
    Δ :: Float64 = 0.0;
    sites_db :: Vector{Vector{Float64}} = []; # array of all sites that have been evaluated (RBF or Lagrange)
    values_db :: Vector{Vector{Float64}} = []; # array of all true values computed so far

    model_meta :: Array{Any,1} = Any[]; # iteration dependent surrogate data

    # Arrays (1 entry per iteration)
    iterate_indices :: Vector{ Int64 } = [];
    trial_point_indices :: Vector{Int64} = [];
    stepsize_array :: Vector{Float64} = [];  # a bit redundant, since iterates are given
    Δ_array :: Vector{Float64} = [];
    ω_array :: Vector{Float64} = [];
    ρ_array :: Vector{Float64} = [];
    num_crit_loops_array :: Vector{Int64} = [];
end
Broadcast.broadcastable(id :: IterData) = Ref(id);

@with_kw mutable struct AlgoConfig

    n_vars ::Int64 = 0; # is reset during optimization
    n_objfs :: Int64 = 0; # total number of (scalarized) objectives

    #ε_bounds = 0.0;   # minimum distance to boundaries if problem is constraint, needed for functions undefined outside bounds

    problem :: Union{MixedMOP,Nothing} = nothing;

    max_iter :: Int64 = 500;
    max_evals :: Int64 = typemax(Int64);    # maxiumm number of expensive function evaluations

    descent_method :: Symbol = :steepest # :steepest or :direct_search ( TODO implement local Pascoletti-Serafini )
    ideal_point :: Vector{Float64} = [];
    image_direction :: Vector{Float64} = [];
    θ_ideal_point :: Float64 = 1.5;

    all_objectives_descent :: Bool = false;  # compute ρ as the minimum of descent ratios for ALL objetives

    radius_update :: Symbol = :standard # :standard or :steplength

    # criticallity parameters
    μ :: Float64 = 2e3;
    β :: Float64 = 1e3;
    ε_crit :: Float64 = 1e-3;
    max_critical_loops :: Int64 = 30;
    true_ω_stop :: Float64 = Inf;

    # acceptance parameters
    ν_success :: Float64 = 0.8;
    ν_accept :: Float64 = -1e-15;
    # trust region update parameters
    γ_crit :: Float64 = 0.5; # scaling factor for Δ in criticallity test
    γ_grow :: Float64 = 2;
    γ_shrink :: Float64 = 0.9;
    γ_shrink_much :: Float64 = 0.501;

    Δ₀ :: Float64 = 0.1;
    Δ_max :: Float64 = 1.0;

    # additional stopping criteria (mostly inspired by thoman)
    Δ_critical = 1e-4;
    Δ_min = Δ_critical * 1e-3;
    stepsize_min = Δ_critical * 1e-2;   # stop if Δ < Δ_critical & step_size < stepsize_min
    # NOTE thomann uses stepsize in image space due to PS scalarization

    iter_data :: Union{Nothing,IterData} = nothing; # Refercnce to an object storing the iteration information

    # assertions for parameter consistency
    #@assert 0 <= θ_pivot <= 1/θ_enlarge_1 "θ_pivot = $θ_pivot must be in range [0, $(1/θ_enlarge_1)]."
    @assert β <= μ "μ = $μ must be larger than or equal to β = $β."
    @assert Δ₀ <= Δ_max "Δ_max = $Δ_max is smaller than initial trust region radius Δ₀ = $Δ₀."
    @assert ν_accept <= ν_success "Acceptance parameters must be 0<= ν_accept <= ν_success."
    @assert 0 < γ_crit < 1 "Criticality reduction factor γ_crit must be in (0,1)."
    @assert 0 < γ_shrink <= 1 "Trust region reduction factor γ_shrink must be in (0,1]"
    @assert 1 < γ_grow "Trust region grow factor γ_grow must be bigger than 1."
    @assert max_iter > 0 "Maximal number of iterations must be a positive integer."
    # TODO make sure Δ_max is bounded by a fraction of global boundaries (requires mutable struct?)

end
Broadcast.broadcastable(id :: AlgoConfig) = Ref(id);

# Outer Constructor to obtain default configuration adapted for n_vars input variables.
AlgoConfig( n_vars ) = AlgoConfig( θ_enlarge_2 = max(sqrt(n_vars), 4) )
