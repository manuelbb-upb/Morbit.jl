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
    θ_ideal_point :: Float64 = 2;

    all_objectives_descent :: Bool = false;  # compute ρ as the minimum of descent ratios for ALL objetives

    # criticallity parameters
    μ :: Float64 = 3e3;
    β :: Float64 = 5e3;
    ε_crit :: Float64 = 1e-3;
    max_critical_loops :: Int64 = 30;

    # acceptance parameters
    ν_success :: Float64 = 0.4;
    ν_accept :: Float64 = -1e-15;
    # trust region update parameters
    γ_crit :: Float64 = 0.5; # scaling factor for Δ in criticallity test
    γ_grow :: Float64 = 2;
    γ_shrink :: Float64 = 0.9;
    γ_shrink_much :: Float64 = 0.501;

    Δ₀ :: Float64 = 0.1;
    Δ_max :: Float64 = 0.5;

    # additional stopping criteria (mostly inspired by thoman)
    Δ_critical = 1e-3;   # max ub - lb / 10
    Δ_min = Δ_critical * 1e-3;
    stepsize_min = 1e-2 * Δ_critical;   # stop if Δ < Δ_critical & step_size < stepsize_min
    # NOTE thomann uses stepsize in image space due to PS scalarization

    iter_data :: Union{Nothing,IterData} = nothing; # Refercnce to an object storing the iteration information

    # assertions for parameter consistency
    #@assert 0 <= θ_pivot <= 1/θ_enlarge_1 "θ_pivot = $θ_pivot must be in range [0, $(1/θ_enlarge_1)]."
    @assert μ <= β "μ = $μ must be smaller than or equal to β = $β."
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
