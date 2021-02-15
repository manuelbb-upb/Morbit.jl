abstract type AbstractConfig end;
abstract type AbstractObjective <: Function end;

include("SurrogateModelInterface.jl");

include("objectives.jl")
###############  MULTIOBJECTIVE OPTIMIZATION PROBLEM DEFINITION ##############

@with_kw mutable struct MixedMOP
    #vector_of_objectives :: Vector{ Union{ ObjectiveFunction, VectorObjectiveFunction } } = [];
    vector_of_objectives :: Vector{ AbstractObjective } = [];
    n_objfs :: Int64 = 0;

    internal_sorting :: Vector{Int64} = [];
    reverse_sorting :: Vector{Int64} = [];
    non_exact_indices ::Vector{Int64} = [];     # indices of vector returned by `eval_all_objectives` corresponding to non-exact objectives

    x_0 :: RVec = Real[];
    lb :: Union{Nothing,RVec} = nothing;
    ub :: Union{Nothing,RVec} = nothing;
    is_constrained = begin
        !( isnothing(lb) || isnothing(ub) ) && 
        !( isempty(lb) || isempty(ub) )
    end

    #store functions and n_out for research & debugging 
    original_functions :: Vector{Tuple{F where F<:Function, Int}} = []; # TODO check if ‘F where …‘ is necessary 
end
Broadcast.broadcastable(m::MixedMOP) = Ref(m);

##################### INNER SETTINGS #########################################

# collectible data during iterations (used for plotting and analysis)
@with_kw mutable struct IterData
    # "global" data (used in each iteration)
    x :: RVec;  # current iteration site
    f_x :: RVec; # true objective values at current iterate,
    x_index :: Int64 = 0;
    Δ :: Real = 0;
    sites_db :: RVecArray = RVec[]; # array of all sites that have been evaluated (RBF or Lagrange)
    values_db :: RVecArray = RVec[]; # array of all true values computed so far

    model_meta :: Array{Any,1} = Any[]; # iteration dependent surrogate data

    # Arrays (1 entry per iteration)
    iterate_indices :: Vector{ Int64 } = [];
    trial_point_indices :: Vector{Int64} = [];
    stepsize_array :: Vector{R} = Real[];
    Δ_array :: RVec = Real[]  
    ω_array :: RVec = Real[]
    ρ_array :: RVec = Real[]
    num_crit_loops_array :: Vector{Int64} = [];
end
Broadcast.broadcastable(id :: IterData) = Ref(id);

@with_kw mutable struct AlgoConfig <: AbstractConfig

    n_vars ::Int64 = 0; # is reset during optimization
    n_objfs :: Int64 = 0; # total number of (scalarized) objectives

    problem :: Union{MixedMOP,Nothing} = nothing;

    max_iter :: Int64 = 500;
    count_nonlinear_iterations :: Bool = false; # include nonlinear iterations in 'max_iter'?
    max_evals :: Int64 = typemax(Int64);    # maxiumm number of expensive function evaluations

    descent_method :: Symbol = :steepest # :steepest, :cg, :ps (Pascoletti-Serafini) or :direct_search 
    ideal_point :: RVec = Real[];
    image_direction :: RVec = Real[];
    θ_ideal_point :: Real = Float16(1.5);

    all_objectives_descent :: Bool = false;  # compute ρ as the minimum of descent ratios for ALL objetives

    radius_update :: Symbol = :standard # :standard or :steplength

    # criticallity parameters
    μ :: Real = Float16(2e3);
    β :: Real = Float16(1e3);
    ε_crit :: Real = Float16(1e-3);
    max_critical_loops :: Int64 = 10;

    # User benchmark functions for stopping
    x_stop_function :: Union{F where F<:Function, Nothing} = nothing;
    # TODO other functions that depend on Δ, ρ or somesuch

    # acceptance parameters
    ν_success :: Real = Float16(0.4);
    ν_accept :: Real = Float16(0.0);
    # trust region update parameters
    γ_crit :: Real = Float16(0.5); # scaling factor for Δ in criticallity test
    γ_grow :: Real = Float16(2);
    γ_shrink :: Real = Float16(0.75);
    γ_shrink_much :: Real = Float16(0.501);

    Δ₀ :: Real = Float16(0.1);
    Δ_max :: Real = Float16(1.0);

    # additional stopping criteria (mostly inspired by thoman)
    Δ_critical :: Real = Float16(1e-4);
    Δ_min :: Real = Δ_critical * Float16(1e-3);
    stepsize_min :: Real = Δ_critical * Float16(1e-2);   # stop if Δ < Δ_critical & step_size < stepsize_min
    # NOTE thomann uses stepsize in image space due to PS scalarization

    use_eval_database :: Bool = true; # NOTE this was a quick hack!! don't use when employing different surrogates
    use_info_database :: Bool = true;
    iter_data :: Union{Nothing,IterData} = nothing; # Refercnce to an object storing the iteration information

    # assertions for parameter consistency
    #@assert 0 <= θ_pivot <= 1/θ_enlarge_1 "θ_pivot = $θ_pivot must be in range [0, $(1/θ_enlarge_1)]."
    @assert β <= μ "μ = $μ must be larger than or equal to β = $β."
    @assert Δ₀ <= Δ_max "Δ_max = $Δ_max is smaller than initial trust region radius Δ₀ = $Δ₀."
    @assert ν_accept <= ν_success "Acceptance parameters must be 0<= ν_accept <= ν_success."
    @assert 0 < γ_crit < 1 "Criticality reduction factor γ_crit must be in (0,1)."
    @assert 0 < γ_shrink <= 1 "Trust region reduction factor γ_shrink must be in (0,1]"
    @assert 1 <= γ_grow "Trust region grow factor γ_grow must be bigger than 1."
    @assert max_iter > 0 "Maximal number of iterations must be a positive integer."
    # TODO make sure Δ_max is bounded by a fraction of global boundaries (requires mutable struct?)

end
Broadcast.broadcastable(id :: AlgoConfig) = Ref(id);

