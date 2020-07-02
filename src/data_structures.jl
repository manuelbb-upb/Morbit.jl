export ParetoSet, ParetoFrontier
export MixedMOP, add_objective!

@with_kw struct ParetoSet
    n_vars :: Int64 = 0;
    coordinate_arrays :: Vector{ Vector{Float64} } = [];
end

function ParetoSet( matrix :: Array{Float64, 2}; points_as_columns = true )
    if points_as_columns
        n_vars = size(matrix, 2)
        arrays = collect( eachrow(matrix) )
    else
        n_vars = size( matrix, 1 )
        arrays = collect( eachcol(matrix) )
    end
    ParetoSet( n_vars, arrays )
end

function ParetoSet( data::Vararg{ Vector{Float64} , N} ) where{N}
    n_vars = length(data);
    coordinate_arrays = [ data... ]
    ParetoSet(n_vars, coordinate_arrays)
end

@with_kw struct ParetoFrontier
    n_objfs :: Int64 = 0;
    objective_arrays :: Vector{ Vector{Float64} } = [];
end

function ParetoFrontier( matrix :: Array{Float64, 2}; points_as_columns = true )
    if points_as_columns
        n_objf = size(matrix, 2)
        arrays = collect( eachrow(matrix) )
    else
        n_objf = size( matrix, 1 )
        arrays = collect( eachcol(matrix) )
    end
    ParetoFrontier( n_objf, arrays )
end

function ParetoFrontier( f :: T where{T <: Function}, pset :: ParetoSet )
    n_points = length( pset.coordinate_arrays[1] )
    evals = f.( [ [pset.coordinate_arrays[d][i] for d = 1 : pset.n_vars ] for i = 1 : n_points ] )
    eval_matrix = hcat( evals... )
    ParetoFrontier( eval_matrix )
end

@with_kw mutable struct TrainingData
    Y :: Array{Float64,2} = Matrix{Float64}(undef, 0,0);   # matrix of other sites, translated by -x
    Z :: Array{Float64,2} = Matrix{Float64}(undef, 0,0);;   # first column of orthogonal basis to improve non linear model
end

isempty( t :: TrainingData ) = isempty(t.Y) || isempty(t.Z);

@with_kw mutable struct ModelInfo
    center_index :: Int64 = 1;
    round1_indices :: Vector{Int64} = [];
    round2_indices :: Vector{Int64} = [];
    round3_indices :: Vector{Int64} = [];
    fully_linear :: Bool = true;
end

@with_kw mutable struct RBFModel
    training_sites :: Array{Array{Float64, 1},1} = [];
    training_values :: Array{Array{Float64, 1}, 1} = [];
    kernel :: String = "multiquadric";
    shape_parameter :: Float64 = 1.0;
    fully_linear :: Bool = false;
    polynomial_degree :: Int64 = -1;
    function_handle :: T where{T <: Function} = x -> 0.0;
    output_handles :: Union{Nothing,Array{Any,1}} = nothing;
    jacobian_handle :: T where{T <: Function} = x -> Array{Float64,2}();
    gradient_handles :: Union{Nothing,Array{Any,1}} = nothing;

    tdata :: TrainingData = TrainingData();
    model_info :: ModelInfo = ModelInfo();
    @assert polynomial_degree <= 1 "For now only polynomials with degree -1, 0, or 1 are allowed."
    #model_params :: Dict = Dict();
end

# collectible data during iterations (used for plotting and analysis)
@with_kw mutable struct IterData
    # "global" data (actually used during iteration)
    x :: Vector{Float64} = []  # current iteration site
    f_x :: Vector{Float64} = []  # true objective values at current iterate,
    Δ :: Float64 = 0.0;
    sites_db :: Vector{Vector{Float64}} = []; # array of all sites that have been evaluated.
    values_db :: Vector{Vector{Float64}} = []; # array of all true values computed so far

    # Arrays (1 entry per iteration)
    iterate_indices :: Vector{ Int64 } = [];
    model_info_array :: Vector{ModelInfo} = [];
    stepsize_array :: Vector{Float64} = [];  # a bit redundant, since iterates are given
    Δ_array :: Vector{Float64} = [];
    ω_array :: Vector{Float64} = [];
    ρ_array :: Vector{Float64} = [];
    num_crit_loops_array :: Vector{Int64} = [];
end

@with_kw mutable struct AlgoConfig
    verbosity :: Bool = true;

    n_vars ::Int64 = 0; # is reset during optimization
    n_exp :: Int64 = 0; # number of expensive objectives
    n_cheap :: Int64 = 0; # number of cheap objectives

    ε_bounds = 0.0;   # minimum distance to boundaries if problem is constraint, needed for functions undefined outside bounds

    f :: Union{Function, Nothing} = nothing;    # reset during algorithm initilization

    rbf_kernel :: String = "multiquadric";
    rbf_poly_deg :: Int64 = 1;
    rbf_shape_parameter :: T where T<:Function = Δ -> 1;
    max_model_points ::Int64 = 2*n_vars^2 + 1;  # maximum number of points to be included in the construction of 1 model

    max_iter :: Float64 = 1000;
    max_evals :: Union{Int64,Float64} = Inf;    # maxiumm number of expensive function evaluations

    descent_method :: Symbol = :steepest # :steepest or :direct_search ( TODO implement local Pascoletti-Serafini )

    all_objectives_descent :: Bool = false;  # compute ρ as the minimum of descent ratios for ALL objetives

    # criticallity parameters
    μ :: Float64 = 3e3;
    β :: Float64 = 5e3;
    ε_crit :: Float64 = 1e-3;
    max_critical_loops :: Int64 = 30;

    # acceptance parameters
    ν_success :: Float64 = 0.4;
    ν_accept :: Float64 = 0.0;
    # trust region update parameters
    γ_crit :: Float64 = 0.5; # scaling factor for Δ in criticallity test
    γ_grow :: Float64 = 2;
    γ_shrink :: Float64 = 0.9;
    γ_shrink_much :: Float64 = 0.5;

    Δ₀ :: Float64 = 0.1;
    Δ_max :: Float64 = 1;

    θ_enlarge_1 :: Float64 = 10;        # as in ORBIT according to Wild
    θ_enlarge_2 :: Float64 = 0.0;     # is reset during optimization
    θ_pivot :: Float64 = 1e-3;# 1 / θ_enlarge_1;
    θ_pivot_cholesky :: Float64 = 1e-7;

    # additional stopping criteria (mostly inspired by thoman)
    Δ_critical = 1e-3;   # max ub - lb / 10
    Δ_min = Δ_critical / 10.0;
    stepsize_min = 1e-6 * Δ_critical;   # stop if Δ < Δ_critical & step_size < stepsize_min
    # NOTE thomann uses stepsize in image space due to PS scalarization

    iter_data :: Union{Nothing,IterData} = nothing;

    # assertions for parameter consistency
    @assert 0 <= θ_pivot <= 1/θ_enlarge_1 "θ_pivot = $θ_pivot must be in range [0, $(1/θ_enlarge_1)]."
    @assert μ <= β "μ = $μ must be smaller than or equal to β = $β."
    @assert Δ₀ <= Δ_max "Δ_max = $Δ_max is smaller than initial trust region radius Δ₀ = $Δ₀."
    @assert 0 <= ν_accept <= ν_success "Acceptance parameters must be 0<= ν_accept <= ν_success."
    @assert 0 < γ_crit < 1 "Criticality reduction factor γ_crit must be in (0,1)."
    @assert 0 < γ_shrink < 1 "Trust region reduction factor γ_shrink must be in (0,1)"
    @assert 1 < γ_grow "Trust region grow factor γ_grow must be bigger than 1."
    @assert max_iter > 0 "Maximal number of iterations must be a positive integer."
    @assert rbf_kernel ∈ ["exp", "multiquadric", "cubic", "thin_plate_spline"] "Kernel '$rbf_kernel' not supported yet."
    # TODO make sure Δ_max is bounded by a fraction of global boundaries (requires mutable struct?)

end

# Outer Constructor to obtain default configuration adapted for n_vars input variables.
AlgoConfig( n_vars ) = AlgoConfig( θ_enlarge_2 = max(sqrt(n_vars), 4) )

# wrapper for a multiobjective optimization problem.
# does not provide many benefits as for now, but will be usefull for constrained problems
@with_kw struct MOP
    f::Function     # objective function, vector valued
    x_0 :: Array{Float64,1} = [];
    lb :: Array{Float64, 1} = [];   # lower variable boundaries, empty = -Inf for each variable
    ub :: Array{Float64, 1} = [];   # upper variable boundaries, empty = Inf for each variable
    @assert isempty(lb) & isempty(ub) || all( isinf.(lb) .& isinf.(ub) ) || all( isfinite.(lb) .& isfinite.(ub) ) "Problem must either be unconstraint or fully box constrained."

    # TODO enable passing of gradients so that no forwarddiff is needed
    # TODO define constraints
end

@with_kw struct HeterogenousMOP
    f_expensive :: Function = x -> Array{Float64,1}();
    f_cheap :: Function = x -> Array{Float64,1}();            # don't build surrogate models for this objective function (vector-valued)
    x_0 :: Array{Float64,1} = [];
    lb :: Array{Float64, 1} = [];   # lower variable boundaries, empty = -Inf for each variable
    ub :: Array{Float64, 1} = [];

    @assert isempty(lb) & isempty(ub) || all( isinf.(lb) .& isinf.(ub) ) || all( isfinite.(lb) .& isfinite.(ub) ) "Problem must either be unconstraint or fully box constrained."
end

@with_kw struct MixedMOP
    vector_of_funcs :: Vector{Function} = [];
    vector_of_cheap_gradients :: Vector{ Function } = [];   # TODO Implement
    expensive_indices :: Vector{Int64} = [];
    cheap_indices :: Vector{Int64} = [];
    cheap_has_gradient_flags :: Vector{Bool} = [];
    x_0 :: Vector{Float64} = [];
    lb :: Vector{Float64} = [];
    ub :: Vector{Float64} = [];
    @assert isempty(lb) & isempty(ub) || all( isinf.(lb) .& isinf.(ub) ) || all( isfinite.(lb) .& isfinite.(ub) ) "Problem must either be unconstraint or fully box constrained."
end

function add_objective!( problem :: MixedMOP, func :: T where{T <: Function}, type ::Symbol = :expensive )
    push!( problem.vector_of_funcs, func );
    if type == :expensive
        push!(problem.expensive_indices, length(problem.vector_of_funcs) )
        println("Added an expensive objective with internal index $(length(problem.vector_of_funcs)).")
    else
        push!(problem.cheap_indices, length( problem.vector_of_funcs) )
        push!(problem.cheap_has_gradient_flags, false)
        println("Added an cheap objective (autodiff enabled) with internal index $(length(problem.vector_of_funcs)).")
    end
end

function add_objective!( problem :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})
    push!( problem.vector_of_funcs, func );
    push!( problem.vector_of_cheap_gradients, func );
    # only cheap functions have gradient
    push!(cheap_indices, length( problem.vector_of_funcs) )
    push!(problem.cheap_has_gradient_flags, true)
    println("Added an cheap objective (and its gradient) with internal index $(length(problem.vector_of_funcs)).")
end
