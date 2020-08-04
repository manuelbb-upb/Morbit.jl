export ParetoSet, ParetoFrontier
export MixedMOP, add_objective!, add_vector_objective!

import Base: push!

# ##### Objective structs #####

# a custom wrapper for objective functions …
@with_kw struct ObjectiveFunction <: Function
    function_handle :: Union{T, Nothing} where{T<:Function} = nothing
end

# … to allow for batch evaluation outside of julia or to exploit parallelized objective functions
@with_kw struct BatchObjectiveFunction <: Function
    function_handle :: Union{T, Nothing} where{T<:Function} = nothing
end

# for 'usual' function evaluation calls (args = single vector), simply delegate to function_handle
function (objf::Union{ObjectiveFunction, BatchObjectiveFunction})(args...)
    objf.function_handle(args...)
end

# overload broadcasting for BatchObjectiveFunction who are assumed to handle arrays themselves
function broadcasted( objf::BatchObjectiveFunction, args... )
    objf.function_handle( args... )
end

# ##### Convenience structs for plotting #####

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


# ##### MOP structs #####

# wrapper for a multiobjective optimization problem.
# does not provide many benefits as for now, but will be usefull for constrained problems
@with_kw struct MOP
    f::Function     # objective function, vector valued
    x_0 :: Array{Float64,1} = [];
    lb :: Array{Float64, 1} = [];   # lower variable boundaries, empty = -Inf for each variable
    ub :: Array{Float64, 1} = [];   # upper variable boundaries, empty = Inf for each variable
    @assert isempty(lb) & isempty(ub) || all( isinf.(lb) .& isinf.(ub) ) || all( isfinite.(lb) .& isfinite.(ub) ) "Problem must either be unconstraint or fully box constrained."
end

@with_kw mutable struct MixedMOP
    vector_of_expensive_funcs :: Vector{Function} = [];
    vector_of_cheap_funcs :: Vector{Function} = [];

    n_exp :: Int64 = 0;
    n_cheap :: Int64 = 0;

    internal_sorting :: Vector{Int64} = [];

    vector_of_gradient_funcs :: Vector{Function} = [];

    x_0 :: Vector{Float64} = [];
    lb :: Vector{Float64} = [];
    ub :: Vector{Float64} = [];
    is_constrained = !( isempty(lb) || isempty( ub ) ) && ( all( isfinite.(lb) .& isfinite.(ub) ) )
    @assert isempty(lb) & isempty(ub) || all( isinf.(lb) .& isinf.(ub) ) || all( isfinite.(lb) .& isfinite.(ub) ) "Problem must either be unconstraint or fully box constrained."
end

Broadcast.broadcastable(m::MixedMOP) = Ref(m);

@doc """
    add_objective!( problem :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )

Add scalar-valued objective function `func` to `problem` structure.
`func` must take an `Vector{Float64}` as its (first) argument, i.e. represent a function ``f: ℝ^n → ℝ``.
`type` must either be `:expensive` or `:cheap` to determine whether the function is replaced by a surrogate model or not.

If `type` is `:cheap` and `func` takes 1 argument only then its gradient is calculated by ForwardDiff.
A cheap function `func` with custom gradient function `grad` (representing ``∇f : ℝ^n → ℝ^n``) is added by

    add_objective!(problem, func, grad)

The optional argument `n_out` allows for the specification of vector-valued objective functions.
This is mainly meant to be used for *expensive* functions that are in some sense inter-dependent.

The flag `can_batch` defaults to false so that the objective function is simply looped over a bunch of arguments if required.
If `can_batch == true` then the objective function must be able to return an array of results when provided an array of input vectors
(whilst still returning a single result, not a singleton array containing the result, for a single input vector).

# Examples
```jldoctest
# Define 2 scalar objective functions and a MOP ℝ^2 → ℝ^2

f1(x) =  x[1]^2 + x[2]

f2(x) = exp(sum(x))
∇f2(x) = exp(sum(x)) .* ones(2);

mop = MixedMOP()
add_objective!(mop, f1, :cheap)     # gradient will be calculated using ForwardDiff
add_objective!(mop, f2, ∇f2 )       # gradient is provided
```
"""
function add_objective!( problem :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )
    n_objfs = problem.n_exp + problem.n_cheap;
    func_indices = collect((n_objfs + 1) : (n_objfs + n_out ));

    if can_batch == false
        wrapped_func = ObjectiveFunction( func )
    else
        wrapped_func = BatchObjectiveFunction( func )
    end
    if type == :expensive
        push!( problem.vector_of_expensive_funcs, wrapped_func);
        insert_position = problem.n_exp + 1;
        for ℓ = 1 : n_out
            insert!( problem.internal_sorting, insert_position, func_indices[ℓ] )
            insert_position += 1
        end
        problem.n_exp += n_out;
        println("Added $n_out expensive objective(s) with indices $func_indices.")
    else
        push!( problem.vector_of_cheap_funcs, wrapped_func );
        push!( problem.internal_sorting, func_indices... )
        for ℓ = 1 : n_out
            grad_fn = function (x :: Vector{Float64} )
                gradient(X -> wrapped_func(X)[ℓ], x)  # for n_out > 1 this is not super effective but to be honest it is not meant to be
            end
            push!( problem.vector_of_gradient_funcs, grad_fn)
        end
        problem.n_cheap += n_out;
        println("Added $n_out cheap (autodiff) objective(s) with indices $func_indices.")
    end
end

@doc """
    add_objective!( problem :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})

Add scalar-valued objective function `func` and its vector-valued gradient `grad` to `problem` struture.
"""
function add_objective!( problem :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})
    push!( problem.vector_of_cheap_funcs, func);
    push!( problem.vector_of_gradient_funcs, grad);
    push!( problem.internal_sorting, problem.n_exp + problem.n_cheap + 1 );
    problem.n_cheap += 1;
    println("Added an cheap objective (and its gradient) with internal index $(problem.n_exp + problem.n_cheap).")
end
# =====================================================================================


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

# collectible data during iterations (used for plotting and analysis)
@with_kw mutable struct IterData
    # "global" data (actually used during iteration)
    x :: Vector{Float64} = []  # current iteration site
    f_x :: Vector{Float64} = []  # true objective values at current iterate,
    Δ :: Float64 = 0.0;
    sites_db :: Vector{Vector{Float64}} = []; # array of all sites that have been evaluated.
    values_db :: Vector{Vector{Float64}} = []; # array of all true values computed so far

    min_value :: Vector{Float64} = isempty(values_db) ? [] : vec(minimum( hcat( values_db... ), dims = 2 ));
    max_value :: Vector{Float64} = isempty(values_db) ? [] : vec(maximum( hcat( values_db... ), dims = 2 ));
    update_extrema :: Bool = false;

    # Arrays (1 entry per iteration)
    iterate_indices :: Vector{ Int64 } = [];
    model_info_array :: Vector{ModelInfo} = [];
    stepsize_array :: Vector{Float64} = [];  # a bit redundant, since iterates are given
    Δ_array :: Vector{Float64} = [];
    ω_array :: Vector{Float64} = [];
    ρ_array :: Vector{Float64} = [];
    num_crit_loops_array :: Vector{Int64} = [];
end

function push!( id :: IterData, new_vals... )
    if !isempty(new_vals)
        push!(id.values_db, new_vals... )
        if id.update_extrema
            if isempty(id.min_value)
                id.min_value = isempty(id.values_db) ? [] : vec(minimum( hcat( id.values_db... ), dims = 2 ));
            else
                new_vals_min = vec(minimum( hcat( new_vals... ), dims = 2 ));
                id.min_value = vec(minimum( hcat( id.min_value, new_vals_min ), dims = 2 ));
            end
            if isempty(id.max_value)
                id.max_value = isempty(id.values_db) ? [] : vec(maximum( hcat( id.values_db... ), dims = 2 ));
            else
                new_vals_max = vec(maximum( hcat( new_vals... ), dims = 2 ));
                id.max_value = vec(maximum( hcat( id.max_value, new_vals_max ), dims = 2 ));
            end
        end
    end
end

@with_kw mutable struct AlgoConfig
    verbosity :: Bool = true;

    n_vars ::Int64 = 0; # is reset during optimization
    n_exp :: Int64 = 0; # number of expensive objectives
    n_cheap :: Int64 = 0; # number of cheap objectives

    #ε_bounds = 0.0;   # minimum distance to boundaries if problem is constraint, needed for functions undefined outside bounds

    problem :: Union{MixedMOP,Nothing} = nothing;
    #f :: Union{Function, Nothing} = nothing;    # reset during algorithm initilization
    #index_permutation :: Vector{Int64} = [];    # used to interally reorder ideal point and image direction for MixedMOP

    rbf_kernel :: Symbol = :multiquadric;
    rbf_poly_deg :: Int64 = 1;
    rbf_shape_parameter :: T where T<:Function = Δ -> 1;
    max_model_points ::Int64 = 2*n_vars^2 + 1;  # maximum number of points to be included in the construction of 1 model

    max_iter :: Int64 = 1000;
    max_evals :: Union{Int64,Float64} = Inf;    # maxiumm number of expensive function evaluations

    descent_method :: Symbol = :steepest # :steepest or :direct_search ( TODO implement local Pascoletti-Serafini )
    ideal_point :: Vector{Float64} = [];
    image_direction :: Vector{Float64} = [];

    scale_values :: Bool = false;    # scale_values internally
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

    Δ₀ :: Float64 = 0.4;
    Δ_max :: Float64 = 1;

    θ_enlarge_1 :: Float64 = 4.0;        # as in ORBIT according to Wild
    θ_enlarge_2 :: Float64 = 0.0;     # is probably reset during optimization
    θ_pivot :: Float64 = 1 / (2 * θ_enlarge_1);
    θ_pivot_cholesky :: Float64 = 1e-7;

    # additional stopping criteria (mostly inspired by thoman)
    Δ_critical = 1e-3;   # max ub - lb / 10
    Δ_min = Δ_critical * 1e-3;
    stepsize_min = 1e-2 * Δ_critical;   # stop if Δ < Δ_critical & step_size < stepsize_min
    # NOTE thomann uses stepsize in image space due to PS scalarization

    iter_data :: Union{Nothing,IterData} = nothing;

    # assertions for parameter consistency
    @assert 0 <= θ_pivot <= 1/θ_enlarge_1 "θ_pivot = $θ_pivot must be in range [0, $(1/θ_enlarge_1)]."
    @assert μ <= β "μ = $μ must be smaller than or equal to β = $β."
    @assert Δ₀ <= Δ_max "Δ_max = $Δ_max is smaller than initial trust region radius Δ₀ = $Δ₀."
    @assert 0 <= ν_accept <= ν_success "Acceptance parameters must be 0<= ν_accept <= ν_success."
    @assert 0 < γ_crit < 1 "Criticality reduction factor γ_crit must be in (0,1)."
    @assert 0 < γ_shrink <= 1 "Trust region reduction factor γ_shrink must be in (0,1]"
    @assert 1 < γ_grow "Trust region grow factor γ_grow must be bigger than 1."
    @assert max_iter > 0 "Maximal number of iterations must be a positive integer."
    @assert rbf_kernel ∈ Symbol.(["exp", "multiquadric", "cubic", "thin_plate_spline"]) "Kernel '$rbf_kernel' not supported yet."
    # TODO make sure Δ_max is bounded by a fraction of global boundaries (requires mutable struct?)

end

# Outer Constructor to obtain default configuration adapted for n_vars input variables.
AlgoConfig( n_vars ) = AlgoConfig( θ_enlarge_2 = max(sqrt(n_vars), 4) )
