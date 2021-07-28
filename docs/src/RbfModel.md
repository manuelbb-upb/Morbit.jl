```@meta
EditURL = "<unknown>/src/RbfModel.jl"
```

# Radial Basis Function Surrogate Models

## Intro and Prerequisites

We want to offer radial basis function (RBF) surrogate models (implementing the `SurrogateModel` interface).
To this end, we leverage the package [`RadialBasisFunctionModels.jl`](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl).
A scalar RBF model consists of a ``n``-variate Polynomial and linear combination of shifted radial kernels.
For more information, see [the documentation of `RadialBasisFunctionModels.jl`](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/stable).

````julia
import RadialBasisFunctionModels as RBF
````

The polyonmials will have a degree of at most 1.
To construct “good” linear polynomials, we need to make sure to have construction sites,
that span the decision space well. Such a set of construction sites is called Λ-poised or
sufficiently affinely independent.
The file `AffinelyIndependentPoints` implements some helpers to find suitable points as described by
Wild et. al.[^wild_diss]

````julia
include("AffinelyIndependentPoints.jl")
````

We also need this little helper, to restrict ourselves to points from a current box trust region only:

````julia
"Return indices of results in `db` that lie in a box with corners `lb` and `ub`."
function results_in_box_indices(db, lb, ub, exclude_indices = Int[] )
	return [ id for id = eachindex(db) if
		id ∉ exclude_indices && all(lb .<= get_site(db,id) .<= ub ) ]
end
````

## Surrogate Interface implementations

The model used in our algorithm simply wraps an interpolation model from the `RBF` package.

````julia
@with_kw struct RbfModel{R <: RBF.RBFInterpolationModel } <: SurrogateModel
	model :: R

	# indicator: is the model fully linear?
	fully_linear :: Bool = false
end

fully_linear( rbf :: RbfModel ) :: Bool = rbf.fully_linear
````

We offer a large range of configuration parameters in the `RBFConfig`, which implements
a `SurrogateConfig`.

````julia
"""
    RbfConfig(; kwarg1 = val1, … )

Configuration type for local RBF surrogate models.

To choose a kernel, use the kwarg `kernel` and a value of either
`:cubic` (default), `:inv_multiquadric`, `:multiquadric`, `:gaussian` or `:thin_plate_spline`.
The kwarg `shape_parameter` takes a constant number or a string
that defines a calculation on `Δ`, e.g, "Δ/10".
Note, that `shape_parameter` has a different meaning for the different kernels.
For ``:gaussian, :inv_multiquadric, :multiquadric` it actually is a floating point shape_parameter.
For :cubic it is the (odd) integer exponent and for `thin_plate_spline` it is an integer exponent as well.

To see other configuration parameters use `fieldnames(Morbit.RbfConfig)`.
They have individual docstrings attached.
"""
@with_kw mutable struct RbfConfig <: SurrogateConfig
    "(default `:cubic`) RBF kernel (Symbol), either `:cubic`, `:multiquadric`, `:exp` or `:thin_plate_spline`."
    kernel :: Symbol = :cubic

	"(default `1`) RBF shape paremeter, either a number or a string containing `Δ`."
    shape_parameter :: Union{String, Float64} = 1

	"(default `1`) Degree of polynomial attached to RBF. `-1` means no polynomial."
    polynomial_degree :: Int64 = 1;

    "(default `2`) Local enlargment factor of trust region for sampling."
    θ_enlarge_1 :: Float64 = 2

	"(default `5`) Maximum enlargment factor of maximum trust region for sampling."
    θ_enlarge_2 :: Float64 = 2  # reset

	"(default `1/(2*θ_enlarge_1)` Sampling parameter to generate Λ-poised set. The higher, the more poised."
    θ_pivot :: Float64 = 1 / (2 * θ_enlarge_1)

	"(default `1e-7`) Parameter for 2nd sampling algorithm to ensure boundedness of Cholesky factors."
    θ_pivot_cholesky :: Float64 = 1e-7

    "(default `false`) Require models to be fully linear in each iteration."
    require_linear :: Bool = false

    "(default `-1`) Maximum number of training sites. `-1` is reset to `2n+1`."
    max_model_points :: Int64 = -1 # is probably reset in the algorithm
    "(default `false`) Sample new sites to always use the maximum number of points."
    use_max_points :: Bool = false

    "(default `:orthogonal`) Algorithm to use for finding affinely independent set."
    sampling_algorithm :: Symbol = :orthogonal # :orthogonal or :monte_carlo

	"(default `:standard_rand`) Algorithm to use if additional points are required."
    sampling_algorithm2 :: Symbol = :standard_rand

    "(default `typemax(Int64)`) Maximum number of objective evaluations."
    max_evals :: Int64 = typemax(Int64)

    @assert sampling_algorithm ∈ [:orthogonal, :monte_carlo] "Sampling algorithm must be either `:orthogonal` or `:monte_carlo`."
    @assert kernel ∈ Symbol.(["exp", "inv_multiquadric", "multiquadric", "cubic", "thin_plate_spline"]) "Kernel '$kernel' not supported yet."
	# Some sanity checks for the shape parameters
    @assert kernel != :thin_plate_spline || ( shape_parameter % 1 == 0 && shape_parameter >= 1 ) "Invalid shape_parameter for :thin_plate_spline."
	@assert kernel != :cubic || ( shape_parameter % 1 == 0 && shape_parameter % 2 == 1 ) "Invalid shape_parameter for :cubic."
	@assert shape_parameter > 0 "Shape parameter must be strictly positive."
    # @assert θ_enlarge_1 >=1 && θ_enlarge_2 >=1 "θ's must be >= 1."
end
````

The required method implementations are straightforward.
Note, thate we allow the models to be combined to vector functions if they
share the same configuration to avoid redundant efforts whilst constructing models.

````julia
max_evals( cfg :: RbfConfig ) :: Int = cfg.max_evals
combinable( cfg :: RbfConfig ) :: Bool = true
combine(cfg1 :: RbfConfig, :: RbfConfig) :: RbfConfig = cfg1
````

To allow the user to set the shape parameter relative to the current trust region radius
using a verbose string, we need this little helper function, which evaluates the string.

````julia
function parse_shape_param_string( Δ :: F, expr_str) :: F where F
    ex = Meta.parse(expr_str)
    return @eval begin
        let Δ=$Δ
            $ex
        end
    end
end
````

The `RbfMeta` is used to store construction and update data for the models.
To be specific, we have several inidices lists that store database indices
of (potentially unevaluated) results that are later used for fitting the model.

````julia
@with_kw mutable struct RbfMeta{F<:AbstractFloat} <: SurrogateMeta
    center_index :: Int = -1
    round1_indices :: Vector{Int} = []
    round2_indices :: Vector{Int} = []
    round3_indices :: Vector{Int} = []
    round4_indices :: Vector{Int} = []
    fully_linear :: Bool = false
	improving_directions :: Vector{F} = []
end
````

A little helper to retrieve all those indices:

````julia
function _collect_indices( meta :: RbfMeta; include_x = true ) :: Vector{Int}
	return [
		include_x ? meta.center_index : [];
		meta.round1_indices;
		meta.round2_indices;
		meta.round3_indices;
		meta.round4_indices
	]
end
````

## Construction

The initial `prepare_init_model` function should return a meta object that can be used
to build an initial surrogate model.
We delegate the work to `prepare_update_model`.

````julia
function prepare_init_model( cfg :: RbfConfig, objf :: AbstractObjective, mop :: AbstractMOP,
	id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig)

	meta = RbfMeta{F}(; center_index = get_x_index( id ) )
	return prepare_update_model(nothing, objf, meta, mop, id, db, ac; ensure_fully_linear = true)
end
````

Usually, `prepare_update_model` would only accept a model as its first argument.
Because of the trick from above, we actually allow `nothing`, too.

````julia
function prepare_update_model( mod :: Union{Nothing, RbfModel}, objf, meta :: RbfMeta, mop, iter_data, db, algo_config; ensure_fully_linear = false)

	# Retrieve current iteration information and some meta data.
	Δ = get_Δ(iter_data)
	Δ_max = Δᵘ(algo_config)
	x = get_x(iter_data)
	x_index = get_x_index(iter_data)
	cfg = model_cfg( objf )

	F = eltype(x)
	n_vars = length(x)

	# By default, assume that our model is not fully linear
	fully_linear = false

	# First round of sampling:
	### Try to find points in slightly enlarged trust region
	Δ_1 = F.(cfg.θ_enlarge_1 * Δ)
	lb_1, ub_1 = local_bounds( mop, x, Δ_1 )
	piv_val_1 = F.(cfg.θ_pivot * Δ_1) # threshold value for acceptance in filter

	### only consider points from within current trust region …
	candidate_indices_1 = results_in_box_indices( db, lb_1, ub_1, [x_index],)

	### … and filter them to obtain affinely independent points.
	filter = AffinelyIndependentPointFilter(;
		x_0 = x,
		seeds = get_site.(db, candidate_indices_1),
		return_indices = true,
		pivot_val = piv_val_1
	)

	### Store indices in meta data object:
	filtered_indices_1 = candidate_indices_1[ collect( filter ) ]
	empty!(meta.round1_indices)
	append!(meta.round1_indices, filtered_indices_1)

	# Second round of sampling:
	### If there are not enough sites to have a fully linear model …
	### … try to at least find more sites in maximum allowed radius
	n_missing = n_vars - length( filtered_indices_1 )

	if n_missing == 0 || ensure_fully_linear
		### (if `ensure_fully_linear == True`, we skip this step and go to round 3)
		fully_linear = true
		filter_2 = filter
		empty!(meta.round2_indices)
	else
		### `Δ_2` is the maximum allowed trust region radius.
		Δ_2 = F.(cfg.θ_enlarge_2 * Δ_max )
		lb_2, ub_2 = local_bounds( mop, x, Δ_2 )
		piv_val_2 = piv_val_1 # the pivot value stays the same

		### as before, only consider points in box of radius `Δ_2`, but ignore `x` and the previous points
		candidate_indices_2 = results_in_box_indices( db, lb_2, ub_2, [x_index; candidate_indices_1])

		filter_2 = AffinelyIndependentPointFilter(;
			x_0 = x,
			seeds = get_site.(db, candidate_indices_2),
			Y = filter.Y,	# pass prior matrices, so that new points are projected onto span of Z
			Z = filter.Z,
			n = n_missing,
			return_indices = true,
			pivot_val = piv_val_2
		)

		filtered_indices_2 = candidate_indices_2[ collect(filter_2) ]

		### Store indices
		empty!(meta.round2_indices)
		append!(meta.round2_indices, filtered_indices_2)
	end

	# Round 3:
	### If we still don't have enough sites, generate them
	### along model improving directions (from first round of sampling)
	empty!(meta.improving_directions)
	append!(meta.improving_directions, reverse(collect(Vector{F}, eachcol(filter.Z))) )

	n_missing -= length(meta.round2_indices)
	if n_missing > 0

		### take into consideration the maximum number of evaluations allowed.
		max_new = min( max_evals(algo_config), max_evals(cfg) ) - 1 - num_evals( objf )
		n_new = min(n_missing, max_new)

		new_indices = Int[]
		while !isempty(meta.improving_directions) && length( new_indices ) < n_new
			dir = popfirst!( meta.improving_directions )
			len = intersect_bounds( x, dir, lb_1, ub_2; return_vals = :absmax )
			offset = len .* dir
			if norm( offset, Inf ) > piv_val_1
				new_id = new_result!( db, x .+ offset, F[] )
				push!(new_indices, new_id)
			end
		end

		### If round 2 did not yield any new points, the model will be fully linear now.
		if length(meta.round2_indices) == 0
			fully_linear = true
		end

		empty!(meta.round3_indices)
		append!(meta.round3_indices, new_indices)
	end


	return meta
end
````

An improvement step consists of adding a new site to the database, along an improving direction:

````julia
function prepare_improve_model( mod :: Union{Nothing, RbfModel}, objf, meta :: RbfMeta,
	mop, iter_data, db, algo_config; kwargs... )
	if !meta.fully_linear
		if isempty(meta.improving_directions)
			@warn "RBF model is not fully linear, but there are no improving directions."
		else
			cfg = model_cfg(objf)
			piv_val_1 = F.(get_Δ(iter_data) * cfg.θ_enlarge_1 * cfg.θ_pivot)

			success = false
			dir = popfirst!( meta.improving_directions )
			len = intersect_bounds( x, dir, lb_1, ub_2; return_vals = :absmax )
			offset = len .* dir
			if norm( offset, Inf ) > piv_val_1
				new_id = new_result!( db, x .+ offset, F[] )
				push!(new_indices, new_id)
				success = true
			end

			if isempty( meta.improving_directions ) && success
				meta.fully_linear = true
			end
		end
	end
	return meta
end
````

Now, in the 2-phase construction process, first all `prepare_` functions are called for all surrogate models.
Then, the unevaluated results are evaluated and we can proceed with the model building.
As before, `_init_model` simply delegates work to `update_model`.

````julia
function _init_model( cfg :: RbfConfig, objf :: AbstractObjective, mop :: AbstractMOP,
	iter_data :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig, meta :: RbfMeta; kwargs... )
	return update_model( nothing, objf, meta, mop, iter_data, db, ac; kwargs... )
end
````

In contrast to the old RBF mechanism, the models in `RadialBasisFunctionModels` sometimes
accept 2 parameters for the kernel. We use this little helper, to get defaults from the shape parameter.
Note, that sanity check are performed in the RbfConfig constructor.

````julia
function _get_kernel_params( sp, kernel_name :: Symbol )
	if kernel_name == :gaussian
		return sp
	elseif kernel_name == :inv_multiquadric
		return (sp, 1//2)
	elseif kernel_name == :multiquadric
		return (sp, 1//2)
	elseif kernel_name == :cubic
		return Int(sp)
	elseif kernel_name == :thin_plate_spline
		return Int(sp)
	else
		return sp
	end
end

function update_model( mod::Union{Nothing,RbfModel}, objf:: AbstractObjective, meta :: RbfMeta,
	mop :: AbstractMOP, id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; kwargs... )

	Δ = get_Δ(iter_data)
	cfg = model_cfg(objf)

	x = get_x(iter_data)
	F = eltype(x)

	# get the kernel parameter and the kernel parameters
	sp = if cfg.shape_parameter isa String
		parse_shape_param_string( Δ, sp )
	else
		F(cfg.shape_parameter)
	end
	kernel_params = _get_kernel_params( sp, cfg.kernel_name )

	# get the training data from `meta` and the database `db`
	training_indices = _collect_indices( meta )
	training_results = get_results( db, training_indices )
	training_sites = get_site.( training_results )
	oi = output_indices( objf, mop)	# only consider the objective output indices
	training_values = [ v[oi] for v in get_value.( traning_results ) ]

	inner_model = RBF.RBFInterpolationModel( training_sites, training_values, cfg.kernel_name, kernel_params )

	return RbfModel( inner_model, meta.fully_linear ), meta

end
````

The improvement function also simply cals the update function:

````julia
function improve_model( mod::Union{Nothing,RbfModel}, objf:: AbstractObjective, meta :: RbfMeta,
	mop :: AbstractMOP, id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; kwargs... )
	return update_model( mod, objf, meta, mop, id, db, ac; kwargs... )
end
````

## Evaluation

All the work is done by the inner model :)

````julia
"Evaluate `mod::RbfModel` at scaled site `x̂`."
function eval_models( mod :: RbfModel, x̂ :: Vec)
	return mod.model( x̂ )
end

"Evaluate output `ℓ` of `mod::RbfModel` at scaled site `x̂`."
function eval_models( mod :: RbfModel, x̂ :: Vec, ℓ :: Int)
	return mod.model( x̂, ℓ)
end

@doc "Gradient vector of output `ℓ` of `mod` at scaled site `x̂`."
function get_gradient( mod :: RbfModel, x̂ :: Vec, ℓ :: Int64)
    return RBF.grad( mod.model, x̂, ℓ )
end

@doc "Jacobian Matrix of ExactModel `em` at scaled site `x̂`."
function get_jacobian( mod :: RbfModel, x̂ :: Vec )
    return RBF.jac( mod.model, x̂ )
end
````

[^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Stefan M. Wild, 2009

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

