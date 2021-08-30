# # Radial Basis Function Surrogate Models

# ## Intro and Prerequisites

# We want to offer radial basis function (RBF) surrogate models (implementing the `SurrogateModel` interface).
# To this end, we leverage the package [`RadialBasisFunctionModels.jl`](https://github.com/manuelbb-upb/RadialBasisFunctionModels.jl).
# A scalar RBF model consists of a ``n``-variate Polynomial and linear combination of shifted radial kernels.
# For more information, see [the documentation of `RadialBasisFunctionModels.jl`](https://manuelbb-upb.github.io/RadialBasisFunctionModels.jl/stable).

import RadialBasisFunctionModels as RBF
using LinearAlgebra: qr, Hermitian, cholesky, inv, I, givens, diag

# The polyonmials will have a degree of at most 1. 
# To construct “good” linear polynomials, we need to make sure to have construction sites,
# that span the decision space well. Such a set of construction sites is called Λ-poised or 
# sufficiently affinely independent.
# The file `AffinelyIndependentPoints` implements some helpers to find suitable points as described by 
# Wild et. al.[^wild_diss]

include("AffinelyIndependentPoints.jl")

# ## Surrogate Interface Implementations

# The model used in our algorithm simply wraps an interpolation model from the `RBF` package.
@with_kw struct RbfModel{R} <: SurrogateModel
	model :: R

	## indicator: is the model fully linear?
	fully_linear :: Bool = false
end

fully_linear( rbf :: RbfModel ) :: Bool = rbf.fully_linear

# We offer a large range of configuration parameters in the `RBFConfig`, which implements 
# a `SurrogateConfig`.
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
Use `NaN` for defaults.

To see other configuration parameters use `fieldnames(Morbit.RbfConfig)`.
They have individual docstrings attached.
"""
@with_kw mutable struct RbfConfig <: SurrogateConfig
    "(default `:cubic`) RBF kernel (Symbol), either `:cubic`, `:multiquadric`, `:exp` or `:thin_plate_spline`."
    kernel :: Symbol = :cubic
    
	"(default `1`) RBF shape paremeter, either a number or a string containing `Δ`."
    shape_parameter :: Union{String, Float64} = NaN
    
	"(default `1`) Degree of polynomial attached to RBF. `-1` means no polynomial."
    polynomial_degree :: Int64 = 1;

    "(default `2`) Local enlargment factor of trust region for sampling."
    θ_enlarge_1 :: Float64 = 2
    
	"(default `5`) Maximum enlargment factor of maximum trust region for sampling."
    θ_enlarge_2 :: Float64 = 2
    
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

##    "(default `:orthogonal`) Algorithm to use for finding affinely independent set."
##    sampling_algorithm :: Symbol = :orthogonal # :orthogonal or :monte_carlo

##	"(default `:standard_rand`) Algorithm to use if additional points are required."
##    sampling_algorithm2 :: Symbol = :standard_rand

    "(default `typemax(Int64)`) Maximum number of objective evaluations."
    max_evals :: Int64 = typemax(Int64)

	@assert θ_enlarge_1 * θ_pivot ≤ 1 "θ_pivot must be <= θ_enlarge_1^(-1)."

##	@assert sampling_algorithm ∈ [:orthogonal, :monte_carlo] "Sampling algorithm must be either `:orthogonal` or `:monte_carlo`."
    @assert kernel ∈ Symbol.(["gaussian", "inv_multiquadric", "multiquadric", "cubic", "thin_plate_spline"]) "Kernel '$kernel' not supported yet."
	## Some sanity checks for the shape parameters
    @assert kernel != :thin_plate_spline || ( isnan(shape_parameter) || shape_parameter % 1 == 0 && shape_parameter >= 1 ) "Invalid shape_parameter for :thin_plate_spline."
	@assert kernel != :cubic || ( isnan(shape_parameter) || shape_parameter % 1 == 0 && shape_parameter % 2 == 1 ) "Invalid shape_parameter for :cubic."
	@assert (isa( shape_parameter, String ) || isnan(shape_parameter)) || shape_parameter > 0 "Shape parameter must be strictly positive."
    ## @assert θ_enlarge_1 >=1 && θ_enlarge_2 >=1 "θ's must be >= 1."
end

# The required method implementations are straightforward.
# Note, thate we allow the models to be combined to vector functions if they 
# share the same configuration to avoid redundant efforts whilst constructing models.

max_evals( cfg :: RbfConfig ) :: Int = cfg.max_evals
combinable( cfg :: RbfConfig ) :: Bool = true
combine(cfg1 :: RbfConfig, :: RbfConfig) :: RbfConfig = cfg1

# We also need to introduce our own implementation for `isequal` and `hash` for 
# `RbfConfig`s to be combinable, see [the docs too](https://docs.julialang.org/en/v1/base/base/).
function Base.hash( cfg :: RbfConfig, h :: UInt )
	return hash( getfield.( cfg, Tuple( fn for fn ∈ fieldnames(RbfConfig) ) ), h )
end
function Base.isequal( cfg1 :: RbfConfig, cfg2 :: RbfConfig )
	all( isequal( getfield(cfg1, fn), getfield(cfg2, fn) ) for fn in fieldnames( RbfConfig) )
end

# To allow the user to set the shape parameter relative to the current trust region radius 
# using a verbose string, we need this little helper function, which evaluates the string.

function parse_shape_param_string( Δ :: F, expr_str) :: F where F
    ex = Meta.parse(expr_str)
    sp = @eval begin
        let Δ=$Δ
            $ex
        end
    end 
	return sp
end

# The `RbfMeta` is used to store construction and update data for the models.
# To be specific, we have several inidices lists that store database indices 
# of (potentially unevaluated) results that are later used for fitting the model.
@with_kw mutable struct RbfMeta{F<:AbstractFloat} <: SurrogateMeta
    center_index :: Int = -1
    round1_indices :: Vector{Int} = []
    round2_indices :: Vector{Int} = []
    round3_indices :: Vector{Int} = []
    round4_indices :: Vector{Int} = []
    fully_linear :: Bool = false
	improving_directions :: Vector{Vector{F}} = []
end


saveable_type( meta :: T ) where {T<:RbfMeta} = T
saveable( meta :: RbfMeta ) = deepcopy(meta)

# A little helper to retrieve all those indices:
function _collect_indices( meta :: RbfMeta; include_x = true ) :: Vector{Int}
	return [ 
		include_x ? meta.center_index : [];
		meta.round1_indices;
		meta.round2_indices;
		meta.round3_indices;
		meta.round4_indices
	]
end

# And a helper, to partially copy some data from `src` to `dest`.
# This is due to the fact, that the first 3 rounds of construction 
# data gathering are the same for all possible RBF models and we can safe 
# some effort.
function copy_meta!(dest, src)
	dest.center_index = src.center_index
	for fn in [ Symbol("round$(i)_indices") for i = 1: 3 ]
		dest_arr = getfield(dest, fn)
		empty!( dest_arr )
		append!( dest_arr, getfield( src, fn) )
	end
	empty!(dest.improving_directions)
	append!(dest.improving_directions, src.improving_directions)
end

export RbfConfig, RbfMeta, RbfModel

# ## Construction

# The initial `prepare_init_model` function should return a meta object that can be used
# to build an initial surrogate model.
# We delegate the work to `prepare_update_model`.
function prepare_init_model( cfg :: RbfConfig, objf :: AbstractObjective, mop :: AbstractMOP, 
	id :: AbstractIterData{F}, db :: AbstractDB, ac :: AbstractConfig; 
	ensure_fully_linear = true, kwargs...) where F<:AbstractFloat
	meta = RbfMeta{F}()
	return prepare_update_model(nothing, objf, meta, mop, id, db, ac; ensure_fully_linear = true, kwargs... )
end

# Usually, `prepare_update_model` would only accept a model as its first argument.
# Because of the trick from above, we actually allow `nothing`, too.
function prepare_update_model( mod :: Union{Nothing, RbfModel}, objf :: AbstractObjective, meta :: RbfMeta, 
	mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, algo_config :: AbstractConfig;
	ensure_fully_linear = false, force_rebuild = false, meta_array = nothing )
	
	!force_rebuild && @logmsg loglevel2 "Trying to find results for fitting an RBF model."	

	## Retrieve current iteration information and some meta data.
	Δ = get_Δ(iter_data)
	Δ_max = Δᵘ(algo_config)
	x = get_x(iter_data)
	x_index = get_x_index(iter_data)
	cfg = model_cfg( objf )

	F = eltype(x)
	n_vars = length(x)

	## Can we skip the first rounds? (Because we already found interpolation sets for other RBFModels?)
	all_objfs = list_of_objectives(mop)
	skip_first_rounds = false
	for (i,other_meta) in enumerate(meta_array)
		other_objf = all_objfs[i]
		if other_meta isa RbfMeta
			other_cfg = model_cfg(other_objf)
			if other_cfg.θ_pivot == cfg.θ_pivot && other_cfg.θ_enlarge_1 == cfg.θ_enlarge_1 && other_cfg.θ_enlarge_2 == cfg.θ_enlarge_2
				copy_meta!( meta, other_meta )
				skip_first_rounds = true
			end
		end
	end	

	## By default, assume that our model is not fully linear
	meta.fully_linear = false

	## use center as first training site ⇒ at least `n_vars` required still
	meta.center_index = x_index

	## First round of sampling: 
	### Try to find points in slightly enlarged trust region 
	Δ_1 = F.(cfg.θ_enlarge_1 * Δ)
	lb_1, ub_1 = local_bounds( mop, x, Δ_1 )
	piv_val_1 = F.(cfg.θ_pivot * Δ_1) # threshold value for acceptance in filter

	### `Δ_2` is the maximum allowed trust region radius and used in rounds 2 & 4
	Δ_2 = F.(cfg.θ_enlarge_2 * Δ_max )
	lb_2, ub_2 = local_bounds( mop, x, Δ_2 )
	piv_val_2 = piv_val_1 # the pivot value stays the same 

	skip_first_rounds && @goto round4

	if force_rebuild
		### `force_rebuild` makes us skip the point searching procedures to …
		### … rebuild the model along the coordinate axes.
		filtered_indices_1 = Int[]
		improving_directions = [ [zeros(F,i-1); one(F); zeros(F,n_vars - i)] for i = 1:n_vars ]
	else
		@logmsg loglevel3 "Round1: Inspect box with radius $(Δ_1) and pivot value $(piv_val_1)."

		### only consider points from within current trust region …
		candidate_indices_1 = results_in_box_indices( db, lb_1, ub_1, [x_index],)

		### … and filter them to obtain affinely independent points.
		filter = AffinelyIndependentPointFilter(; 
			x_0 = x, 
			seeds = get_site.(db, candidate_indices_1),
			return_indices = true, 
			pivot_val = piv_val_1
		)		
		
		filtered_indices_1 = candidate_indices_1[ collect( filter ) ]
		### TODO should we rather use Z₂ to sample along unexplored directions? (for now, i simply reverse Z₁) #src
		improving_directions = reverse(collect(Vector{F}, eachcol(filter.Z)))

		@logmsg loglevel3 "Round1: Found $(length(filtered_indices_1)) sites in database."
	end
	### Store indices in meta data object:
	empty!(meta.round1_indices)
	append!(meta.round1_indices, filtered_indices_1)
	empty!(meta.improving_directions)
	append!(meta.improving_directions, improving_directions )

	## Second round of sampling:
	### If there are not enough sites to have a fully linear model …
	### … try to at least find more sites in maximum allowed radius
	n_missing = n_vars - length( filtered_indices_1 )

	if n_missing == 0 || force_rebuild || ensure_fully_linear || Δ ≈ Δ_max && cfg.θ_enlarge_1 == cfg.θ_enlarge_2
		@logmsg loglevel3 "Skipping round 2."
	
		meta.fully_linear = true
		filter_2 = filter
		empty!(meta.round2_indices)
	else
		### actually perform round 2

		@logmsg loglevel3 "Missing $(n_missing) sites still."
		@logmsg loglevel3 "Round2: Inspect box with radius $(Δ_2) and pivot value $(piv_val_1)."
			
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

		@logmsg loglevel3 "Round2: Found $(length(meta.round2_indices)) sites and model is $(meta.fully_linear ? "" : "not " )fully linear."
	end

	## Round 3:
	### If we still don't have enough sites, generate them 
	### along model improving directions (from first round of sampling)

	n_missing -= length(meta.round2_indices)
	empty!(meta.round3_indices)
	if n_missing > 0

		@logmsg loglevel3 "Round3: Still missing $(n_missing). Sampling in box of radius $(Δ_1)."
		
		### If round 2 did not yield any new points, the model will hopefully be made fully linear now.
		if length(meta.round2_indices) == 0
			meta.fully_linear = true 
		end

		### Take into consideration the maximum number of evaluations allowed:
		### TODO count unevaluated results too! This is a new requirement due to the 2-phase construction process. #src
		max_new = min( max_evals(algo_config), max_evals(cfg) ) - 1 - num_evals( objf )
		n_new = min(n_missing, max_new)
		
		new_points = Vector{F}[]
		while !isempty(meta.improving_directions) && length( new_points ) < n_new
			dir = popfirst!( meta.improving_directions )
			len = intersect_bounds( x, dir, lb_1, ub_1; return_vals = :absmax )
			offset = len .* dir
			if norm( offset, Inf ) <= piv_val_1
				### the new point does not pass the thresholding test 
				if ensure_fully_linear && !force_rebuild
					### If we need a fully linear model, we dismiss the inidices gathered so far …
					### … and call for a rebuild along the coordinate axis:
					return prepare_update_model(mod, objf, meta, mop, iter_data, db, algo_config; ensure_fully_linear = true, force_rebuild = true)
				else
					### we include the point nonetheless, but the model will not qualify as fully linear...
					meta.fully_linear = false
				end
			end	
			push!( new_points, x .+ offset )
		end

		### by adding the points to the database at this point in time we avoid 
		### requesting unnecessary results from a round 3 interrupted by rebuilding
		new_indices = Int[]
		for p ∈ new_points
			new_id = new_result!( db, p, F[] )
			push!(new_indices, new_id)		
		end

		append!(meta.round3_indices, new_indices)
	end

	@label round4
	
	## In round 4 we have found `n_vars + 1` training sites and try to find additional points within the 
	## largest possible trust region.
	empty!(meta.round4_indices)
		
	max_points = cfg.max_model_points <= 0 ? 2 * n_vars + 1 : cfg.max_model_points
	indices_found_so_far = _collect_indices( meta )
	N = length(indices_found_so_far)
	
	candidate_indices_4 = results_in_box_indices( db, lb_2, ub_2, indices_found_so_far )
	
	max_tries = 10 * max_points	
	num_tries = 0

	if N < max_points && ( !isempty(candidate_indices_4) || cfg.use_max_points )
		@logmsg loglevel3 "Round4: Can we find $(max_points - N) additional sites?"
		round4_indices = Int[]

		chol_pivot = cfg.θ_pivot_cholesky

		centers = get_site.(db, indices_found_so_far)
		φ = _get_radial_function( Δ, cfg )
		Φ, Π, kernels, polys = RBF.get_matrices( φ, centers; poly_deg = cfg.polynomial_degree )
		
		## prepare matrices as by Wild, R has to be augmented by rows of zeros
		Q, R = qr( transpose(Π) )
        R = [
            R;
            zeros( size(Q,1) - size(R,1), size(R,2) )
        ]
        Z = Q[:, N + 1 : end ] ## columns of Z are orthogonal to Π

		## Note: usually, Z, ZΦZ and L should be empty (if N == n_vars + 1)
        ZΦZ = Hermitian(Z'Φ*Z)	## make sure, it is really symmetric
        L = cholesky( ZΦZ ).L   ## perform cholesky factorization
        L⁻¹ = inv(L)				 ## most likely empty at this point

        φ₀ = Φ[1,1]

		@logmsg loglevel3 "Round4: Considering $(length(candidate_indices_4)) candidates."
		
		while N < max_points && num_tries <= max_tries
			
			if !isempty( candidate_indices_4 )
				id = popfirst!( candidate_indices_4 )
				### get candidate site ξ ∈ ℝⁿ
				ξ = get_site( db, id )
			else
				if cfg.use_max_points
					### there are no more sites in the db, but we **want**
					### to use as many as possible
					id = -1
					ξ = _rand_box_point( lb_2, ub_2, F)
					num_tries += 1
				else 
					break 
				end
			end

			### apply all RBF kernels
			φξ = kernels( ξ )
		
			### apply polynomial basis system and augment polynomial matrix
			πξ = polys( ξ )
			Rξ = [ R; πξ' ]

			### perform Givens rotations to turn last row in Rξ to zeros
			row_index = size( Rξ, 1)
			G = Matrix(I, row_index, row_index) # whole orthogonal matrix
			for j = 1 : size(R,2) 
				## in each column, take the diagonal as pivot to turn last elem to zero
				g = givens( Rξ[j,j], Rξ[row_index, j], j, row_index )[1]
				Rξ = g*Rξ;
				G = g*G;
			end

			### now, from G we can update the other matrices 
			Gᵀ = transpose(G)
			g̃ = Gᵀ[1 : end-1, end]
			ĝ = Gᵀ[end, end]

			Qg = Q*g̃;
			v_ξ = Z'*( Φ*Qg + φξ .* ĝ )
			σ_ξ = Qg'*Φ*Qg + (2*ĝ) * φξ'*Qg + ĝ^2*φ₀

			τ_ξ² = σ_ξ - norm( L⁻¹ * v_ξ, 2 )^2 
			## τ_ξ (and hence τ_ξ^2) must be bounded away from zero 
    		## for the model to remain fully linear
			if τ_ξ² > chol_pivot
				
				if id < 0
					id = new_result!( db, ξ, F[] )
				end
				push!(round4_indices, id)	# accept the result
				
				τ_ξ = sqrt(τ_ξ²)

				## zero-pad Q and multiply with Gᵗ
				Q = [
					Q 					zeros( size(Q,1), 1);
					zeros(1, size(Q,2)) 1
				] * Gᵀ

				Z = [ 
					Z  						Qg;
					zeros(1, size(Z,2)) 	ĝ 
				]
				
				L = [
					L          zeros(size(L,1), 1) ;
					v_ξ'L⁻¹'   τ_ξ 
				]

				L⁻¹ = [
					L⁻¹                zeros(size(L⁻¹,1),1);
					-(v_ξ'L⁻¹'L⁻¹)./τ_ξ   1/τ_ξ 
				]

				R = Rξ

				## finally, augment basis matrices and add new kernel for next iteration
				Π = [ Π πξ ]

				Φ = [ 
					Φ   φξ;
					φξ' φ₀
				]
				push!( kernels, RBF.make_kernel(φ, ξ) )

				## assert all( diag( L * L⁻¹) .≈ 1 )
				N += 1
			end#if 
		end#for 
		append!(meta.round4_indices, round4_indices)
		@logmsg loglevel3 "Round4: found $(length(round4_indices)) additional sites."
	end#if

	return meta
end

# !!! note
#     At the moment, we do not store the matrices calculated in round 4 of the 
#     update procedure. This could be done to save some work when actually 
#     calculating the coefficients.

# In contrast to the old RBF mechanism, the models in `RadialBasisFunctionModels` sometimes
# accept 2 parameters for the kernel. We use this little helper, to get defaults from the shape parameter.
# Note, that sanity check are performed in the RbfConfig constructor.
function _get_kernel_params( Δ , cfg )

	F = eltype(Δ)

	sp = if cfg.shape_parameter isa String
		parse_shape_param_string( Δ, cfg.shape_parameter )
	else
		F(cfg.shape_parameter)
	end

	isnan(sp) && return nothing
	
	kernel_name = cfg.kernel
	
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

function _get_radial_function( Δ, cfg )
	kernel_params = _get_kernel_params( Δ, cfg )

	return RBF._get_rad_func( cfg.kernel, kernel_params )
end

# An improvement step consists of adding a new site to the database, along an improving direction:
function prepare_improve_model( mod :: Union{Nothing, RbfModel}, objf :: AbstractObjective, 
	meta :: RbfMeta, mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, 
	algo_config :: AbstractConfig; kwargs... )
	if !meta.fully_linear
		if isempty(meta.improving_directions)
			@warn "RBF model is not fully linear, but there are no improving directions."
		else
			cfg = model_cfg(objf)
			x = get_x(iter_data)
			fx = get_fx(iter_data)
			F = typeof(fx)
			Δ = get_Δ(iter_data)
			Δ_1 = Δ * cfg.θ_enlarge_1
			lb_1, ub_1 = local_bounds(mop, x, Δ_1)
			piv_val_1 = Δ_1 * cfg.θ_pivot

			
			success = false
			dir = popfirst!( meta.improving_directions )
			len = intersect_bounds( x, dir, lb_1, ub_1; return_vals = :absmax )
			offset = len .* dir
			if norm( offset, Inf ) > piv_val_1
				new_id = new_result!( db, x .+ offset, F() )
				push!(meta.round1_indices, new_id)
				success = true
			end	

			success && @logmsg loglevel3 "Performed an improvement step."
			if isempty( meta.improving_directions ) && success
				meta.fully_linear = true 
				@logmsg loglevel3 "The RBF Model is now fully linear."
			end
		end
	end
	return meta
end

# Now, in the 2-phase construction process, first all `prepare_` functions are called for all surrogate models.
# Then, the unevaluated results are evaluated and we can proceed with the model building.
# As before, `_init_model` simply delegates work to `update_model`.

function _init_model( cfg :: RbfConfig, objf :: AbstractObjective, mop :: AbstractMOP,
	iter_data :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig, meta :: RbfMeta; kwargs... )
	return update_model( nothing, objf, meta, mop, iter_data, db, ac; kwargs... )
end

function update_model( mod::Union{Nothing,RbfModel}, objf:: AbstractObjective, meta :: RbfMeta, 
	mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; 
	kwargs... ) 

	Δ = get_Δ(iter_data)
	cfg = model_cfg(objf)

	x = get_x(iter_data)
	F = eltype(x)

	kernel_params = _get_kernel_params( Δ, cfg )

	## get the training data from `meta` and the database `db`
	training_indices = _collect_indices( meta )
	training_results = get_result.( db, training_indices )
	training_sites = get_site.( training_results )
	oi = output_indices( objf, mop)	# only consider the objective output indices 
	training_values = [ v[oi] for v in get_value.( training_results ) ]
	
	inner_model = RBF.RBFInterpolationModel( training_sites, training_values, cfg.kernel, kernel_params; save_matrices = false )
	
	@logmsg loglevel3 "The model is $(meta.fully_linear ? "" : "not ")fully linear."
	return RbfModel( inner_model, meta.fully_linear ), meta 
		
end

# The improvement function also simply cals the update function:
function improve_model( mod::Union{Nothing,RbfModel}, objf:: AbstractObjective, meta :: RbfMeta, 
	mop :: AbstractMOP, id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig; kwargs... ) 
	return update_model( mod, objf, meta, mop, id, db, ac; kwargs... )
end

# ## Evaluation 

# All the work is done by the inner model :)

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

# [^wild_diss]: “Derivative-Free Optimization Algorithms For Computationally Expensive Functions”, Stefan M. Wild, 2009