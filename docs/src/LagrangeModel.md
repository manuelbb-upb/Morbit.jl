```@meta
EditURL = "<unknown>/src/LagrangeModel.jl"
```

# Lagrange Polynomial Models

## Intro and Prerequisites

Polyoniaml interpolation models are a common choice for surrogate modeling.
In our setting we want to construct models for ``n``-variate objectives and
use polynomials of degree 1 or 2.
We hence need a basis for the space ``Π_n^d`` of polynomials.
Given a point set that is suited for interpolation (a *poised* set) we can
use the Lagarange basis ``\{l_i\}`` with ``l_i(x_j) = δ_{i,j}`` to easily
find the coefficients for vector valued models.

We use `DynamicPolynomials` for polynomial arithmetic, `NLopt` to optimize
polynomials and some more packages:

````julia
using DynamicPolynomials
import NLopt
import Combinatorics
````

## Surrogate Interface Implementations

The model itself is defined only by its vector of Lagrange basis polynomials
and the coefficients.

````julia
@with_kw struct LagrangeModel{
        B <: AbstractArray{<:AbstractPolynomialLike},
        G <: AbstractArray{<:AbstractArray{<:AbstractPolynomialLike}},
        V <: AbstractVector{<:AbstractVector{<:AbstractFloat} } } <: SurrogateModel
    basis :: B
    grads :: G
    coeff :: V
    fully_linear :: Bool = false
end

fully_linear( lm :: LagrangeModel ) = lm.fully_linear
````

There is a multitude of configuration parameters, most of which will
be explained later:

````julia
@with_kw mutable struct LagrangeConfig <: SurrogateConfig

    "Degree of the surrogate model polynomials."
    degree :: Int = 2

    "Enlargement parameter to consider more points for inclusion."
    θ_enlarge :: Real = 2

    "Quality parameter in Λ-Poisedness Algorithm."
    LAMBDA :: Real = 1.5

    "Whether or not the interpolation sets must be Λ-poised (and the models fully linear)."
    allow_not_linear :: Bool = false

    "Whether or not to try to construct a new interpolation set in each iteration."
    optimized_sampling :: Bool = true

    # if optimized_sampling = false, shall we try to use saved sites?
    save_path :: String = ""
    io_lock :: Union{Nothing, Threads.ReentrantLock} = nothing

    algo1_max_evals :: Int = -1
    algo2_max_evals :: Int = -1

    algo1_solver :: Symbol = :LN_BOBYQA
    algo2_solver :: Symbol = :LN_BOBYQA

    max_evals :: Int64 = typemax(Int64);

    @assert 1 <= degree <= 2 "Only linear and quadratic models are supported."
    @assert LAMBDA > 1 "`LAMBDA` must be > 1."
    @assert let algo_str = string( algo1_solver );
        length( algo_str ) > 2 && string(algo_str[2]) == "N"
    end "`algo1_solver` must be a derivative-free NLopt algorithm."
    @assert let algo_str = string( algo2_solver );
        length( algo_str ) > 2 && string(algo_str[2]) == "N"
    end "`algo2_solver` must be a derivative-free NLopt algorithm."
end
````

Overwrite `lock` and `unlock` so we can use `nothing` as a "lock":

````julia
function Base.lock(::Nothing) end
function Base.unlock(::Nothing) end
````

The required method implementations are straightforward.
Note, thate we allow the models to be combined to vector functions if they
share the same configuration to avoid redundant efforts whilst constructing models.

````julia
max_evals( cfg :: LagrangeConfig ) :: Int = cfg.max_evals
combinable( cfg :: LagrangeConfig ) :: Bool = true
````

We also need to introduce our own implementation for `isequal` and `hash` for
`LagrangeConfig`s to be combinable, see [the docs too](https://docs.julialang.org/en/v1/base/base/).

````julia
function Base.hash( cfg :: LagrangeConfig, h :: UInt )
	return hash( getfield.( cfg, Tuple( fn for fn ∈ fieldnames(LagrangeConfig) ) ), h )
end
function Base.isequal( cfg1 :: LagrangeConfig, cfg2 :: LagrangeConfig )
	all( isequal( getfield(cfg1, fn), getfield(cfg2, fn) ) for fn in fieldnames( LagrangeConfig) )
end
````

The `LagrangeMeta` simply holds the database indices of the results
we want to interpolate at.
We also store the output indices of the model for convenience and carry
polynomials that act on ``[0,1]^n``.

````julia
@with_kw struct LagrangeMeta{
        CB <: Union{Nothing, Vector{<:AbstractPolynomialLike}},
        LB <: Union{Nothing, Vector{<:AbstractPolynomialLike}},
        P <: Union{Nothing, AbstractVector{<:AbstractVector{<:Real}}}
    } <: SurrogateMeta
    interpolation_indices :: Vector{Int} = []
    out_indices :: Vector{Int} = []
    canonical_basis :: CB = nothing
    lagrange_basis :: LB = nothing
    stamp_points :: P = nothing     ## used only if `unoptimized_sampling == false`
    fully_linear :: Bool = false
end

saveable_type( T :: LagrangeMeta ) = LagrangeMeta{Nothing,Nothing}
saveable( meta :: LagrangeMeta ) = LagrangeMeta(;
    interpolation_indices = meta.interpolation_indices, out_indices = meta.output_indices )

export LagrangeConfig, LagrangeMeta, LagrangeModel
````

## Construction

### A Bit of Theory
The canonical basis is obtained by calculating the non-negative
integral solutions to the euqation
```math
x_1 + … + x_n \le d.
```
These solutions can be found using the `Combinatorics`
package via `multiexponents(n,d)` (`d` must be successively increased).

````julia
function non_negative_ineq_solutions(deg, n_vars)
	Iterators.flatten( ( collect( Combinatorics.multiexponents( n_vars, d )) for d = 0 : deg ) )
end

function get_poly_basis( deg, n_vars)
	exponents = non_negative_ineq_solutions(deg, n_vars )
	polys = let
		@polyvar x[1:n_vars]
		[ prod(x.^e) for e in exponents ]
	end
	return polys
end
````

We are going to use the canonical basis to
determine a **poised set** of points.
This does in fact work with any polynomial basis for ``Π_n^d``. \
In the process of doing so, we also modify (a copy of?)
the basis so that it becomes the Lagrange basis for the returned point set.

The Larange basis is formed by normalizing and orthogonalizing with respect to the point set:

````julia
function orthogonalize_polys( poly_arr, x, i )
	# normalize i-th polynomial with respect to `x`
	p_i = poly_arr[i] / poly_arr[i](x)

	# orthogonalize
	return [ j != i ? poly_arr[j] - ( poly_arr[j](x) * p_i ) : p_i for j = eachindex( poly_arr ) ]
end
````

We use Algorithm 6.2 and Algorithm 6.3 from the book
"Introduction to Derivative-Free Optimization"
by Conn et. al. \
Algorithm 6.2 makes the set poised (suited for interpolation) and
returns the corresponding Lagrange basis.
Algorithm 6.3 takes the poised set and the Lagrange basis and
tries to make it ``Λ``-poised. ``Λ`` must be greater 1 and
a smaller value makes the set more suited for good models.

````julia
"""
    get_poised_set( basis, points; solver = :LN_BOBYQA, max_solver_evals = -1 )

Compute a point set suited for polynomial interpolation.

Input:
* `basis`: A vector of polynomials constituting a basis for the polynomial space.
* `points`: (optional) A set of candidate points to be tried for inclusion into the poised set.
* `solver`: NLopt solver to use. Should be derivative-free.
* `max_solver_evals`: Maximum number of evaluations in each optimization run.

Return:
* `poised_points :: Vector{T}` where `T` is either a `Vector{F}` or an `SVector{n_vars, F}` and `F` is the precision of the points in `points`, but at least `Float32`.
* `lagrange_basis :: Vector{<:AbstractPolynomialLike}`: The Lagrange basis corresponding to `poised_points`.
* `point_indices`: An array indicating which points from `points` are also in `poised_points`. A positive entry corresponds to the index of a poised point in `points`. If a poised point is new, then the entry is `-1`.
"""
function get_poised_set( basis, points :: AbstractArray{T} = Vector{MIN_PRECISION}[];
		solver = :LN_BOBYQA, max_solver_evals = -1 ) where {
		T <: AbstractArray{<:Real}
	}

	p = length(basis)
	@assert p > 0 "`basis` must not be an empty array."

    @logmsg loglevel3 "Trying to find a poised set with $(p) points."

	vars = variables( basis[end] )
	n_vars = length(vars)
	@assert n_vars > 0 "The number of variables must be positive."

	if max_solver_evals < 0
		max_solver_evals = 2000 * n_vars
	end

	F = promote_type( eltype( T ), MIN_PRECISION )
	#P_type = n_vars > 100 ? Vector{Vector{F}} : Vector{SVector{n_vars, F}}
    P_type = Vector{Vector{F}}
	ZERO_TOL = min(eps(F) * 100, eps(Float16) * 10)

	# indicates which points from points have been accepted
	point_indices = fill(-1, p)
	not_accepted_indices = collect( eachindex( points ) )
	# return array of points that form a poised set
	poised_points = P_type(undef, p)

	new_basis = basis
	for i = 1 : p
		_points = points[not_accepted_indices]

		# find the point that maximizes the i-th polynomial
		# if the polynomial is constant, then the first remaining point is used (j = 1)
		l_max, j = if isempty(_points)
			0.0, 0
		else
			findmax( abs.( [ new_basis[i]( x ) for x in _points ] ) )
		end

		if l_max > ZERO_TOL
			# accept the `j`-th point from `_points`
			poised_points[i] = _points[j]
			### indicate what the actual point index was
			point_indices[i] = not_accepted_indices[j]
			### delete from further consideration
			deleteat!(not_accepted_indices, j)
		else
			# no point was suitable to add to the set
			# trying to find the maximizer for a | l_i(x) |
			opt = NLopt.Opt( solver, n_vars )
			opt.lower_bounds = zeros(F, n_vars )
            opt.upper_bounds = ones(F, n_vars )
            opt.maxeval = max_solver_evals
            opt.xtol_rel = 1e-3
            opt.max_objective = (x,g) -> abs( new_basis[i](x) )

            # try to find a good starting point
			x₀_tmp = [ rand(F, n_vars) for i = 1 : 50 * n_vars ]
            x₀ = x₀_tmp[argmax( abs.(new_basis[i].(x₀_tmp)) ) ]

			_, ξ, ret = NLopt.optimize(opt, x₀)

			poised_points[i] = ξ
		end

		new_basis = orthogonalize_polys( new_basis, poised_points[i], i )
	end

	return poised_points, new_basis, point_indices
end

"""
    make_set_lambda_poised( basis, points;
        LAMBDA = 1.5, solver = :LN_BOBYQA, max_solver_evals = -1, max_loops = -1, skip_indices = [1,] )

Make the output of `get_poised_set` even better suited for interpolation.

Input:
* `basis`: A vector of polynomials constituting a Lagrange basis for the polynomial space.
* `points`: The vector of points belonging to the Lagrange basis.
* `LAMBDA :: Real > 1`: Determines the quality of the interpolation.
* `solver`: NLopt solver to use. Should be derivative-free.
* `max_solver_evals`: Maximum number of evaluations in each optimization run.
* `max_loops`: Maximum number of loops that try to make the set Λ-poised.
* `skip_indices`: Inidices of points to discard last.

Return:
* `poised_points :: Vector{T}` where `T` is either a `Vector{F}` or an `SVector{n_vars, F}` and `F` is the precision of the points in `points`, but at least `Float32`.
* `lagrange_basis :: Vector{<:AbstractPolynomialLike}`: The Lagrange basis corresponding to `poised_points`.
* `point_indices`: An array indicating which points from `points` are also in `poised_points`. A positive entry corresponds to the index of a poised point in `points`. If a poised point is new, then the entry is `-1`.
"""
function make_set_lambda_poised( basis, points :: AbstractArray{T};
		LAMBDA :: Real = 1.5, solver = :LN_BOBYQA, max_solver_evals = -1,
		max_loops = -1, skip_indices = [1,] ) where {
		T <: AbstractArray{<:Real}
	}

	@assert length(basis) == length(points) "Polynomial array `basis` and point array `points` must have the same length."
	if length(points) > 0
		n_vars = length(points[1])
		@assert n_vars > 0 "The number of variables must be positive."

		F = promote_type( eltype( T ), MIN_PRECISION )
		#P_type = n_vars > 100 ? Vector{Vector{F}} : Vector{SVector{n_vars, F}}
        P_type = Vector{Vector{F}}

		if max_loops < 0
			max_loops = length(basis) * 100
		end

		if max_solver_evals < 0
			max_solver_evals = 2000 * n_vars
		end

       	@logmsg loglevel3 "Trying $(max_loops) times to make a set poised with Λ = $(LAMBDA)."

		iₖ = -1
		xₖ = points[1]

		new_basis = basis
		new_points = P_type(points)
		point_indices = collect(eachindex(new_points))

		for k = 1 : max_loops
			for (i, polyᵢ) in enumerate(basis)
				opt = NLopt.Opt( solver, n_vars )
				opt.lower_bounds = zeros(F, n_vars)
				opt.upper_bounds = ones(F, n_vars)
				opt.maxeval = max_solver_evals
				opt.xtol_rel = 1e-3
				opt.max_objective = (x,g) -> abs( polyᵢ( x ) )

				x₀_tmp = [ rand(F, n_vars) for i = 1 : 50 * n_vars ]
				x₀ = x₀_tmp[argmax( abs.(new_basis[i].(x₀_tmp)) ) ]

				abs_lᵢ, xᵢ, _ = NLopt.optimize(opt, x₀)

				if abs_lᵢ > LAMBDA
					iₖ = i
					xₖ = xᵢ
					if iₖ ∉ skip_indices
						# i is not prioritized we can brake here
						break
					end#if
				end#if
			end#for

			if iₖ > 0
                @logmsg loglevel4 "Discarding point $(iₖ)."
				# perform a point swap
				new_points[iₖ] = xₖ
				point_indices[iₖ] = -1
				# adapt coefficients of lagrange basis
				new_basis = orthogonalize_polys( new_basis, xₖ, iₖ )
			else
				# we are done, the set is lambda poised
				break
			end#if
		end#for

		return new_points, new_basis, point_indices
	else
		return points, basis, collect(eachindex(points))
	end

end
````

And a convenient function that combines both steps:

````julia
function get_lambda_poised_set( basis, points; solver1 = :LN_BOBYQA, solver2 = :LN_BOBYQA, max_solver_evals1 = -1, max_solver_evals2 = -1, LAMBDA = 1.5, max_lambda_loops = -1 )
	lagrange_points, lagrange_basis, lagrange_indices = get_poised_set(
		basis, points; solver = solver1, max_solver_evals = max_solver_evals1 )
	lambda_points, lambda_basis, lambda_indices = make_set_lambda_poised(
		lagrange_basis, lagrange_points; LAMBDA, max_loops = max_lambda_loops,
		solver = solver2, max_solver_evals = max_solver_evals2 )
	combined_indices = [ i < 0 ? i : lagrange_indices[j] for (j,i) in enumerate( lambda_indices ) ]
	return lambda_points, lambda_basis, combined_indices
end
````

We actually only try to find points suitable points in the hypercube ``[0,1]^n``.
The points can be (un)scaled with the usual methods.
But for `Polynomial`s we can actually use substition to make evaluation more effective.

````julia
"Return vector of polynomials that unscales variables from [0,1]^n to [lb,ub]."
function get_unscaling_poly( vars, lb, ub )
    # we don't have to check for Inf here because of finite trust region
    w = ub .- lb
    return vars .* w .+ lb
end

"Return vector of polynomials that scales variables from [lb, ub] to [0,1]^n."
function get_scaling_poly( vars, lb, ub )
    w = ub .- lb
    return ( vars .- lb ) ./ w
end
````

### Method Implementations

We will use the functions from above in the `prepare_XXX` routines:\
The initial `prepare_init_model` function should return a meta object that can be used
to build an initial surrogate model.
We delegate the work to `prepare_update_model`.

````julia
function prepare_init_model( cfg :: LagrangeConfig, objf :: AbstractObjective, mop :: AbstractMOP,
	id :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig;
	ensure_fully_linear = true, kwargs...)

    n_vars = num_vars( mop )

	meta = LagrangeMeta(;
        canonical_basis = get_poly_basis( cfg.degree, n_vars ),
        out_indices = output_indices(objf, mop)
    )
	return prepare_update_model(nothing, objf, meta, mop, id, db, ac; ensure_fully_linear, kwargs... )
end
````

Usually, `prepare_update_model` would only accept a model as its first argument.
Because of the trick from above, we actually allow `nothing`, too.

````julia
"""
    _consume_points(data_base, poised_points, poised_indices, candidate_indices)

Helper to return array of database indices for `poised_points` and
`poised_indices`. Add result to database if index is -1.
`candidate_indices` are the database indices of the points from the trust region.
"""
function _consume_points( db, poised_points, poised_indices, candidate_indices, lb, ub, F )
    interpolation_indices = Int[]
    for (i,ind) in enumerate(poised_indices)
        if ind < 0
            # we need an additional new site
            new_db_id = new_result!(db, _unscale(poised_points[i], lb, ub), F[] )
            push!(interpolation_indices, new_db_id)
        else
            # we could recycle a candidate point
            push!(interpolation_indices, candidate_indices[ind])
        end
    end
    return interpolation_indices
end

function _scale_poly_basis( poised_basis, lb, ub )
    # we modify the basis so that the input is scaled to [0,1]^n with respect to
    # the enlarged trust region bounds, because the poisedness algos sought points there
    poly_vars = variables( poised_basis[1] )
    scaling_poly = get_scaling_poly( poly_vars, lb, ub )

    zero_pol = sum( 0 .* poly_vars ) # TODO remove once https://github.com/JuliaAlgebra/DynamicPolynomials.jl/issues/92 is fixed

    return [ subs(p, poly_vars => scaling_poly) + zero_pol for p in poised_basis ]
end

function prepare_update_model( mod :: Union{Nothing, LagrangeModel}, objf :: AbstractObjective,
    meta :: LagrangeMeta,  mop :: AbstractMOP, iter_data :: AbstractIterData,
    db :: AbstractDB, algo_config :: AbstractConfig;
    ensure_fully_linear = true, kwargs... )

    x = get_x( iter_data )
    fx = get_fx( iter_data )
    F = eltype(fx)
    x_index = get_x_index( iter_data )
    n_vars = length(x)
    Δ = get_Δ( iter_data )

    cfg = model_cfg(objf)
    lb, ub = local_bounds(mop, x, Δ * cfg.θ_enlarge )

    if cfg.optimized_sampling
        # Find points in current trust region …
        candidate_indices = [x_index; results_in_box_indices( db, lb, ub, [x_index,] )]
        # … and scale them to [0,1]^n
        candidate_points = [_scale(ξ, lb, ub) for ξ in get_site.(db, candidate_indices)]

        # Get a poised set and lagrange basis
        poised_points, poised_basis, poised_indices = get_poised_set(
            meta.canonical_basis, candidate_points;
            solver = cfg.algo1_solver, max_solver_evals = cfg.algo1_max_evals
        )

        fully_linear = false
        # Make set even better
        if ensure_fully_linear || !cfg.allow_not_linear
            ### We would like to keep x if possible
            skip_indices = let l = findfirst( i -> i == 1, poised_indices );
                isnothing(l) ? [] : [l,]
            end

            poised_points, poised_basis, indices_2 = make_set_lambda_poised(
                poised_basis, poised_points;
                LAMBDA = cfg.LAMBDA, solver = cfg.algo2_solver,
                max_solver_evals = cfg.algo2_max_evals, skip_indices
            )
            poised_indices = [ i < 0 ? i : poised_indices[j] for (j,i) in enumerate( indices_2 ) ]
            fully_linear = true
        end

        interpolation_indices = _consume_points( db, poised_points, poised_indices, candidate_indices, lb, ub, F)
        scaled_basis = _scale_poly_basis( poised_basis, lb, ub )

        return LagrangeMeta(;
            interpolation_indices,
            out_indices = meta.out_indices,
            canonical_basis = meta.canonical_basis,
            lagrange_basis = scaled_basis,
            fully_linear
        )

    else
        # unoptimized sampling: we only look for a good point set once
        # in the very first iteration and store the basis and the points
        # in the meta data which is then passed through in subsequent iterations
        lpoints, lbasis = if isnothing(meta.lagrange_basis)
            candidate_points = [ fill(.5, n_vars) ]
            lpoints, lbasis, _ = get_lambda_poised_set(
                meta.canonical_basis, candidate_points;
                solver1 = cfg.algo1_solver, solver2 = cfg.algo2_solver,
                max_solver_evals1 = cfg.algo1_max_evals, max_solver_evals2 = cfg.algo2_max_evals,
                LAMBDA = cfg.LAMBDA )

            lpoints, _scale_poly_basis( lbasis, lb, ub )
        else
            meta.stamp_points, meta.lagrange_basis
        end

        candidate_indices = [x_index,]
        @show lindices = fill(-1, length(lpoints))

        # check if x (scaled to [0,1] wrt trust region bounds) is center of `lpoints`
        #src TODO does using `≈` make problems for small trust region radii? `==` always fails
        x_s = _scale(x, lb, ub)
        x_in_points_index = findfirst(χ -> χ ≈ x_s, lpoints )
        if !isnothing(x_in_points_index)
             candidate_indices[ x_in_points_index ] = 1
        end

        interpolation_indices = _consume_points( db, lpoints, lindices, candidate_indices, lb, ub, F)

        return LagrangeMeta(;
            interpolation_indices,
            out_indices = meta.out_indices,
            lagrange_basis = lbasis,
            stamp_points = lpoints,
            fully_linear = true
        )
    end
end#function
````

The improvement preparation enforces a Λ-poised set:

````julia
function prepare_improve_model( mod :: Union{Nothing, LagrangeModel}, objf :: AbstractObjective, meta :: LagrangeMeta,
    mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, algo_config :: AbstractConfig;
    kwargs... )
    return prepare_update_model( mod, objf, meta, mop, iter_data, db, algo_config; ensure_fully_linear = true, kwargs...)
end
````

Now, in the 2-phase construction process, first all `prepare_` functions are called for all surrogate models.
Then, the unevaluated results are evaluated and we can proceed with the model building.
As before, `_init_model` simply delegates work to `update_model`. \
Not much is left to do, only to retrieve the correct values from the database to use as
coefficients.
We also store the gradient (vector of polynomials) for each basis polynomial.

````julia
function _init_model( cfg :: LagrangeConfig, objf :: AbstractObjective, mop :: AbstractMOP,
	iter_data :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig, meta :: LagrangeMeta; kwargs... )
	return update_model( nothing, objf, meta, mop, iter_data, db, ac; kwargs... )
end

function update_model( mod::Union{Nothing,LagrangeModel}, objf:: AbstractObjective,
    meta :: LagrangeMeta, mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig;
	kwargs... )

    coeff = [ c[ meta.out_indices ] for c in get_value.(db, meta.interpolation_indices) ]

    return LagrangeModel(;
        coeff, fully_linear = meta.fully_linear,
        basis = copy(meta.lagrange_basis),
        # NOTE I don't know why I need to copy here
        # but if i don't copy then testing fails:
        # the meta data does hold a valid Lagrange basis but the model does not !?
        grads = [ differentiate( p, variables(p) ) for p in meta.lagrange_basis ]
    ), meta
end

function improve_model( mod::Union{Nothing,LagrangeModel}, objf:: AbstractObjective,
    meta :: LagrangeMeta, mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, ac :: AbstractConfig;
	kwargs... )
    return update_model( mod, objf, meta, mop, iter_data, db, algo_config; kwargs...)
end
````

## Evaluation
The evaluation of some output is
```math
\sum_{i=1}^p c_i l_i( x ),
```
where ``p = \dim Π_n^d``.

````julia
function _eval_poly_vec( poly_vec, x )
    [ p(x) for p in poly_vec ]
end

function eval_models( lm :: LagrangeModel, x̂ :: Vec, ℓ :: Int)
    return sum( c[ℓ] * p(x̂) for (c,p) in zip( lm.coeff, lm.basis ) )
end

function eval_models( lm :: LagrangeModel, x̂ :: Vec )
    return sum( c * p(x̂) for (c,p) in zip( lm.coeff, lm.basis ) )
end

function get_gradient( lm :: LagrangeModel, x̂ :: Vec, ℓ :: Int )
    sum( c[ℓ] * _eval_poly_vec(p,x̂) for (c,p) in zip( lm.coeff, lm.grads ) )
end

function get_jacobian( lm :: LagrangeModel, x̂ :: Vec )
    grad_evals = [ _eval_poly_vec(p,x̂) for p in lm.grads ]
    no_out = length(lm.coeff[1])
    return transpose( hcat( (sum( c[ℓ] * g for (c,g) in zip( lm.coeff, grad_evals) ) for ℓ = 1 : no_out)... ) )
end
````

## Summary & Quick Examples

1. To use the default configuration for a scalar objective `f` do
   ```julia
   add_objective!(mop, f, LagrangeConfig())
   ```
2. For a vector valued objective do
   ```julia
   add_vector_objective!(mop, f, LagrangeConfig(); n_out = 2)
   ```
3. If you want a linear polyonmial only:
   ```julia
   add_objective!(mop, f, LagrangeConfig(;degree=1))
   ```
4. By default, a new interpolation set is built in everey iteration.
   To use a "stamp" instead, turn of optimized sampling:
   ```julia
   add_objective!(mop, f, LagrangeConfig(;optimized_sampling=true))
   ```

### Complete usage example
```julia
using Morbit
Morbit.print_all_logs()
mop = MixedMOP(3)

F = x -> [ sum( ( x .- 1 ).^2 ); sum( ( x .+ 1 ).^2 ) ]

add_vector_objective!( mop, F, LagrangeConfig() )

x_fin, f_fin, _ = optimize( mop, [-π, ℯ, 0])
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

