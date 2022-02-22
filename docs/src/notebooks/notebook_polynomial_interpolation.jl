### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ e9fdca3a-2d74-4a42-acda-2b289172a1d4
begin
	import Pkg
	Pkg.activate(tempname())
	Pkg.add("DynamicPolynomials")
	Pkg.add("NLopt")
	Pkg.add("StaticArrays")
	Pkg.add("GLMakie")
	Pkg.add("Makie")
	Pkg.add("Combinatorics")
	Pkg.add("BenchmarkTools")
end

# ╔═╡ 9956f874-e481-4e3e-97fc-150aef8b69fa
begin
	using DynamicPolynomials
	using BenchmarkTools
	import NLopt
	using StaticArrays
	using Makie, GLMakie
	using Combinatorics
end

# ╔═╡ 0bf0c2a6-7487-49a9-8adf-73381ea2e9fc
md"""
We denote by ``Π_n^d`` the space of ``n``-variate Polyoniams of degree at most ``d``.
To construct polynomials we use `DynamicPolynomials.jl`.
"""

# ╔═╡ af1b5146-5b3a-425d-95a7-6d0532bcca2f
md"""
The canonical basis is obtained by calculating the non-negative integral solutions to the euqation
```math
x_1 + … + x_n \le d.
```
"""

# ╔═╡ 6c51dda0-5c69-4f58-8348-d62245b1eaeb
md"""
These solutions can be found using the `Combinatorics` package via `multiexponents(n,d)` (`d` must be successively increased).
"""

# ╔═╡ 802e1cd2-fceb-4ee7-b178-62b05ced5409
function non_negative_ineq_solutions(deg, n_vars)
	collect(Iterators.flatten( ( collect(multiexponents( n_vars, d )) for d = 0 : deg ) ))
end

# ╔═╡ c94bdc4e-2c95-42ae-a7dd-4172da278585
function get_poly_basis( deg, n_vars)
	exponents = non_negative_ineq_solutions(deg, n_vars )
	polys = let
		@polyvar x[1:n_vars]
		[ prod(x.^e) for e in exponents ]
	end
	return polys
end

# ╔═╡ b70e7d30-6856-4011-8a59-4727c690634f
md"""
We are going to use the canonical basis to determine a **poised set** of points.
This does in fact work with any polynomial basis for ``Π_n^d``. \
In the process of doing so, we also modify (a copy of?) the basis so that it becomes the Lagrange basis for the returned point set.
"""

# ╔═╡ 42ac42a6-d11e-4c03-bb18-0276d6a319fa
md"The Larange basis is formed by normalizing and orthogonalizing with respect to the point set:"

# ╔═╡ 688d4cf3-4e4d-45de-a43c-141646f01b00
function orthogonalize_polys( poly_arr, x, i )
	# normalize i-th polynomial with respect to `x`
	p_i = poly_arr[i] / poly_arr[i](x)
	
	# orthogoalize 
	return [ j != i ? poly_arr[j] - poly_arr[j](x) * p_i : p_i for j = eachindex( poly_arr ) ]
end

# ╔═╡ e8136cef-9235-4ad8-bfac-a9e176923de7
md"""
We use Algorithm 6.2 and Algorithm 6.3 from the book 
"Introduction to Derivative-Free Optimization"
by Conn et. al. \
Algorithm 6.2 makes the set poised (suited for interpolation) and returns the corresponding Lagrange basis.
Algorithm 6.3 takes the poised set and the Lagrange basis and tries to make it ``Λ``-poised. ``Λ`` must be greater 1 and a smaller value makes the set more suited for good models.
"""

# ╔═╡ f48b8d93-e98c-4f30-960c-203a8077b6b6
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
function get_poised_set( basis, points :: AbstractArray{T} = Vector{Float32}[]; 
		solver = :LN_BOBYQA, max_solver_evals = -1 ) where {
		T <: AbstractArray{<:Real}
	}
	
	p = length(basis)
	@assert p > 0 "`basis` must not be an empty array."

	vars = variables( basis[end] )
	n_vars = length(vars)
	@assert n_vars > 0 "The number of variables must be positive."
	
	if max_solver_evals < 0
		max_solver_evals = 2000 * n_vars
	end

	F = promote_type( eltype( T ), Float32 )
	P_type = n_vars > 100 ? Vector{Vector{F}} : Vector{SVector{n_vars, F}}
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
			## indicate what the actual point index was  
			point_indices[i] = not_accepted_indices[j]
			## delete from further consideration
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
			
			x₀_tmp = [ rand(F, n_vars) for i = 1 : 50 * n_vars ]
            x₀ = x₀_tmp[argmax( abs.(new_basis[i].(x₀_tmp)) ) ] 
            
			_, ξ, ret = NLopt.optimize(opt, x₀)
			
			poised_points[i] = ξ
		end		
		
		new_basis = orthogonalize_polys( new_basis, poised_points[i], i )
	end
	
	return poised_points, new_basis, point_indices
end
			

# ╔═╡ 85d345a0-2d3f-46ed-8b39-f88f700d545b
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
		
		F = promote_type( eltype( T ), Float32 )
		P_type = n_vars > 100 ? Vector{Vector{F}} : Vector{SVector{n_vars, F}}

		if max_loops < 0 
			max_loops = length(basis) * 100
		end

		if max_solver_evals < 0
			max_solver_evals = 2000 * n_vars
		end

		new_basis = basis
		new_points = P_type(points)
		point_indices = collect(eachindex(new_points))

		for k = 1 : max_loops
			iₖ = -1
			xₖ = points[1]
			for (i, polyᵢ) in enumerate(new_basis)
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
					if i ∉ skip_indices
						# i is not prioritized we can brake here
						break
					end#if
				end#if
			end#for

			if iₖ > 0
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

# ╔═╡ c417a8e4-2564-451a-8105-58ae5f713720
md"For the sake of convenience we combine both functions:"

# ╔═╡ a57e64c4-cd79-485a-b21d-97591784cd1c
function get_lambda_poised_set( basis, points; solver1 = :LN_BOBYQA, solver2 = :LN_BOBYQA, max_solver_evals1 = -1, max_solver_evals2 = -1, LAMBDA = 1.5, max_lambda_loops = -1 )
	lagrange_points, lagrange_basis, lagrange_indices = get_poised_set( 
		basis, points; solver = solver1, max_solver_evals = max_solver_evals1 )
	lambda_points, lambda_basis, lambda_indices = make_set_lambda_poised( 
		lagrange_basis, lagrange_points; LAMBDA, max_loops = max_lambda_loops,
		solver = solver2, max_solver_evals = max_solver_evals2 )
	combined_indices = [ i < 0 ? i : lagrange_indices[j] for (j,i) in enumerate( lambda_indices ) ]
	return lambda_points, lambda_basis, combined_indices
end

# ╔═╡ ffc8cd79-9d85-4f04-81e1-3ae0193395b9
md"Let's have a look at what the points look like:"

# ╔═╡ f72d72de-119b-4c4c-a0d7-38b8ac76c2d6
begin 
	basis = get_poly_basis(2,2)
	custom_points =  [ ones(Float32, 2), ones(Float32,2)] 
	
	#lambda_points, lambda_basis, c_indices = get_lambda_poised_set( basis,custom_points)
	lambda_points, lambda_basis, c_indices = get_poised_set( basis,custom_points)
	c_indices
end

# ╔═╡ 9659c50c-c30f-4e78-a80a-eaee23209a4d
scatter(Tuple.(lambda_points))

# ╔═╡ Cell order:
# ╠═e9fdca3a-2d74-4a42-acda-2b289172a1d4
# ╠═9956f874-e481-4e3e-97fc-150aef8b69fa
# ╟─0bf0c2a6-7487-49a9-8adf-73381ea2e9fc
# ╠═af1b5146-5b3a-425d-95a7-6d0532bcca2f
# ╠═6c51dda0-5c69-4f58-8348-d62245b1eaeb
# ╠═802e1cd2-fceb-4ee7-b178-62b05ced5409
# ╠═c94bdc4e-2c95-42ae-a7dd-4172da278585
# ╠═b70e7d30-6856-4011-8a59-4727c690634f
# ╟─42ac42a6-d11e-4c03-bb18-0276d6a319fa
# ╠═688d4cf3-4e4d-45de-a43c-141646f01b00
# ╠═e8136cef-9235-4ad8-bfac-a9e176923de7
# ╠═f48b8d93-e98c-4f30-960c-203a8077b6b6
# ╠═85d345a0-2d3f-46ed-8b39-f88f700d545b
# ╟─c417a8e4-2564-451a-8105-58ae5f713720
# ╠═a57e64c4-cd79-485a-b21d-97591784cd1c
# ╟─ffc8cd79-9d85-4f04-81e1-3ae0193395b9
# ╠═f72d72de-119b-4c4c-a0d7-38b8ac76c2d6
# ╠═9659c50c-c30f-4e78-a80a-eaee23209a4d
