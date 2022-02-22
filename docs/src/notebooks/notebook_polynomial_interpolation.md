```@raw html
<style>
    table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    pre, div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "82012d577aa3f4c8a17e8ae42e5be8699cf7aae035a78d6f6e96e99bcbd777d2"
    julia_version = "1.6.1"
-->
<pre class='language-julia'><code class='language-julia'>begin
	import Pkg
	Pkg.activate(tempname())
	Pkg.add("DynamicPolynomials")
	Pkg.add("NLopt")
	Pkg.add("StaticArrays")
	Pkg.add("CairoMakie")
	Pkg.add("Makie")
	Pkg.add("Combinatorics")
	Pkg.add("BenchmarkTools")
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
	using DynamicPolynomials
	using BenchmarkTools
	import NLopt
	using StaticArrays
	using Makie, CairoMakie
	using Combinatorics
end</code></pre>



<div class="markdown"><p>We denote by <span class="tex">$Π_n^d$</span> the space of <span class="tex">$n$</span>-variate Polyoniams of degree at most <span class="tex">$d$</span>. To construct polynomials we use <code>DynamicPolynomials.jl</code>.</p>
</div>


<div class="markdown"><p>The canonical basis is obtained by calculating the non-negative integral solutions to the euqation</p>
<p class="tex">$$x_1 &#43; … &#43; x_n \le d.$$</p>
</div>


<div class="markdown"><p>These solutions can be found using the <code>Combinatorics</code> package via <code>multiexponents&#40;n,d&#41;</code> &#40;<code>d</code> must be successively increased&#41;.</p>
</div>

<pre class='language-julia'><code class='language-julia'>function non_negative_ineq_solutions(deg, n_vars)
	collect(Iterators.flatten( ( collect(multiexponents( n_vars, d )) for d = 0 : deg ) ))
end</code></pre>
<pre id='var-non_negative_ineq_solutions' class='pre-class'><code class='code-output'>non_negative_ineq_solutions (generic function with 1 method)</code></pre>

<pre class='language-julia'><code class='language-julia'>function get_poly_basis( deg, n_vars)
	exponents = non_negative_ineq_solutions(deg, n_vars )
	polys = let
		@polyvar x[1:n_vars]
		[ prod(x.^e) for e in exponents ]
	end
	return polys
end</code></pre>
<pre id='var-get_poly_basis' class='pre-class'><code class='code-output'>get_poly_basis (generic function with 1 method)</code></pre>


<div class="markdown"><p>We are going to use the canonical basis to determine a <strong>poised set</strong> of points. This does in fact work with any polynomial basis for <span class="tex">$Π_n^d$</span>. <br />In the process of doing so, we also modify &#40;a copy of?&#41; the basis so that it becomes the Lagrange basis for the returned point set.</p>
</div>


<div class="markdown"><p>The Larange basis is formed by normalizing and orthogonalizing with respect to the point set:</p>
</div>

<pre class='language-julia'><code class='language-julia'>function orthogonalize_polys( poly_arr, x, i )
	# normalize i-th polynomial with respect to `x`
	p_i = poly_arr[i] / poly_arr[i](x)
	
	# orthogoalize 
	return [ j != i ? poly_arr[j] - poly_arr[j](x) * p_i : p_i for j = eachindex( poly_arr ) ]
end</code></pre>
<pre id='var-orthogonalize_polys' class='pre-class'><code class='code-output'>orthogonalize_polys (generic function with 1 method)</code></pre>


<div class="markdown"><p>We use Algorithm 6.2 and Algorithm 6.3 from the book  &quot;Introduction to Derivative-Free Optimization&quot; by Conn et. al. <br />Algorithm 6.2 makes the set poised &#40;suited for interpolation&#41; and returns the corresponding Lagrange basis. Algorithm 6.3 takes the poised set and the Lagrange basis and tries to make it <span class="tex">$Λ$</span>-poised. <span class="tex">$Λ$</span> must be greater 1 and a smaller value makes the set more suited for good models.</p>
</div>

<pre class='language-julia'><code class='language-julia'>"""
    get_poised_set( basis, points; solver = :LN_BOBYQA, max_solver_evals = -1 )

Compute a point set suited for polynomial interpolation.

Input:
* `basis`: A vector of polynomials constituting a basis for the polynomial space.
* `points`: (optional) A set of candidate points to be tried for inclusion into the poised set.
* `solver`: NLopt solver to use. Should be derivative-free.
* `max_solver_evals`: Maximum number of evaluations in each optimization run. 

Return:
* `poised_points :: Vector{T}` where `T` is either a `Vector{F}` or an `SVector{n_vars, F}` and `F` is the precision of the points in `points`, but at least `Float32`. 
* `lagrange_basis :: Vector{&lt;:AbstractPolynomialLike}`: The Lagrange basis corresponding to `poised_points`.
* `point_indices`: An array indicating which points from `points` are also in `poised_points`. A positive entry corresponds to the index of a poised point in `points`. If a poised point is new, then the entry is `-1`.
"""
function get_poised_set( basis, points :: AbstractArray{T} = Vector{Float32}[]; 
		solver = :LN_BOBYQA, max_solver_evals = -1 ) where {
		T &lt;: AbstractArray{&lt;:Real}
	}
	
	p = length(basis)
	@assert p &gt; 0 "`basis` must not be an empty array."

	vars = variables( basis[end] )
	n_vars = length(vars)
	@assert n_vars &gt; 0 "The number of variables must be positive."
	
	if max_solver_evals &lt; 0
		max_solver_evals = 2000 * n_vars
	end

	F = promote_type( eltype( T ), Float32 )
	P_type = n_vars &gt; 100 ? Vector{Vector{F}} : Vector{SVector{n_vars, F}}
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
		
		if l_max &gt; ZERO_TOL
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
            opt.max_objective = (x,g) -&gt; abs( new_basis[i](x) )
			
			x₀_tmp = [ rand(F, n_vars) for i = 1 : 50 * n_vars ]
            x₀ = x₀_tmp[argmax( abs.(new_basis[i].(x₀_tmp)) ) ] 
            
			_, ξ, ret = NLopt.optimize(opt, x₀)
			
			poised_points[i] = ξ
		end		
		
		new_basis = orthogonalize_polys( new_basis, poised_points[i], i )
	end
	
	return poised_points, new_basis, point_indices
end
			</code></pre>


<pre class='language-julia'><code class='language-julia'>"""
    make_set_lambda_poised( basis, points; 
        LAMBDA = 1.5, solver = :LN_BOBYQA, max_solver_evals = -1, max_loops = -1, skip_indices = [1,] )

Make the output of `get_poised_set` even better suited for interpolation.

Input:
* `basis`: A vector of polynomials constituting a Lagrange basis for the polynomial space.
* `points`: The vector of points belonging to the Lagrange basis.
* `LAMBDA :: Real &gt; 1`: Determines the quality of the interpolation. 
* `solver`: NLopt solver to use. Should be derivative-free.
* `max_solver_evals`: Maximum number of evaluations in each optimization run. 
* `max_loops`: Maximum number of loops that try to make the set Λ-poised.
* `skip_indices`: Inidices of points to discard last.

Return:
* `poised_points :: Vector{T}` where `T` is either a `Vector{F}` or an `SVector{n_vars, F}` and `F` is the precision of the points in `points`, but at least `Float32`. 
* `lagrange_basis :: Vector{&lt;:AbstractPolynomialLike}`: The Lagrange basis corresponding to `poised_points`.
* `point_indices`: An array indicating which points from `points` are also in `poised_points`. A positive entry corresponds to the index of a poised point in `points`. If a poised point is new, then the entry is `-1`.
"""
function make_set_lambda_poised( basis, points :: AbstractArray{T}; 
		LAMBDA :: Real = 1.5, solver = :LN_BOBYQA, max_solver_evals = -1,
		max_loops = -1, skip_indices = [1,] ) where {
		T &lt;: AbstractArray{&lt;:Real}
	}
	
	@assert length(basis) == length(points) "Polynomial array `basis` and point array `points` must have the same length."
	
	if length(points) &gt; 0
		n_vars = length(points[1])
		@assert n_vars &gt; 0 "The number of variables must be positive."
		
		F = promote_type( eltype( T ), Float32 )
		P_type = n_vars &gt; 100 ? Vector{Vector{F}} : Vector{SVector{n_vars, F}}

		if max_loops &lt; 0 
			max_loops = length(basis) * 100
		end

		if max_solver_evals &lt; 0
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
				opt.max_objective = (x,g) -&gt; abs( polyᵢ( x ) ) 

				x₀_tmp = [ rand(F, n_vars) for i = 1 : 50 * n_vars ]
				x₀ = x₀_tmp[argmax( abs.(new_basis[i].(x₀_tmp)) ) ] 

				abs_lᵢ, xᵢ, _ = NLopt.optimize(opt, x₀)

				if abs_lᵢ &gt; LAMBDA
					iₖ = i
					xₖ = xᵢ
					if i ∉ skip_indices
						# i is not prioritized we can brake here
						break
					end#if
				end#if
			end#for

			if iₖ &gt; 0
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
	
end</code></pre>



<div class="markdown"><p>For the sake of convenience we combine both functions:</p>
</div>

<pre class='language-julia'><code class='language-julia'>function get_lambda_poised_set( basis, points; solver1 = :LN_BOBYQA, solver2 = :LN_BOBYQA, max_solver_evals1 = -1, max_solver_evals2 = -1, LAMBDA = 1.5, max_lambda_loops = -1 )
	lagrange_points, lagrange_basis, lagrange_indices = get_poised_set( 
		basis, points; solver = solver1, max_solver_evals = max_solver_evals1 )
	lambda_points, lambda_basis, lambda_indices = make_set_lambda_poised( 
		lagrange_basis, lagrange_points; LAMBDA, max_loops = max_lambda_loops,
		solver = solver2, max_solver_evals = max_solver_evals2 )
	combined_indices = [ i &lt; 0 ? i : lagrange_indices[j] for (j,i) in enumerate( lambda_indices ) ]
	return lambda_points, lambda_basis, combined_indices
end</code></pre>
<pre id='var-get_lambda_poised_set' class='pre-class'><code class='code-output'>get_lambda_poised_set (generic function with 1 method)</code></pre>


<div class="markdown"><p>Let&#39;s have a look at what the points look like:</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin 
	basis = get_poly_basis(2,2)
	custom_points =  [ ones(Float32, 2), ones(Float32,2)] 
	
	#lambda_points, lambda_basis, c_indices = get_lambda_poised_set( basis,custom_points)
	lambda_points, lambda_basis, c_indices = get_poised_set( basis,custom_points)
	c_indices
end</code></pre>
<pre id='var-lambda_points' class='pre-class'><code class='code-output'>6-element Vector{Int64}:
  1
 -1
 -1
 -1
 -1
 -1</code></pre>

<pre class='language-julia'><code class='language-julia'>scatter(Tuple.(lambda_points))</code></pre>
<img src="data:image/png;base64,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IO/6flMN9tcicGUpKiqqrKzMaWBl8WskmzfIZs6ceeDAgf4r+/fvr6mpudQNWAAAV5VsAquhoaGpqWnfvn19D7u6urZs2dLQ0HA5z92zZ8/hw4fvvPPOLI4LAHBFyCawli1bVl9fv2LFipaWlra2tpUrVxYXF69atarvp6+++moqlVq3bl3fw6997Wvr1q07dOhQV1fX5s2b77///qlTpz7yyCNh/wIAgAKTTWAVFxdv3LixtrZ29uzZ06ZNO3HixNatWy/1zQtPPPHEtm3bFi5cmE6nH3jggfnz57/++uuVlZVDGxsAoHBleUfYpEmTzl+jusDSpUv7f6jw5ptv/uEPf5jdUQAArkS+BQoAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCFeV7AICc2N36wWsHjyVJ8vmbU/VTJuR7HODqIrCAkeb9ru5/+tc7Nx14/6OF/ffUXf/Xy37v+tJr8jkWcDXxFiEwomQyyf0//mW/ukqSJNl44P37f/zLTCZfQwFXHYEFjCjbmtu2/bp9gPVft29rbhv+eYCrk8ACRpTX3zl5qR/9/eGO4ZwEuJoJLGBE+YT3AXu9RwgMF4EFjCizpl7yA4O3+SwhMFwEFjCifPGWiQOG1KypE754y8Thnwe4OgksYEQZPSq1fsXnLmis26ZM+NsHPzd6VCpfUwFXG9+DBYw01elrdz42f9PbJ37RdCxJkjm1k/7glonqChhOAgsYgUaPSi2Zfv2cydckSVJWVpbvcYCrjrcIAQCCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACBYloF1/Pjx5cuXp9Pp0tLSJUuWNDY2Rm0GALjSZRNYZ8+eXbx48cGDB3fu3Hno0KGKiooFCxa0trYOfTMAwKD89mxvvkcYQDaB9dJLL7355pvPP/98dXV1RUXFc889193dvWbNmqFvBgC4HB2nz37z5d03/JuNtzzzxsR/vanh+dcbj3bme6j/L5vA2rBhQ21t7YwZM/oelpaWLlq0aP369UPfDADwqT7sPjf/B6/9h5+3HOk4nSRJx+mz/7Px2F3PbH/zvQ/yPdrv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">

<!-- PlutoStaticHTML.End -->
```