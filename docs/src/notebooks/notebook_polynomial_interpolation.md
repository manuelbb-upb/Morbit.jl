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
<img src="data:image/png;base64,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Vk6ZPMIz25TJk9bdWTn28wATk8ACxpXby6Y9+ek7r15/8tN33l42beznASYmX9MAjDebV9V+pOKWp//+N4dau5IkWVxV9ug9H7mvfm6h5wImEIEFjEP31c+9r37u+fNdSZKUl5cVehxgwhFYwLiVShV6AmCi8hksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIKVFHqAsfbbSwN/9fJv9p04d+qdC3VzW/9l/dwNf3J7KlXosQCAcWRiBdaZ7t6V333l1+cuDB3+uuPMjw+f+cmRs/99wx9PElkAQJCJ9RbhozsbL9fVZX9z6PSz//hWQeYBAMalCRRYvQOD/+v1thFv+tEvW8d4GABgHJtAgdXe03tpYHDEm948f3GMhwEAxrEJFFjl09PX+qBV5pYpYzwMADCOTaDAmjm15O6azIg3/Ys7K8d4GABgHJtAgZUkydOfr5sxZfIVi3VzZj60oqYg8wAA49LECqw/ub1s/9fuXrvwQ7dMmZwkScWMKV9ZNv/nf7G8dOrE+roKACCvJlxYLKma9X82Levr7z9x6szHPnJHoccBAMahiXUF67JJqVT5tCvfKwQACDFBAwsAIH8EFgBAMIEFABBMYAEABBNYAADBBBYAQDCBBQAQTGABAAQTWAAAwQQWAEAwgQUAEExgAQAEE1gAAMEEFgBAMIEFABAsx8Bqb2/fsGFDJpMpLS1dt25dY2PjtXa+9NJLqT9022235TotAMBNIJfA6u/vX7t27YkTJw4ePHjy5MmKiopVq1a1tbW9z11ef/317O+dO3cu12kBAG4CuQTWc889d+jQoR/84AfV1dUVFRXbt2/v6+vbunVr+HAAADejXAJr165dtbW1ixYtGjosLS1ds2bNzp07QwcDALhZ5RJYjY2NCxYsGL6ycOHC5ubmixcvXusuq1evTqfTc+fO3bhxY2traw4nBQC4WeQSWJ2dnWVlZcNXysvLs9ns+fPnr948derUJ554Yv/+/Z2dnc8888z+/fuXLVv29ttv5zgvAEDRK8nhPtls9gNXLluxYsWKFSuG/vnee+99/vnnFy9evG3btieffHJoccuWLd/4xjeG3+Wee+5ZunTpm2++mcNs12lgYKCjo2NgYCB/pwAKrru7O0mSrq6uQg8C5NHZs2cvXbpUUpJL0lynrq6uKy4tfaBcpslkMlc8YXV1daVSqfLy8g+8b319/R133HHgwIHLK1u2bNmyZcvwPUOH8+fPz2G26zQwMDBt2rR58+bl7xRAwQ09U432aRG4uZSUlFRWVuY1sHJ4GsnlLcK6urrjx48PXzl27FhNTc306dNz+LcBAIwzuQRWQ0NDU1PT0aNHhw57enr27t3b0NBwPfc9fPjwqVOn7rrrrhzOCwBwU8glsO6///76+voHH3ywpaWlo6Nj06ZN6XR68+bNQ7e++OKLqVRqx44dQ4df/vKXd+zYcfLkyZ6enj179nzhC1+YN2/eww8/HPZfAABQZHIJrHQ6vXv37tra2qVLl86fP//cuXP79u2rqqoacfPjjz/+8ssvr169OpPJPPDAAytXrnz11VcrKytvbGwAgOKV4yfCZs+effka1RXWr18//IcKP/rRj37/+9/P7SwAADejHH/ZMwAA1yKwAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAiWY2C1t7dv2LAhk8mUlpauW7eusbExajMAwM0ul8Dq7+9fu3btiRMnDh48ePLkyYqKilWrVrW1td34ZsbGhb73Cj0CAMR4t3+w0COMoCSH+zz33HOHDh06cuRIdXV1kiTbt2+vqqraunXr008/fYObyau23176tz9946dHz77d01cxY8raBR/6T59ZNP/W6YWeCwBGretS/7//6Rs/PnymtetS2bT0ytrMf/z0oro5Mws91+/kcgVr165dtbW1ixYtGjosLS1ds2bNzp07b3wz+XO2u3fZd/b/9S9Ovd3TlyRJx4W+H/2y9RN/9fKb71ws9GgAMDoX+t5b+d3/+19eaWntupQkSdel/v/dePaT39l/6PRvCz3a7+QSWI2NjQsWLBi+snDhwubm5osXR3ipHtVm8uebu49f3VJnunufeOGNgswDADn7z/ubf9V2ZUt19w48urNYPuedy1uEnZ2dS5cuHb5SXl6ezWbPnz8/ffqV7zd94OYLFy709PQM39Df319SUvLee3n8nNB7v5e/UxSbvz3aPuL6T99on1B/DkwoQ3+3/Q2H8ecnjWdGXN/X1PHbi30zpkyOPV02m02lUqO6Sy6Blc1mP3Dl+jd/73vf27p16/CVxYsX19fXnzkz8p9diIGBgY6OjkmTJtC3VJzr6R1xvfPdvtNtbZNG+fcGbgrd3d1JkrheDuNPe/fIj+vBbPb4m6fnlqZjT9fT0zNz5ug+3ZVLYGUyma6uruErXV1dqVSqvLw8h82bN2/evHnz8A1btmxJkqSqqiqH2a7TwMBAOp2eN29e/k5RbO649fjRsz1Xr1fNmnbH7beP/TwwBoaefMrKygo9CBCsuuLUrzsuXb0+tWRSfe38qSXBF1BGW1dJbp/BqqurO378+PCVY8eO1dTUXP3+4Gg3kz9/vmTkYP1XSyZQZQIwPvz5NV68Pv9Hc8LrKje5DNHQ0NDU1HT06NGhw56enr179zY0NNz4ZvLn36yq/VR15orFpVVl/2HtghH3A0DReuCf3vH5P5pzxWJN5panG+oKMs/Vcgms+++/v76+/sEHH2xpaeno6Ni0aVM6nb78Nt+LL76YSqV27NhxPZsZM7dMmbz3q5/81uc+9s8XfOgjFbes+ehtT376zle+vrxsWvAb1QCQb5MnpZ5/4E+3/9nH199Z+eGyqcurb/13/+yf/L9HV1aVTSv0aL+Ty2ew0un07t27H3vssaVLl/b19S1fvnzfvn3X+sjUqDaTV1NLJj22qvaxVbWFHgQAbtSkVOoryz78lWUfPn36dGVlZUlJLkmTPzlOM3v27MvXqK6wfv36K35O8H02AwCMP0XxQTAAgPFEYAEABBNYAADBBBYAQDCBBQAQTGABAAQTWAAAwQQWAEAwgQUAEExgAQAEE1gAAMEEFgBAMIEFABBMYAEABBNYAADBBBYAQDCBBQAQrKTQA4ygpaWlpaVly5Yt+TvF4ODghQsXZs6cmb9TAAXX29ubJMnUqVMLPQiQR93d3TNmzJg0KY/XjPbt21ddXT2quxTjFawlS5aM9j9jtPr6+o4cOZLXUwAF19bW1tbWVugpgPw6cuRIX19fXk9RXV29ZMmSUd0llc1m8zRNMWtqalq7dm1TU1OhBwHyaOhCeF4vhwMFV1tb+7Of/ay2trbQg/yBYryCBQBwUxNYAADBBBYAQDCBBQAQrBi/pmEMZDKZhx56qNBTAPm1atWqQo8A5N1DDz2UyWQKPcWVJuhPEQIA5I+3CAEAggksAIBgAgsAIJjAAgAINp4Dq729fcOGDZlMprS0dN26dY2NjVGbgeJx/Q/el156KfWHbrvttrEcFcjB4ODgnj17Nm7cOGvWrFQq1dLS8v77i+QFfdwGVn9//9q1a0+cOHHw4MGTJ09WVFSsWrXqWr/2dVSbgeKRw4P39ddfz/7euXPnxmxUIDcHDhx46qmnVqxYsXnz5g/cXEQv6Nlx6oc//GGSJEeOHBk67O7unjVr1iOPPHLjm4HiMaoH7+7du5M/DCzgJvLtb387SZLm5ub32VM8L+jj9grWrl27amtrFy1aNHRYWlq6Zs2anTt33vhmoHh48ALDFc9zwrgNrMbGxgULFgxfWbhwYXNz88WLF29wM1A8cnjwrl69Op1Oz507d+PGja2trfmfERg7xfOCPm4Dq7Ozs6ysbPhKeXl5Nps9f/78DW4GiseoHrxTp0594okn9u/f39nZ+cwzz+zfv3/ZsmVvv/32WA0L5F3xvKCP28DKXvUrgK5eyW0zUDxG9eBdsWLFN7/5zYULF86cOfPee+99/vnnW1tbt23blucZgbFTPC/o4zawMplMV1fX8JWurq5UKlVeXn6Dm4HicSMP3vr6+jvuuOPAgQN5mw4Ya8Xzgj5uA6uuru748ePDV44dO1ZTUzN9+vQb3AwUDw9eYLjieU4Yt4HV0NDQ1NR09OjRocOenp69e/c2NDTc+GageNzIg/fw4cOnTp2666678jkgMKaK6AV97L8ZYmz09fXV19cvW7asubn53LlzX/ziFysqKt56662hW1944YUkSZ599tnr2QwUrVE90r/0pS89++yzLS0t3d3dL7300oIFC6qqqs6ePVu48YFRGPF7sIr2BX3cXsFKp9O7d++ura1dunTp/Pnzz507t2/fvqqqqhvfDBSPUT14H3/88Zdffnn16tWZTOaBBx5YuXLlq6++WllZOcYzA6MyMDAw9LutHnnkkSRJampqUqnUZz/72RE3F88Leirrx+UAAEKN2ytYAACFIrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACC/X8OL+Wp9QCffQAAAABJRU5ErkJggg==">

<!-- PlutoStaticHTML.End -->