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
    input_sha = "6c3fb72456e239d8a9c944698d441b1e12fa6cf347e2e5c24860397edd51f824"
    julia_version = "1.6.1"
-->
<pre class='language-julia'><code class='language-julia'>begin
	import Pkg
	Pkg.activate(tempname())
	Pkg.add("DynamicPolynomials")
	Pkg.add("NLopt")
	Pkg.add("StaticArrays")
	Pkg.add("GLMakie")
	Pkg.add("Makie")
	Pkg.add("Combinatorics")
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
	using DynamicPolynomials
	using BenchmarkTools
	import NLopt
	using StaticArrays
	using Makie, GLMakie
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
<img src="data:image/png;base64,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Ye3ata2trcViMQAAskoBAEBWKQAAyCoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySgEAQFYpAADIKgUAAFmlAAAgqxQAAGSVAgCArFIAAJBVCgAAskoBAEBWKQAAyCoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySgEAQFYpAADIKgUAAFmlAAAgqxQAAGSVAgCArFIAAJBVCgAAskoBAEBWKQAAyCoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySgEAQFYpAADIKgUAAFmlAAAgqxQAAGSVAgCArFIAAJBVCgAAskoBAEBWKQAAyCoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySgEAQFYpAADIKgUAAFmlAAAgqxQAAGSVAgCArFIAAJBVCgAAskoBAEBWKf4HKpVKuVwulUoBAMCAFH+V3t7eG2644bvf/e6qVat23333s88++8ADD4xX8f73v3/Tpk3xCgcffPCZZ54ZAADVKMVf5aKLLrryyiv//u//fu7cud///vePP/747373u7NmzYqtefrpp9evX9/U1BQDDjzwwAAAqFIpBu+xxx675pprzj777M997nMRcdJJJx155JGXXXbZ9773vUKhEFuzyy673HLLLYVCIQAAql2KwZs/f353d/e73/3u6DdhwoS3v/3t119//eLFi6dMmRKvolAoBADA60CKwXv00UdbW1unTJkSA/bbb7/LL7/8+eefnzJlSmxNX1/fhz70oe7u7hkzZrzjHe/YZZddAgCgSqUYvBUrVjQ1NTU2NsaAlpaWSqXS3t4er+KJJ54YP358Z2fnHXfccdVVV1133XVvfetbo9+iRYtiC+Vyube3N7aZ3gHFYjGAKtXb2xsRvb29AVSpcrnc269UKsU2Uy6Xi8ViDEaKwevq6oqIQqEQAxoaGgqFQryKiy++eP/9929oaOjr63vooYdOP/30884776CDDmppaYmI008/PTY3Z86cjRs3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AGSVAgCArFIAAJBVCgAAskoBAEBWKQAAyCoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySgEAQFYpAADIKgUAAFmlAAAgqxT/Axs3bkwp1dbWxl+gq6urUqkMGzYsAACqWoq/yv3333/FFVc88cQTDQ0Nc+fOPe+880aOHBmvYvHixZdddtkvf/nLvr6+/fbb7/zzz99jjz0CAKBKpRi8hx9++OSTT959990vuOCCpUuXXn755UuWLLnppptKpVJs4eWXX37f+963ZMmSc845p7a29rrrrnvPe97zox/9aIcddoi/kUollq3rXra6q9iwqXV4XQAAr02rN256dnXX+EL9jq11hUIMHSkG78orr6yrq7vhhhsmT54cES0tLR/96EdPO+20I444IrZw2223/eY3v7n99tuPPvroiNh3332PPPLI66+//uKLL46/hUeWrv38gmfve6Z9fXfvuBELj99r/D/PmTp2eF0AAK8dK9Z3f2HBs7c/uuylju7hdemwaa3nzpm67w7NMTSkGKSVK1cuWLDgmGOOmTx5cvR761vfesEFF9x9991HHHFEbOHOO++cNm3anDlzot++/e65555PfvKTNTU1sX09vOTl93zrkYXtG6Lfs6s2fv6+Z59cvv6b79ln5LCaAABeC17u3PQPt/3hh08sj34bevpu/a8//27Jy/9xyr5vnDQyhoAUg/Tiiy+uWrVq5syZMaCtrW3ixImPPfZYbKGzs/PZZ5/dddddhw0bFv2KxeLMmTNvvfXW1atXt7W1xXZUrlQu+9kzC9s3xOZ++MTyG3675GOH7hwAwGvBDb9d8sMnlsfmFrZvuOxnz3z71P2KhUL8raUYpFWrVvX29ra1tcWAlFJLS0t7e3tfX1+pVIpX2Lhx49q1a1tbW+MVRo0atX79+o0bN0a/Rx99NLbQ19fX09MTWS1e07ngmfbYmjsfX3bWgROLhUIAVaSnpycienp6Aqgi5UrlzseXxdYseKb92RXrdmwZFln19fWVSqUYjBSDVOkXr1AoFIrFYryK3t7e2Fx9fX28wjnnnBObmz17dmdn55o1ayKrJSs2dnT3xtas7OhauWpNbakQQBVZt25dRJTL5QCqSE9fZWVHV2xNR3fvkhWrm6Ihsurs7Bw+fHgMRopBamxsLBaLHR0dMaBcLm/cuLGhoaFYLMbmamtrGxsbOzo64hXWrVtXU1NTW1sb/W677bbY3DXXXNPY2DhmzJjIape6rtbGZ15c1xVbmDJ6+IS2sYVCANWktrY2IpqbmwOoIpVKTBm9+PEVG2MLrY11u0xqG9NUH1k1NjbGIKUYpLa2toaGhueffz4GdHR0vPTSS4ccckihUIjNNTQ0TJgw4cUXXyyXy8ViMfo988wzbW1tzc3N0a+1tTW2UCgUisViZLXDyIa3zWj7918vjs0VCnHC3hNKpWIA1aVYLEZEsVgMoLqcsPeEu/60olKJ/5+3zWjbYWRD5FYoFGKQUgzSDjvssMcee/ziF7/o6empra2NiN///vfLli1785vfHP06Ojoqlcrw4cOLxWKpVDrooINuuOGGZ599dvr06RHR3t7+0EMPHXTQQcOHD4/t7l/eMu1PK9bf/9yqGFAoxAdm7fjefScGAPAa8d59J/7mhTVffXBxpRL/n9k7j/6Xt0yLoSHFINXU1Jx55plnn332l7/85dNOO23lypWf+cxnpk6d+va3vz36zZs37/bbb7/77rt33333iDjllFNuvvnmiy+++JJLLqmpqbn88svb29vPOOOM+FuYMqrh1r/f9ysPLr77yeWrN3RPGT383ftMPPWNO9SUigEAvEbUlIpfOnbmfjuMvPW//rxo1fpRjXVH7972wVk7jm+qj6EhxeCdeuqpTz311CWXXHL55Zd3dnZOmDDh6quvHjduXAxYs2ZNDNh7772/+MUvXnDBBW984xuLxWKhUPjsZz97+OGHx9/I+Kb6i/9u13PfvOPK1WsmjR+bSqUAAF5rakvFM980+bQ3TlyybMWYUS3DG+pjKEkxeKVS6bLLLjvttNOeeeaZhoaGvffee/To0THgQx/60Ac/+MHx48fHgFNOOeXwww//4x//WC6Xd9ttt5122in+1mpKhcbaUrFQCADgNatYKDTWlmpKhRhiUvy1dusXW5g+fXpsYUK/AAB4HUgBAEBWKQAAyCoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySjH0LFiwILax3t7e7u7uhoaGQqEQQJXq6uqKiPr6+gCqVKVS2bhxY11dXUoptpkFCxbMmTMnBiPF61LqF0BVq6+vD6CqFQqFxsbGGHpSDD333XdfbGO//e1vP/nJT/7nf/5nc3NzAFXq4osvjoh58+YFUKXWrl174oknfuYznznggANim5k3b14MUgoAALJKAQBAVikAAMgqBQAAWaUAACCrFK9LBxxwwPz58wOoavPmzQugqjU3N8+fPz+GnhQAAGSVAgCArFIAAJBVimq3bt267u7u5ubm2tra+O90dnauX79+2LBhw4cPD+C14+WXX960aVNLS0tKKV5duVyOzRWLxQBeIyqVSqFQiL/Ahg0bNm7c2NjY2NDQEH8LKapXe3v7pz71qR/96EednZ2TJk366Ec/evLJJ8erKJfLV1111Y033rh8+fKWlpaTTz75/PPPr62tDWBoe+GFFy666KKf/exnmzZtmjp16nnnnffWt741tqa7u/uoo47q6uqKV7jooouOPvroAIaq9vb2+fPnP/LII48++mhHR8eHP/zhk08+OV5dT0/P5z//+VtuuWX16tWtra1nnHHGRz7ykVKpFNtXiipVLpfPOeecH//4xxdddNH06dP/4z/+44Mf/OCIESPe9ra3xdZce+21//Zv/3b22WcfffTRDz744Oc+97lNmzZ9+tOfDmAI6+zs/MAHPvDkk09efPHFY8eOvf76608//fTvf//7s2bNiq1Zvnz5AQccMHr06BhQKpUCGMLWrFnzT//0T6NHj542bdpvfvOb+O98rt+55547e/bse++998ILL4yIj33sY7F9pahSDzzwwHe/+91LL730rLPOiogDDzzwySefvOKKK+bOnZtSis2tXr36y1/+8jHHHHPppZeWSqXDDjtsxYoV//7v/37GGWfstNNOAQxVd999909/+tNvfetbJ510UkTss88+s2fP/tKXvjRr1qx4FRMnTrz00ksDeI1oaWl54IEHJk2a9OCDD86fPz/+n5YsWfKVr3zlPe95z7x58yJizpw5L7zwwtVXX33KKaeMHTs2tqMUVeree++tqak55phjot+IESOOPvroyy+/fNGiRdOmTYvN/eEPf1i0aNEFF1xQKpWi3zve8Y7rrrvu17/+9U477RTAUHXXXXe1tbUddthh0W/cuHGHHnroPffc097e3traGq+ir6+vUCgUi8UAhrzWfhFRKBTiv/PQQw+tWLHi2GOPjX6FQuHYY4+97bbbHnnkkblz58Z2lKJKPfnkk2PGjBk/fnwM2GOPPTo7O1944YVp06bF5p5++umI2G233WLA1KlT6+vrH3/88QCGqkqlsnDhwokTJ44ePToG7Lnnnt/61reWLVvW2toaW/P000/Pnj27UCjMnDnzfe9736xZswKoFk8++WRdXd20adNiwPTp0wuFwtNPPz137tzYjlJUqZUrVzY0NNTX18eAlpaWSqWydu3a2MKKFSuKxWJzc3MMaGxsbGhoWLVqVQBD1aZNm1avXt3W1lYqlWJAa2vrpk2bOjo6YmtSzpwpAAAFQElEQVRGjBhRW1t78MEHr1q16u677/7Od75z/fXXH3fccQFUheXLl9fU1IwYMSIGjBgxoq6ubtWqVbF9pahSfX19sbmamppCoRBbU6lUYnOlUimAIa+vry82V1NTE6+irq7uzjvvHDduXPR74oknTjzxxE996lNHHXVUY2NjAK99fX19sbmUUvwtpKhSo0aNWrZs2aZNm1JK0W/NmjWVSmX48OGxhZaWlnK5vGHDhhiwYcOGrq6upqamAIaqlNLIkSM3btxYLpeLxWL0W7VqValUamhoiK0ZN25cDJgxY8a73/3uSy65ZPHixTNmzAjgta+1tbW3t7ezszMGdHR09PT0NDU1xfaVokpNnz79/vvvX7ly5eTJk6PfwoULa2trJ0yYEFuYOnVqpVJ5/vnn3/CGN0S/pUuXdnV1TZ8+PYChqlgsTpky5de//vW6detGjhwZ/f70pz+NHDly7Nix8RcYNWpUX19fd3d3AFVhl1126e7ufuGFF3beeefot3jx4r6+vp133jm2rxRVavbs2VddddUvfvGL9773vRHR29v7k5/8ZPfdd995550jore3d/ny5RExceLEiNh7773HjBnz4x//+Ljjjot+P/nJT2pqat70pjcFMIS95S1v+c53vvPQQw8deeSREbF+/fr7779/n332GTduXER0dnauXr26UChMmDAhIrq6uurr62NAT0/P/PnzW1tbx40bF8BrU1dX16pVqyJi4sSJEbHffvs1NjbOnz9/zpw50e+ee+5paWnZZ599YvtKUaUOP/zwgw8++LOf/eyECRN22mmn22677Ze//OWVV145bNiwiFi+fPn+++9/5JFH3nTTTRExceLEU0899dprrz3wwAPf8pa3PPzww9dee+0JJ5wwc+bMAIawd7zjHddcc80FF1zQ3Nw8evTor33ta0899dS8efOKxWJELFy48LDDDnv/+99/6aWXRsStt966cOHC2bNnt7W1rVy58pZbbrnrrrvOOeec8ePHBzBU9fX1feMb34iITZs2VSqVF1988etf/3pEnHzyyQ0NDYsWLTrkkENOPfXUyy+/PCJ22223k0466frrr3/DG94wa9asn//85zfffPPpp58+ZcqU2L5SVKmGhoYvf/nLH/nIR0444YRSv49//ONnnHFGDCiVSvEK//qv/7p69epzzz23UCj09fUdeeSRl112WaFQCGAIGzNmzDXXXPPxj3/8mGOOKZVKtbW1n/nMZ975znfGgFKpFAOKxeJtt912zTXXlMvliGhtbf2XfgEMYeVy+eqrr37ppZcioq2t7Qtf+EJEjBw58l3velf0K5VK8Qqf/vSn169ff9ZZZxWLxXK5fOyxx1544YWx3aWoXjNmzLjrrrueeuqpDRs2TJgwYdKkSTFg3LhxDz/8cLzCiBEjvvrVr37iE59ob28fOXLk9OnTi8ViAEPerFmz7r333qeffrqrq2vy5Mnjx4+PAbvuuusf//jHQqEQ/U499dR3vvOdy5YtW79+fW1t7Q477DBq1KgAhraampoFCxbEFkaMGBERO++886OPPhqvMGbMmG9+85sLFy5cs2bN6NGjp06dWigUYrtLUdVqampmzpwZWyiVSm1tbbGFqf0CeE0ZNmzY3nvvHVuoq6tra2uLV2juF8BrSnNzc7yK2tratra22FyhUNhll13ibyoFAABZpQAAIKsUAABklQIAgKxSAACQVQoAALJKAQBAVikAAMgqBQAAWaUAACCrFAAAZJUCAICsUgAAkFUKAACySgEAQFYpAADIKgUAAFn9b4xRNLitypArAAAAAElFTkSuQmCC">

<!-- PlutoStaticHTML.End -->
```