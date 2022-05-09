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

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "82012d577aa3f4c8a17e8ae42e5be8699cf7aae035a78d6f6e96e99bcbd777d2"
    julia_version = "1.7.2"
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
<pre id='var-non_negative_ineq_solutions' class='code-output documenter-example-output'>non_negative_ineq_solutions (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function get_poly_basis( deg, n_vars)
    exponents = non_negative_ineq_solutions(deg, n_vars )
    polys = let
        @polyvar x[1:n_vars]
        [ prod(x.^e) for e in exponents ]
    end
    return polys
end</code></pre>
<pre id='var-get_poly_basis' class='code-output documenter-example-output'>get_poly_basis (generic function with 1 method)</pre>


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
<pre id='var-orthogonalize_polys' class='code-output documenter-example-output'>orthogonalize_polys (generic function with 1 method)</pre>


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
<pre id='var-get_lambda_poised_set' class='code-output documenter-example-output'>get_lambda_poised_set (generic function with 1 method)</pre>


<div class="markdown"><p>Let&#39;s have a look at what the points look like:</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin 
    basis = get_poly_basis(2,2)
    custom_points =  [ ones(Float32, 2), ones(Float32,2)] 
    
    #lambda_points, lambda_basis, c_indices = get_lambda_poised_set( basis,custom_points)
    lambda_points, lambda_basis, c_indices = get_poised_set( basis,custom_points)
    c_indices
end</code></pre>
<pre id='var-lambda_points' class='code-output documenter-example-output'>6-element Vector{Int64}:
  1
 -1
 -1
 -1
 -1
 -1</pre>

<pre class='language-julia'><code class='language-julia'>scatter(Tuple.(lambda_points))</code></pre>
<img src="data:image/png;base64,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AAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIKNJLA6OzvXrFlTXl6eTqdXrlzZ1NT0UTtffvnl5IfdeOONo5gWAOAakHVg9ff3r1ix4siRI/v37z969GhFRcXSpUs7Ojo+5iavv/565rdOnjw5imkBAK4BWQfWCy+80NjY+OMf/7i6urqiouKZZ57p6+vbuHFjLoYDALgWZR1Y27Ztq62tnTdv3uDFdDq9fPnyrVu3Rg8GAHCtyjqwmpqa5syZM3Rl7ty5ra2tvb29H3WTZcuWpVKpysrKtWvXtre3j2RMAIBrR9aBdfr06WnTpg1dKSsry2QyZ8+evXLz5MmTH3/88T179pw+fXrz5s179uxZsmRJV1fXyOcFACh4JdneIJPJfOLKJfX19fX19YP/ff/997/44osLFizYtGnT9773vaHbNmzY8J3vfGfoyn333bdo0aJjx45lO97V6+/vP3PmTH9/f+4OAeTdO++8k+8RgNzq7Ox87733UqlU7g7R3d192dmlT5R1YJWXl3d3d1921GQyWVZW9om3raurmzVr1t69ey9b37Bhw4YNGy5bSSQSs2fPzna8q9fX13f99ddXVlbm7hBAIcjpIwmQd6lUqqKiYtKkSbk7RLZ1lRjBS4Tz589/8803h640NzfX1NSUlpZm+0cBABSlrAOroaGhpaXl8OHDgxd7enp27drV0NBwNbc9ePDg8ePHFy9enO1BAQCuIVkH1urVq+vq6tatW9fW1nbq1KkHH3wwlUqtX79+8Nrt27cnk8ktW7YMXvz617++ZcuWo0eP9vT07Ny580tf+tLMmTMffvjhyL8BAECByTqwUqnUjh07amtrFy1aNHv27JMnT+7evbuqqmrYzY899tgrr7yybNmy8vLyBx544N577923b9/06dNHPTYAQOHK+k3uiURixowZl85RXWbVqlVDP1R42223/ehHPxrhaAAA16aR/LJnAAA+hsACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAgmsAAAggksAIBgAgsAIJjAAgAIJrAAAIIJLACAYAILACCYwAIACCawAACCCSwAgGACCwAgmMACAAhWku8BAOINXMxs+Ze3X2ruaj5xZu7NJ1fdPn31Z6omTkjmey5g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yba2to/fXyDP5sUcWP39/StWrDhy5Mj+/fuPHj1aUVGxdOnSjo6O0W8GCsQI7rmvv/565rdOnjw5ZqMCI7N3794nnniivr5+/fr1n7i5gJ7NM8Xr2WefTSQShw4dGrx47ty5qVOnPvLII6PfDBSIrO65O3bsSHw4sIBryPe///1EItHa2voxewrn2byYz2Bt27attrZ23rx5gxfT6fTy5cu3bt06+s1AgXDPBYYqnMeEYg6spqamOXPmDF2ZO3dua2trb2/vKDcDBWIE99xly5alUqnKysq1a9e2t7fnfkZg7BTOs3kxB9bp06enTZs2dKWsrCyTyZw9e3aUm4ECkdU9d/LkyY8//viePXtOnz69efPmPXv2LFmypKura6yGBXKucJ7NizmwMlf8FqArV0a2GSgQWd1z6+vrv/vd786dO3fKlCn333//iy++2N7evmnTphzPCIydwnk2L+bAKi8v7+7uHrrS3d2dTCbLyspGuRkoEKO559bV1c2aNWvv3r05mw4Ya4XzbF7MgTV//vw333xz6Epzc3NNTU1paekoNwMFwj0XGKpwHhOKObAaGhpaWloOHz48eLGnp2fXrl0NDQ2j3wwUiNHccw8ePHj8+PHFixfnckBgTBXQs/nYfzPEmOnr66urq1uyZElra+vJkye//OUvV1RUvP3224PX/uIXv0gkEs8999zVbAYKU1Z386997WvPPfdcW1vbuXPnXn755Tlz5lRVVb377rv5Gx/IwrDfg1Wwz+bFfAYrlUrt2LGjtrZ20aJFs2fPPnny5O7du6uqqka/GSgQWd1zH3vssVdeeWXZsmXl5eUPPPDAvffeu2/fvunTp4/xzEBWLly4MPi7rR555JFEIlFTU5NMJr/whS8Mu7lwns2TGZ+VAwAIVcxnsAAA8kJgAQAEE1gAAMEEFgBAMIEFABBMYAEABBNYAADBBBYAQDCBBQAQTGABAAQTWAAAwQQWAEAwgQUAEExgAQAEE1gAAMEEFgBAMIEFABBMYAEABPv/1YDQZy261kMAAAAASUVORK5CYII=">

<!-- PlutoStaticHTML.End -->
```

