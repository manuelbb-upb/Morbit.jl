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
    input_sha = "f31943618e6ab92ed28034eb7cac5ddb44d3a046b3ba415874adf44fc9520e22"
    julia_version = "1.7.2"
-->

<div class="markdown"><h1>Finite Differences for Taylor Models.</h1>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    using StaticArrays
    using Symbolics
    using Symbolics: Sym
    using Parameters
    using BenchmarkTools
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    const RVec = AbstractVector{&lt;:Real}
    const RVecOrR = Union{Real, RVec}
end</code></pre>
<pre id='var-RVecOrR' class='code-output documenter-example-output'>Union{Real, AbstractVector{<:Real}}</pre>


<div class="markdown"><h3>Abstract</h3>
<p>For calculating finite difference gradients and hessians in Morbit we want to separate the grid construction from the function evaluation and derivative calculations. To my knowledge, this is not easily possible using <code>FiniteDiff</code> or <code>FiniteDifferences.jl</code>.</p>
<p>This notebook explores a recursive calculation procedure starting from first order finite difference rules &#40;interfaced by <code>FiniteDiffStamp</code>&#41;. To calculate partial derivatives of <span class="tex">$f:ℝ^n \to ℝ^k$</span> at a fixed point <span class="tex">$x_0 ∈ ℝ^n$</span> a <code>DiffWrapper</code> can be used.  It is initialized devoid of any function evaluation information but only has symbolic representations of required evaluation sites relative to <span class="tex">$x_0$</span>. A <code>DiffWrapper</code> can hence be used to first construct a grid for <span class="tex">$x_0$</span> and at a later stage to approximate gradients or Hessians.</p>
</div>


<div class="markdown"><h2>Finite Difference Rules</h2>
<p>A finite difference rule is defined by a grid of integer offsets, a decision space stepsize <span class="tex">$h$</span>, and coefficients to combine evaluation results at the grid points.</p>
<p>For details, please see <a href="https://en.wikipedia.org/wiki/Finite_difference_coefficient">this Wikipedia page</a>.</p>
<p>We abstract the basic information with the <code>FiniteDiffStamp</code> interface:</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    abstract type FiniteDiffStamp end
    
    # integer offsets to build a grid along some variable dimension
    _grid( :: FiniteDiffStamp ) = nothing
    # stepsize to build a grid along some variable dimension in decision space
    _stepsize( :: FiniteDiffStamp ) = nothing

    # coefficients of the rule
    _coeff( :: FiniteDiffStamp ) = nothing
    
    # order of the method 
    _order( :: FiniteDiffStamp ) = nothing
    
    # accuracy (exponent of residual in Taylor expansion)
    _accuracy( :: FiniteDiffStamp ) = nothing
    
    # zero index of grid
    _zi( :: FiniteDiffStamp ) = nothing
    
    Base.broadcastable( fds :: FiniteDiffStamp ) = Ref(fds)
    
    # evaluation of rule at sites in `X` (a vector or an iterable)
    function (stamp :: FiniteDiffStamp)(X)	
        coeff = _coeff(stamp)
        @assert length(X) == length(coeff)
        h = _stepsize(stamp)
        return sum( c * x for (c,x) in zip(coeff,X) ) ./ h^(_order(stamp))
    end
end</code></pre>



<div class="markdown"><p>We could already define a finite difference function using a <code>FiniteDiffStamp</code>:</p>
</div>

<pre class='language-julia'><code class='language-julia'>"Evaluate all outputs of `f` with respect to ``x_i``." 
function dx(stamp :: FiniteDiffStamp, f :: Function, x0 :: RVec, i = 1)
    h = _stepsize(stamp)
    grid = [ [x0[1:i-1];x0[i] + h*g; x0[i+1:end]] for g in _grid(stamp) ]
    evals = f.(grid)
    return stamp( evals )
end</code></pre>



<div class="markdown"><p>Of course, some concrete implementations are still missing.  We provide some entries from the table in <a href="https://en.wikipedia.org/wiki/Finite_difference_coefficient">Wikipedia</a>.</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    stepsize( x :: F ) where F &lt;: AbstractFloat = 10 * sqrt(eps(F))
    stepsize( F :: Type{&lt;:AbstractFloat} ) = 10 * sqrt(eps(F))
    stepsize( x :: AbstractVector{F} ) where F&lt;:AbstractFloat = stepsize(F)
    stepsize( x ) = stepsize(Float64)
end</code></pre>
<pre id='var-stepsize' class='code-output documenter-example-output'>stepsize (generic function with 4 methods)</pre>

<pre class='language-julia'><code class='language-julia'>begin
    struct CFDStamp{N, F &lt;: AbstractFloat} &lt;: FiniteDiffStamp
        grid :: SVector{N,Int8}
        coeff :: SVector{N,Rational{Int16}}
        order :: UInt8 	
        accuracy :: UInt8
        
        stepsize :: F
        
        zero_index :: UInt8
    end
    
    _grid( stamp :: CFDStamp ) = stamp.grid
    _coeff( stamp :: CFDStamp ) = stamp.coeff 
    _order( stamp :: CFDStamp ) = stamp.order
    _accuracy( stamp :: CFDStamp ) = stamp.accuracy
    _stepsize( stamp :: CFDStamp ) = stamp.stepsize
    _zi( stamp :: CFDStamp ) = stamp.zero_index
    
    function CFDStamp( order :: Int, accuracy :: Int, _stepsize_ = stepsize(Float64) ) 
        return CFDStamp( Val(order), Val( floor(Int, accuracy / 2) * 2 ), _stepsize_ )
    end
    
    function CFDStamp( ::Val{1}, ::Val{2}, _stepsize_ :: AbstractFloat )
        CFDStamp( 
            SVector{3,Int8}([-1, 0, 1]), 
            SVector{3,Rational{Int16}}([-1//2, 0, 1//2]),
            UInt8(1), UInt8(2), _stepsize_, UInt8(2),
        )
    end
    
    function CFDStamp( ::Val{1}, ::Val{4}, _stepsize_ :: AbstractFloat )
        CFDStamp( 
            SVector{5,Int8}([-2,-1,0,1,2]), 
            SVector{5,Rational{Int16}}([1//12, -2//3, 0, 2//3, -1//12]), 
            UInt8(1), UInt8(4), _stepsize_, UInt8(3)
        )
    end
    
    function CFDStamp( ::Val{1}, ::Val{6}, _stepsize_ :: AbstractFloat )
        CFDStamp( 
            SVector{7,Int8}(collect(-3:1:3)), 
            SVector{7,Rational{Int16}}([-1//60, 3//20, -3//4, 0, 3//4, -3//20, 1//60]), 
            UInt8(1), UInt8(6), _stepsize_, UInt8(4)
        )
    end
    
    
end</code></pre>
<pre id='var-_zi' class='code-output documenter-example-output'>CFDStamp</pre>

<pre class='language-julia'><code class='language-julia'>begin
    struct FFDStamp{N, F &lt;: AbstractFloat} &lt;: FiniteDiffStamp
        grid :: SVector{N,Int8}
        coeff :: SVector{N,Rational{Int16}}
        order :: UInt8 	
        accuracy :: UInt8
        
        stepsize :: F
    end
    
    _grid( stamp :: FFDStamp ) = stamp.grid
    _coeff( stamp :: FFDStamp ) = stamp.coeff 
    _order( stamp :: FFDStamp ) = stamp.order
    _accuracy( stamp :: FFDStamp ) = stamp.accuracy
    _stepsize( stamp :: FFDStamp ) = stamp.stepsize
    _zi( stamp :: FFDStamp ) = 1
    
    function FFDStamp( order :: Int, accuracy :: Int, _stepsize_ = stepsize(Float64) ) 
        return FFDStamp( Val(order), Val( ceil(Int, accuracy / 2) ), _stepsize_ )
    end
    
    function FFDStamp( ::Val{1}, ::Val{1}, _stepsize_ :: AbstractFloat )
        FFDStamp( 
            SVector{2,Int8}([0, 1]), 
            SVector{2,Rational{Int16}}([-1, 1]),
            UInt8(1), UInt8(1), _stepsize_
        )
    end
    
    function FFDStamp( ::Val{1}, ::Val{2}, _stepsize_ :: AbstractFloat )
        FFDStamp( 
            SVector{3,Int8}([0, 1, 2]), 
            SVector{3,Rational{Int16}}([-3//2, 2, -1//2]),
            UInt8(1), UInt8(2), _stepsize_
        )
    end
    
    function FFDStamp( ::Val{1}, ::Val{3}, _stepsize_ :: AbstractFloat )
        FFDStamp( 
            SVector{4,Int8}([0, 1, 2, 3]), 
            SVector{4,Rational{Int16}}([-11/6, 3, -3//2, 1//3]),
            UInt8(1), UInt8(3), _stepsize_
        )
    end
    
end</code></pre>
<pre id='var-_zi' class='code-output documenter-example-output'>FFDStamp</pre>

<pre class='language-julia'><code class='language-julia'>begin 
    struct BFDStamp{N, F &lt;: AbstractFloat} &lt;: FiniteDiffStamp
        grid :: SVector{N,Int8}
        coeff :: SVector{N,Rational{Int16}}
        order :: UInt8 	
        accuracy :: UInt8
        
        stepsize :: F
    end
    
    _grid( stamp :: BFDStamp ) = stamp.grid
    _coeff( stamp :: BFDStamp ) = stamp.coeff 
    _order( stamp :: BFDStamp ) = stamp.order
    _accuracy( stamp :: BFDStamp ) = stamp.accuracy
    _stepsize( stamp :: BFDStamp ) = stamp.stepsize
    _zi( stamp :: BFDStamp ) = 1
    
    function BFDStamp( order :: Int, accuracy :: Int, _stepsize_ = stepsize(Float64) ) 
        ffd_stamp = FFDStamp( Val(order), Val( ceil(Int, accuracy / 2) ), _stepsize_ )
        if isodd(order)
            return BFDStamp(
                - _grid( ffd_stamp ),
                - _coeff( ffd_stamp ),
                _order(ffd_stamp), _accuracy(ffd_stamp), _stepsize(ffd_stamp)
            )
        else
            return  BFDStamp(
                _grid( ffd_stamp ), _coeff( ffd_stamp ),
                _order(ffd_stamp), _accuaracy(ffd_stamp), _stepsize(ffd_stamp)
            )
        end
    end

end</code></pre>
<pre id='var-_zi' class='code-output documenter-example-output'>BFDStamp</pre>


<div class="markdown"><p>Does it work?</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    cfd = CFDStamp(1,2)
    Func = x -&gt; x[1]*x[2]^2
    
    # ∂₁Func = x[2]^2
    # ∂₂Func = x[1]
    dx( cfd, Func, [1,2], 1)
end</code></pre>
<pre id='var-anon17656881308478907599' class='code-output documenter-example-output'>4.0</pre>


<div class="markdown"><h2>Tree structure</h2>
</div>


<div class="markdown"><p>Our goal is to recursively approximate high order derivatives of a function <span class="tex">$f\colon ℝ^n \to ℝ^k$</span> using a first order finite difference formula <span class="tex">$G_f$</span>. We assume, that <span class="tex">$G_f$</span> requires <span class="tex">$m$</span> values of derivative order <span class="tex">$i-1$</span> to approximate the partial derivative of order <span class="tex">$i$</span>. </p>
<p>For simplicity, first consider <span class="tex">$k&#61;1$</span> &#40;the single output case&#41;.</p>
<ul>
<li><p>For forming <span class="tex">$∂_1 f&#40;x_0&#41;$</span> we would need values <span class="tex">$f&#40;ξ_1^1&#41;, …, f&#40;ξ_m^1&#41;$</span>, where <span class="tex">$ξ^1$</span> varies along the first coordinate and results from applying <span class="tex">$G_f$</span> to <span class="tex">$x_0$</span>.</p>
</li>
<li><p>For forming <span class="tex">$∂_1 ∂_1 f&#40;x_0&#41;$</span> we would instead need values approximating <span class="tex">$∂_1 f&#40;ξ_1^1&#41;,…, ∂_1 f&#40;ξ_m^1&#41;$</span>.  Using the same rule to approximate the first order partial derivatives requires function evaluations at sites <span class="tex">$ξ_&#123;1,1&#125;^1, …, ξ_&#123;1,m&#125;^1, ξ_&#123;2,1&#125;^1, …, ξ_&#123;m,m&#125;^1$</span> resulting from applying <span class="tex">$G_f$</span> to each of <span class="tex">$ξ_i^1$</span>.</p>
</li>
</ul>
<p>This process can be recursed infinitely.  To actually compute the desired derivative approximation we would first have to construct the evaluation sites <span class="tex">$ξ_I^J$</span>.  We can think of a tree-like structure, with a root node containing <span class="tex">$x_0$</span> where we add – for each derivative order – child nodes by applying <span class="tex">$G_f$</span> to the previous leaves. <br />The tree is hence build from top to bottom but for evaluation we start at the leaves and resolve the lowest order derivative approximations first.  In a sense, this is a dynamic programming approach.</p>
</div>


<div class="markdown"><p>Our tree is made out of <code>FDiffNode</code>s.   An <code>FDiffNode</code> has a field <code>x_sym</code> storing the symbolic representation of some point relative to <span class="tex">$x_0$</span> that we need a value for. <br />If <code>N :: FDiffNode&#123;T,X,C&#125;</code> is a leave node, then <code>T</code> should be a mutable vector type and <code>vals :: T</code> is meant to store the result of <span class="tex">$f$</span> evaluated at <span class="tex">$x$</span>, where <span class="tex">$x$</span> results from substituting <span class="tex">$x_0$</span> and a stepsize <span class="tex">$h$</span> into <code>x_sym</code>. <br />If <code>N</code> is <em>not</em> a leave node, then <code>T</code> should be a vector type to store <span class="tex">$n$</span> vectors of <code>FDiffNode</code>s and each of those vectors should have <span class="tex">$m$</span> elements, so that it can be used to recursively approximate a derivative.</p>
<p>Note, that the variables <code>x_vars::Vector&#123;Symbolics.Num&#125;</code> and the finite difference rule are stored outside of the tree to keep it as simple as possible.</p>
<p>We also allow to use a cache of type <span class="tex">$C$</span> which should be Dict-Like &#40;if not <code>nothing</code>&#41;.  It can be used in the <code>val</code> method to retrieve values that have already been calculated before.</p>
</div>

<pre class='language-julia'><code class='language-julia'>@with_kw struct FDiffNode{T,X,C} &lt;: Trees.Node
    x_sym :: Vector{Symbolics.Num} = []
    x :: X = Float64[]
    vals :: T = nothing
    cache :: C = nothing
end</code></pre>
<pre id='var-@pack_FDiffNode' class='code-output documenter-example-output'>FDiffNode</pre>


<div class="markdown"><p>We inherit from <code>Trees.Node</code> and define a <code>children</code> method&#96; so that some nice iterators are available.</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    Trees.children( n :: FDiffNode{&lt;:RVec} ) = nothing
    Trees.children( n :: FDiffNode ) = Iterators.flatten( n.vals )
end</code></pre>



<div class="markdown"><p>For leave nodes &#40;indicated by <code>T</code> being a real vector type&#41;, we simply retrieve the function values if they are present. Else, <code>NaN</code> is returned.  We have a helper function <code>missing</code> to get the right type of <code>NaN</code> and the important <code>val</code> retrieval function.</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    missing(T::Type{&lt;:AbstractFloat}) = T(NaN)
    missing(T) = Float16(NaN)
end</code></pre>
<pre id='var-missing' class='code-output documenter-example-output'>missing (generic function with 2 methods)</pre>

<pre class='language-julia'><code class='language-julia'>function val( n :: FDiffNode{&lt;:AbstractVector{Y},X,C}, args...; output_index = 1) where {Y&lt;:Real,X,C}
    if isempty(n.vals) 
        return missing(Y)
    else 
        n.vals[output_index]
    end
end</code></pre>
<pre id='var-val' class='code-output documenter-example-output'>val (generic function with 2 methods)</pre>


<div class="markdown"><p>Otherwise, we <em>recurse</em> and apply the finite difference rule <code>stamp</code> to compute the return values.</p>
</div>

<pre class='language-julia'><code class='language-julia'>function val( subnode_list, indices, stamp, output_index )
    coeff = _coeff( stamp )
    h = _stepsize( stamp )
    return stamp( val(sub_node, indices, stamp; output_index) for sub_node in subnode_list ) 
end</code></pre>
<pre id='var-val' class='code-output documenter-example-output'>val (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function val( n :: FDiffNode{T,X,C}, indices, stamp; output_index = 1 ) where{T,X,C}
    copied_indices = copy(indices)
    i = popfirst!( copied_indices )
    
    if C &lt;: Nothing
        return val( n.vals[i], copied_indices, stamp, output_index )
    end
    
    cache_key = [ i; copied_indices; output_index ]
    if haskey( n.cache, cache_key ) 
        ret_val = n.cache[cache_key]
        !isnan(ret_val) && return ret_val
    end
    
    ret_val = val( n.vals[i], copied_indices, stamp, output_index )
    n.cache[cache_key] = ret_val
    return ret_val
end</code></pre>
<pre id='var-val' class='code-output documenter-example-output'>val (generic function with 3 methods)</pre>


<div class="markdown"><p>The tree is build recursively from the top down by calling the <code>build_tree</code> function with a decreasing <code>order</code> value.</p>
<ul>
<li><p>An <code>order</code> value of 0 indicates that we need need a leave node for <code>x_sym</code> and the correct value container is initialized &#40;and the cache deactivated&#41;.</p>
</li>
<li><p>For a higher <code>order</code> value, we collect the <span class="tex">$n$</span> vectors of <span class="tex">$m$</span> sub nodes by calling <code>buid_tree</code> again.</p>
</li>
</ul>
</div>

<pre class='language-julia'><code class='language-julia'>function build_tree( x_sym, stamp, vars, order = 1; x_type = Vector{Float64}, val_type = Vector{Float64}, cache_type = Nothing )
    if order &lt;= 0
        # return a leave node
        return FDiffNode(; x_sym, x = x_type(), vals = val_type(), cache = nothing )
    else
        # collect ``n`` subnode vectors of length ``m``
        grid = _grid( stamp )	# ``m`` grid points to define the points for sub_nodes
        m = length( grid )
        n = length( x_sym )
        h = vars[2]
        #sub_nodes = SizedVector{n, SizedVector{m, FDiffNode}}(undef)
        sub_nodes = SizedVector{n, SizedVector{m, FDiffNode}}(undef)
        for i = 1 : n
            sub_nodes[i] = [ 
                build_tree( 
                    # vary `x_sym` along variable `i`
                    [x_sym[1:i-1]; x_sym[i] + h * g; x_sym[i+1:n]], 
                    stamp, vars, order - 1; x_type, val_type, cache_type ) 
                for g in grid 
            ]
        end
        return FDiffNode(; x_sym, x = x_type(), vals = sub_nodes, cache = cache_type() )
    end
end</code></pre>
<pre id='var-build_tree' class='code-output documenter-example-output'>build_tree (generic function with 2 methods)</pre>


<div class="markdown"><p><code>build_tree</code> is called in default the <code>DiffWrapper</code> constructor to store the <code>tree</code> &#40;out of <code>FDiffNode</code>s&#41; for the derivative <code>order</code> stored within the container.  A <code>DiffWrapper</code> also stores the base point <code>x0</code> and the finite difference rule <code>stamp</code>.</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    "Helper function to get Symbolic variables for `x_1,…,x_n` and the stepsize `h`." 
    function _get_vars(n :: Int)
        return Tuple(Num.(Variable.(:x, 1:n))), Num(Variable(:h))
    end
    
    _get_vars( x :: AbstractVector ) = _get_vars(length(x))
end</code></pre>
<pre id='var-_get_vars' class='code-output documenter-example-output'>_get_vars (generic function with 2 methods)</pre>

<pre class='language-julia'><code class='language-julia'>begin
    vec_typex( x :: AbstractVector{Y}, stamp ) where Y = Vector{ Base.promote_op( +, Y, typeof(_stepsize(stamp))) }
    vec_typef( fx :: AbstractVector{Y}, stamp ) where Y = Vector{ Base.promote_op( *, Y, typeof(_coeff(stamp)[1])) }
end</code></pre>
<pre id='var-vec_typef' class='code-output documenter-example-output'>vec_typef (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>@with_kw struct DiffWrapper{X &lt;: RVec,Y &lt;: RVec,N,S,T,C}
    x0 :: X
    fx0 :: Y
    #vars :: Tuple{Tuple{N,Symbolics.Num}, Symbolics.Num} = _get_vars( x0 )
    vars :: N = _get_vars(x0)
    x0_sym :: Vector{Symbolics.Num} = [vars[1]...]
    stamp :: S = CFDStamp(1,2, stepsize(x0))
    order :: Int = 1
        
    cache_type :: C = Dict{Vector{UInt8}, eltype(vec_typef(fx0, stamp))} 
    tree :: T = build_tree( x0_sym, stamp, vars, order; x_type = vec_typex(x0,stamp), val_type = vec_typef(fx0, stamp), cache_type = cache_type )
end</code></pre>
<pre id='var-@pack_DiffWrapper' class='code-output documenter-example-output'>DiffWrapper</pre>


<div class="markdown"><div class="admonition is-note">
  <header class="admonition-header">
    Note
  </header>
  <div class="admonition-body">
    <p>A <code>DiffWrapper</code> should be initialized with the right &#40;floating point&#41; vector types for <code>x0</code> and <code>fx0</code>. If <code>fx0</code> is not known, but the precision, use something like <code>fx0 &#61; Float32&#91;&#93;</code>.</p>
  </div>
</div>
</div>


<div class="markdown"><p>We exploit the meta data stored to forward the <code>val</code> method. <code>indices</code> should be an array of <code>Int</code>s of length <code>dw.order</code>, indicating the partial derivatives we want. Suppose, <code>dw.order &#61;&#61; 2</code>, then <code>val&#40; dw, &#91;1,2&#93; &#41;</code> will give <span class="tex">$∂_1∂_2f_1&#40;x_0&#41;$</span>.</p>
</div>

<pre class='language-julia'><code class='language-julia'>val( dw :: DiffWrapper, indices; output_index = 1 ) = val( dw.tree, indices, dw.stamp ; output_index)</code></pre>
<pre id='var-val' class='code-output documenter-example-output'>val (generic function with 4 methods)</pre>


<div class="markdown"><div class="admonition is-note">
  <header class="admonition-header">
    Note
  </header>
  <div class="admonition-body">
    <p>The <code>tree</code> of a <code>DiffWrapper</code> needs to be initialized befor calling any derivative method.   That means, the <code>x</code> and <code>vals</code> fields have to be set for each leave node. This can be done in one go, by calling <code>prepare_tree&#33;&#40; dw, f &#41;</code>, or sequentially by calling first <code>substitute_leaves&#33;&#40;dw&#41;</code> and and <code>set_leave_values&#40;dw,f&#41;</code>.</p>
  </div>
</div>
</div>


<div class="markdown"><p>From this, it is easy to define convenience functions for hessians and gradients. For second order <code>DiffWrapper</code>s we can also make use of the fact, that each derivative rule has a grid point with index <code>zi</code> where <span class="tex">$x_0$</span> is not varied. That means, we can collect the gradient from the leaves with the parent node of index <code>zi</code> &#40;relative to the root&#41;.</p>
</div>

<pre class='language-julia'><code class='language-julia'>function gradient( dw :: DiffWrapper; output_index = 1 )
    @assert 1 &lt;= dw.order &lt;= 2 "Gradient retrieval only implemented for DiffWrapper of order 1 and 2."
    return gradient( dw :: DiffWrapper, Val(dw.order); output_index )
end</code></pre>
<pre id='var-gradient' class='code-output documenter-example-output'>gradient (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function gradient( dw :: DiffWrapper, ::Val{1}; output_index = 1 )
    n = length( dw.x0 )
    g = Vector{eltype(dw.fx0)}(undef, n)
    for i = 1 : n
        g[i] = val( dw, [i,]; output_index )
    end
    return g
end</code></pre>
<pre id='var-gradient' class='code-output documenter-example-output'>gradient (generic function with 2 methods)</pre>

<pre class='language-julia'><code class='language-julia'>function gradient( dw :: DiffWrapper, ::Val{2}; output_index = 2 )
    n = length( dw.x0 )
    g = Vector{eltype(dw.fx0)}(undef, n)
    zi = _zi( dw.stamp )	# index of stamp grid point is zero
    node = dw.tree.vals[1][zi]
    for i = 1 : n
        g[i] = val( node, [i,], dw.stamp; output_index )		
    end
    return g
end</code></pre>
<pre id='var-gradient' class='code-output documenter-example-output'>gradient (generic function with 3 methods)</pre>

<pre class='language-julia'><code class='language-julia'>function jacobian( dw :: DiffWrapper )
    @assert 1 &lt;= dw.order &lt;= 2 "Gradient retrieval only implemented for DiffWrapper of order 1 and 2."
    k = isempty( dw.fx0 ) ? length(first(Trees.Leaves(dw.tree)).vals) : length(dw.fx0)
    return transpose( hcat( collect(gradient(dw; output_index = m) for m = 1 : k )...) )
end</code></pre>
<pre id='var-jacobian' class='code-output documenter-example-output'>jacobian (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function hessian( dw :: DiffWrapper; output_index = 1)
    @assert dw.order == 2 "Hessian only implemented for DiffWrapper of 2."
    n = length(dw.x0)
    H = Matrix{eltype(dw.fx0)}(undef, n, n)	
    # TODO make sure that dw.fx0 == vec_typef(fx0, stamp)
    for i = 1 : n
        for j = 1 : n
            H[i,j] = val( dw, [i, j]; output_index ) 
        end
    end
    return H
end</code></pre>
<pre id='var-hessian' class='code-output documenter-example-output'>hessian (generic function with 1 method)</pre>


<div class="markdown"><p>Does it work?</p>
</div>

<pre class='language-julia'><code class='language-julia'>begin
    # I used this cell for fiddling around.
    
    f = x -&gt; [ 2 * x[1]^2 + x[2]^3; sum( x ) ]
    # ∂1 f1 = 4 * x[1] 
    # ∂2 f1 = 3 * x[2]^2
    
    #H1 = [
    #	4.0 0.0
    #	0.0 6x[2]
    # ]
    
    dw = DiffWrapper(; x0 = [1.0; 1.0], fx0 = rand(2) , stamp = BFDStamp(1,3,1e-3), order = 2)
    # to deactivate caching: `cache_type = Nothing`
        
    prepare_tree!(dw,f)
    # this would require 2 tree traversals:
    # substitute_leaves!(dw)	
    # set_leave_values!(dw,f)
    
    H = hessian(dw)
    H, sum( ([ 4 0; 0 6 ] .- H).^2 )
end</code></pre>
<pre id='var-f' class='code-output documenter-example-output'>([4.000000002557513 8.881784197001252e-10; -1.1102230246251565e-9 5.999999997285954], 1.592837543398526e-17)</pre>

<pre class='language-julia'><code class='language-julia'>jacobian(dw)</code></pre>
<pre id='var-hash106719' class='code-output documenter-example-output'>2×2 transpose(::Matrix{Float64}) with eltype Float64:
 4.0  3.0
 1.0  1.0</pre>

<pre class='language-julia'><code class='language-julia'>first(Trees.Leaves(dw.tree))</code></pre>
<pre id='var-hash165135' class='code-output documenter-example-output'>FDiffNode{Vector{Float64}, Vector{Float64}, Nothing}
  x_sym: Array{Num}((2,)) Symbolics.Num[x₁, x₂]
  x: Array{Float64}((2,)) [1.0, 1.0]
  vals: Array{Float64}((2,)) [3.0, 2.0]
  cache: Nothing nothing
</pre>


<div class="markdown"><h4>Helpers for filling the Tree</h4>
<p>I won&#39;t go into detail concering these helper functions. They mostly leverage the <code>Trees.Leaves</code> iterator and should be comprehensible by themselves.</p>
</div>

<pre class='language-julia'><code class='language-julia'>function _substitute_symbols!( node :: FDiffNode, x0, h, vars )
    x_vars, h_var = vars
    empty!(node.x)
    append!(node.x, Symbolics.value.(substitute.(node.x_sym, (
        Dict((x_vars[i] =&gt; x0[i] for i = eachindex(x0))..., 
                    h_var=&gt;h),
    ))))
end</code></pre>
<pre id='var-_substitute_symbols!' class='code-output documenter-example-output'>_substitute_symbols! (generic function with 2 methods)</pre>

<pre class='language-julia'><code class='language-julia'>function _substitute_symbols!(root_iterator, x0, h, vars)
    for node in root_iterator
        _substitute_symbols!(node, x0, h, vars)
    end
end</code></pre>
<pre id='var-_substitute_symbols!' class='code-output documenter-example-output'>_substitute_symbols! (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function substitute_symbols!(root :: FDiffNode, x0, h, vars )
    _substitute_symbols!( Trees.PreOrderDFS( root ), x0, h, vars )
end</code></pre>
<pre id='var-substitute_symbols!' class='code-output documenter-example-output'>substitute_symbols! (generic function with 2 methods)</pre>

<pre class='language-julia'><code class='language-julia'>function substitute_leaves!(root :: FDiffNode, x0, h, vars )
    _substitute_symbols!( Trees.Leaves(root), x0, h, vars )
end</code></pre>
<pre id='var-substitute_leaves!' class='code-output documenter-example-output'>substitute_leaves! (generic function with 2 methods)</pre>

<pre class='language-julia'><code class='language-julia'>function substitute_symbols!(dw :: DiffWrapper)
    substitute_symbols!( dw.tree, dw.x0, _stepsize( dw.stamp ), dw.vars )
end</code></pre>
<pre id='var-substitute_symbols!' class='code-output documenter-example-output'>substitute_symbols! (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function substitute_leaves!(dw :: DiffWrapper)
    substitute_leaves!( dw.tree, dw.x0, _stepsize( dw.stamp ), dw.vars )
end</code></pre>
<pre id='var-substitute_leaves!' class='code-output documenter-example-output'>substitute_leaves! (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function collect_leave_sites( dw :: DiffWrapper )
    return [ node.x for node in Trees.Leaves( dw.tree ) ]
end</code></pre>
<pre id='var-collect_leave_sites' class='code-output documenter-example-output'>collect_leave_sites (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function _set_node_values!( node, f :: Function )
    empty!(node.vals)
    append!(node.vals, f( node.x ))
end</code></pre>
<pre id='var-_set_node_values!' class='code-output documenter-example-output'>_set_node_values! (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function set_leave_values!(dw :: DiffWrapper, f :: Function )
    for node in Trees.Leaves(dw.tree)
        _set_node_values!(node, f)
    end
end</code></pre>
<pre id='var-set_leave_values!' class='code-output documenter-example-output'>set_leave_values! (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>function set_leave_values!(dw :: DiffWrapper, leave_vals :: AbstractVector )
    for (i,node) in enumerate(Trees.Leaves(dw.tree))
        empty!(node.vals)
        append!(node.vals, leave_vals[i] )
    end
end</code></pre>
<pre id='var-set_leave_values!' class='code-output documenter-example-output'>set_leave_values! (generic function with 2 methods)</pre>


<div class="markdown"><div class="admonition is-note">
  <header class="admonition-header">
    Note
  </header>
  <div class="admonition-body">
    <p>The <code>set_leave_values&#33;</code> methods should also reset the cache. Not implemented in this notebook.</p>
  </div>
</div>
</div>

<pre class='language-julia'><code class='language-julia'>function prepare_tree!( dw :: DiffWrapper, f :: Function )
    x0 = dw.x0
    vars = dw.vars
    h = _stepsize( dw.stamp )
    
    for node in Trees.Leaves( dw.tree )
        _substitute_symbols!(node, x0, h, vars)
        _set_node_values!( node, f )
    end
end</code></pre>
<pre id='var-prepare_tree!' class='code-output documenter-example-output'>prepare_tree! (generic function with 1 method)</pre>


<div class="markdown"><h3>Morbit Example</h3>
</div>

<pre class='language-julia'><code class='language-julia'>function unique_with_indices( x :: AbstractVector{T} ) where T
    unique_elems = T[]
    indices = Int[]
    for elem in x
        i = findfirst( e -&gt; all( isequal.(e,elem) ), unique_elems )
        if isnothing(i)
            push!(unique_elems, elem)
            push!(indices, length(unique_elems) )
        else
            push!(indices, i)
        end
    end
    return unique_elems, indices
end</code></pre>
<pre id='var-unique_with_indices' class='code-output documenter-example-output'>unique_with_indices (generic function with 1 method)</pre>

<pre class='language-julia'><code class='language-julia'>begin
    func = x -&gt; [ sum(x.^2); exp( sum( x ) ) ]
    x_t = ones(Float32, 3)
    fx_t = func( x_t )
    diff_wrapper = DiffWrapper(; x0 = x_t, fx0 = fx_t, stamp = CFDStamp(1,2,stepsize(fx_t)), order = 2 )
    
    #prepare_tree!(diff_wrapper, func)
    
    # phase I: get sites
    substitute_leaves!( diff_wrapper )
    all_sites = collect_leave_sites( diff_wrapper )
    
    # phase II: evaluation

    ## find unique sites to avoid costly objective evaluations
    unique_sites, set_indices = unique_with_indices( [[x_t,]; all_sites] )
    ## now unique_sites[set_indices] == [[x_t,]; all_sites]
    
    @assert unique_sites[set_indices][2:end] == all_sites
    
    ## and x_t will be the first site in unique sites
    
    ## call `func` on new sites
    unique_evals = [ [fx_t,]; func.( unique_sites[ 2 : end ] )]
    all_evals = unique_evals[set_indices[2:end]]
    
    # phase III: set the leave values and get jacobian
    set_leave_values!(diff_wrapper, all_evals)
    
    hessian(diff_wrapper; output_index = 1), jacobian(diff_wrapper)
end</code></pre>
<pre id='var-anon1026769008749790319' class='code-output documenter-example-output'>(Float32[1.9999936 0.0 0.0; 0.0 1.9999936 0.0; 0.0 0.0 1.9999936], Float32[1.9999936 1.9999936 1.9999936; 20.086252 20.086252 20.086252])</pre>


<div class="markdown"><hr />
</div>

<pre class='language-julia'><code class='language-julia'>Trees = ingredients( joinpath(@__DIR__, "Trees.jl") ).Trees</code></pre>
<pre id='var-Trees' class='code-output documenter-example-output'>Main.Trees.jl.Trees</pre>


<div class="markdown"><p><code>ingredients</code> thanks to <a href="https://github.com/fonsp/Pluto.jl/issues/115">fonsp</a></p>
</div>

<pre class='language-julia'><code class='language-julia'>function ingredients(path::String)
    # this is from the Julia source code (evalfile in base/loading.jl)
    # but with the modification that it returns the module instead of the last object
    name = Symbol(basename(path))
    m = Module(name)
    Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
    m
end</code></pre>
<pre id='var-ingredients' class='code-output documenter-example-output'>ingredients (generic function with 1 method)</pre>

<!-- PlutoStaticHTML.End -->
```

