module RecursiveFiniteDifferences

# TODO: clean this up :)
# This is simply the notebook also shown in the docs.
# I just wrapped it in a module, fixed the import of Trees and commented out some examples

include(joinpath(@__DIR__, "Trees.jl"))
using .Trees

### A Pluto.jl notebook ###
# v0.15.1

using Markdown
using InteractiveUtils

# ╔═╡ e7d76f0e-eea5-11eb-1bdf-9196c513ff0a
begin
	using StaticArrays
	using Symbolics
	using Symbolics: Sym
	using Parameters
end

# ╔═╡ 7b914a4f-89b4-4284-b8c6-d629157b9a7c
md"# Finite Differences for Taylor Models."

# ╔═╡ 44f2f27b-2ac6-4df1-b222-4cfa37f41985
begin
	const RVec = AbstractVector{<:Real}
	const RVecOrR = Union{Real, RVec}
end

# ╔═╡ 10b9b062-450e-43a2-9c7a-2fe6c7121ab7
md"""
### Abstract

For calculating finite difference gradients and hessians in Morbit we want to separate the grid construction from the function evaluation and derivative calculations.
To my knowledge, this is not easily possible using `FiniteDiff` or `FiniteDifferences.jl`.

This notebook explores a recursive calculation procedure starting from first order finite difference rules (interfaced by `FiniteDiffStamp`).
To calculate partial derivatives of ``f:ℝ^n \to ℝ^k`` at a fixed point ``x_0 ∈ ℝ^n`` a `DiffWrapper` can be used. 
It is initialized devoid of any function evaluation information but only has symbolic representations of required evaluation sites relative to ``x_0``.
A `DiffWrapper` can hence be used to first construct a grid for ``x_0`` and at a later stage to approximate gradients or Hessians.
"""

# ╔═╡ bd83fc2a-d7bd-489e-83f3-ef5b327d5cca
md"""
## Finite Difference Rules

A finite difference rule is defined by a grid of integer offsets, a decision space stepsize ``h``, and coefficients to combine evaluation results at the grid points.

For details, please see [this Wikipedia page](https://en.wikipedia.org/wiki/Finite_difference_coefficient).

We abstract the basic information with the `FiniteDiffStamp` interface:
"""

# ╔═╡ a7d81ac5-0124-441a-8eae-f83cee2b81c3
begin
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
end

# ╔═╡ fdcf00a4-836e-4404-ba1c-6058ae85d9ce
md"We could already define a finite difference function using a `FiniteDiffStamp`:"

# ╔═╡ 07e698bb-fac0-4951-85dc-5adb208cf5dd
md"Of course, some concrete implementations are still missing. 
We provide some entries from the table in [Wikipedia](https://en.wikipedia.org/wiki/Finite_difference_coefficient)."

# ╔═╡ 9462b310-0842-4373-ac2c-f2eaef5e6e7f
begin
	stepsize( x :: F ) where F <: AbstractFloat = 10 * sqrt(eps(F))
	stepsize( F :: Type{<:AbstractFloat} ) = 10 * sqrt(eps(F))
	stepsize( x :: AbstractVector{F} ) where F<:AbstractFloat = stepsize(F)
	stepsize( x ) = stepsize(Float64)
end

# ╔═╡ c19325b6-7516-4acf-a5f1-fe7084bc113d
begin
	struct CFDStamp{N, F <: AbstractFloat} <: FiniteDiffStamp
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
	
	
end

# ╔═╡ 2426f92a-21f3-4455-aeb7-fe6674aafe02
begin
	struct FFDStamp{N, F <: AbstractFloat} <: FiniteDiffStamp
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
	
end

# ╔═╡ 4ebb80a7-539d-49a0-981f-166f0bbf3964
begin 
	struct BFDStamp{N, F <: AbstractFloat} <: FiniteDiffStamp
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

end

# ╔═╡ aeac724f-b87a-4f37-aa57-893d1d131f6d
"Evaluate all outputs of `f` with respect to ``x_i``." 
function dx(stamp :: FiniteDiffStamp, f :: Function, x0 :: RVec, i = 1)
	h = _stepsize(stamp)
	grid = [ [x0[1:i-1];x0[i] + h*g; x0[i+1:end]] for g in _grid(stamp) ]
	evals = f.(grid)
	return stamp( evals )
end

# ╔═╡ a5e02956-c0f9-44a6-80da-abfb497bcfb9
md"Does it work?"

# ╔═╡ 09d78bbc-f568-459a-8bd0-05de84b04af2
begin
	cfd = CFDStamp(1,2)
	Func = x -> x[1]*x[2]^2
	
	# ∂₁Func = x[2]^2
	# ∂₂Func = x[1]
	dx( cfd, Func, [1,2], 1)
end

# ╔═╡ 04861d1b-38d9-4ba7-9fde-b10150f39280
md"## Tree structure"

# ╔═╡ 20907685-2bb6-49d5-9ff0-a175435fa591
md"""
Our goal is to recursively approximate high order derivatives of a function ``f\colon ℝ^n \to ℝ^k`` using a first order finite difference formula ``G_f``.
We assume, that ``G_f`` requires ``m`` values of derivative order ``i-1`` to approximate the partial derivative of order ``i``. 

For simplicity, first consider ``k=1`` (the single output case).
* For forming ``∂_1 f(x_0)`` we would need values ``f(ξ_1^1), …, f(ξ_m^1)``, where ``ξ^1`` varies along the first coordinate and results from applying ``G_f`` to ``x_0``.
* For forming ``∂_1 ∂_1 f(x_0)`` we would instead need values approximating ``∂_1 f(ξ_1^1),…, ∂_1 f(ξ_m^1)``. 
  Using the same rule to approximate the first order partial derivatives requires function evaluations at sites ``ξ_{1,1}^1, …, ξ_{1,m}^1, ξ_{2,1}^1, …, ξ_{m,m}^1`` resulting from applying ``G_f`` to each of ``ξ_i^1``.

This process can be recursed infinitely. 
To actually compute the desired derivative approximation we would first have to construct the evaluation sites ``ξ_I^J``. 
We can think of a tree-like structure, with a root node containing ``x_0`` where we add -- for each derivative order -- child nodes by applying ``G_f`` to the previous leaves. \
The tree is hence build from top to bottom but for evaluation we start at the leaves and resolve the lowest order derivative approximations first. 
In a sense, this is a dynamic programming approach.
"""

# ╔═╡ 65d0f18a-6da9-4836-bf4c-8ff10e6a265f
md"""
Our tree is made out of `FDiffNode`s.  
An `FDiffNode` has a field `x_sym` storing the symbolic representation of some point relative to ``x_0`` that we need a value for. \
If `N :: FDiffNode{T,X,C}` is a leave node, then `T` should be a mutable vector type and `vals :: T` is meant to store the result of ``f`` evaluated at ``x``, where ``x`` results from substituting ``x_0`` and a stepsize ``h`` into `x_sym`. \
If `N` is *not* a leave node, then `T` should be a vector type to store ``n`` vectors of `FDiffNode`s and each of those vectors should have ``m`` elements, so that it can be used to recursively approximate a derivative.

Note, that the variables `x_vars::Vector{Symbolics.Num}` and the finite difference rule are stored outside of the tree to keep it as simple as possible.

We also allow to use a cache of type ``C`` which should be Dict-Like (if not `nothing`). 
It can be used in the `val` method to retrieve values that have already been calculated before.
"""

# ╔═╡ 6cdc6c1e-790e-4d2a-984c-36f5f0c98f63
md"We inherit from `Trees.Node` and define a `children` method` so that some nice iterators are available."

# ╔═╡ dc4d3c61-f2ec-4fb1-b84b-2c1957da3a7e
md"For leave nodes (indicated by `T` being a real vector type), we simply retrieve the function values if they are present. Else, `NaN` is returned. 
We have a helper function `missing` to get the right type of `NaN` and the important `val` retrieval function."

# ╔═╡ c2cb0dd1-e36d-44ff-aa59-6365a15e7ab6
begin
	missing(T::Type{<:AbstractFloat}) = T(NaN)
	missing(T) = Float16(NaN)
end

# ╔═╡ 3f67de97-2a12-4ca7-82bd-16851f958989
md"Otherwise, we *recurse* and apply the finite difference rule `stamp` to compute the return values."

# ╔═╡ 18e51761-98ae-4814-b575-3bb4a40a06cb
function val( subnode_list, indices, stamp, output_index )
	coeff = _coeff( stamp )
	h = _stepsize( stamp )
	return stamp( val(sub_node, indices, stamp; output_index) for sub_node in subnode_list ) 
end

# ╔═╡ a5565a82-2b6d-4c43-909c-9bfeca9890cb
md"""The tree is build recursively from the top down by calling the `build_tree` function with a decreasing `order` value.
* An `order` value of 0 indicates that we need need a leave node for `x_sym` and the correct value container is initialized (and the cache deactivated).
* For a higher `order` value, we collect the ``n`` vectors of ``m`` sub nodes by calling `buid_tree` again.
"""

# ╔═╡ 1db29942-3ff0-49e8-b5c6-2480eadd608f
md"`build_tree` is called in default the `DiffWrapper` constructor to store the `tree` (out of `FDiffNode`s) for the derivative `order` stored within the container. 
A `DiffWrapper` also stores the base point `x0` and the finite difference rule `stamp`.
"

# ╔═╡ af650705-0636-4728-a643-ab658d362290
begin
	"Helper function to get Symbolic variables for `x_1,…,x_n` and the stepsize `h`." 
	function _get_vars(n :: Int)
		return Tuple(Num.(Variable.(:x, 1:n))), Num(Variable(:h))
	end
	
	_get_vars( x :: AbstractVector ) = _get_vars(length(x))
end

# ╔═╡ 7c091ff3-7d34-4ad8-b959-fe601397f091
begin
	vec_typex( x :: AbstractVector{Y}, stamp ) where Y = Vector{ Base.promote_op( +, Y, typeof(_stepsize(stamp))) }
	vec_typef( fx :: AbstractVector{Y}, stamp ) where Y = Vector{ Base.promote_op( *, Y, typeof(_coeff(stamp)[1])) }
end

# ╔═╡ 59211507-8901-4f4c-b2ab-c5d4fb950bf1
md"""
!!! note
    A `DiffWrapper` should be initialized with the right (floating point) vector types for `x0` and `fx0`. If `fx0` is not known, but the precision, use something like `fx0 = Float32[]`.
"""

# ╔═╡ eb61f081-bd96-4a25-b2ba-9a3bc9e87934
md"We exploit the meta data stored to forward the `val` method.
`indices` should be an array of `Int`s of length `dw.order`, indicating the partial derivatives we want.
Suppose, `dw.order == 2`, then `val( dw, [1,2] )` will give ``∂_1∂_2f_1(x_0)``.
"

# ╔═╡ f77f8375-5b8f-4a3e-b571-28ebccb204c4
md"""
!!! note
    The `tree` of a `DiffWrapper` needs to be initialized befor calling any derivative method.  
    That means, the `x` and `vals` fields have to be set for each leave node.
	This can be done in one go, by calling `prepare_tree!( dw, f )`, or sequentially by calling first `substitute_leaves!(dw)` and and `set_leave_values(dw,f)`.
"""

# ╔═╡ fe8171dc-7efd-4acb-b1f9-0da5f1c6dc9d
md"From this, it is easy to define convenience functions for hessians and gradients.
For second order `DiffWrapper`s we can also make use of the fact, that each derivative rule has a grid point with index `zi` where ``x_0`` is not varied.
That means, we can collect the gradient from the leaves with the parent node of index `zi` (relative to the root)."

# ╔═╡ 5dd33903-f4a4-455c-8348-c3c98e5f6404
md"Does it work?"

# ╔═╡ 8ad29cb3-9bb5-48d6-b0a6-ceb46e30293e
md"""
#### Helpers for filling the Tree

I won't go into detail concering these helper functions. They mostly leverage the `Trees.Leaves` iterator and should be comprehensible by themselves.
"""

# ╔═╡ fef9850c-3cbd-47f8-a786-e420f01d9141
function _substitute_symbols!(root_iterator, x0, h, vars)
	for node in root_iterator
		_substitute_symbols!(node, x0, h, vars)
	end
end

# ╔═╡ ac3d101b-c9d1-4fdb-80a4-645464298ba3
function _set_node_values!( node, f :: Function )
	empty!(node.vals)
	append!(node.vals, f( node.x ))
end

# ╔═╡ 5d586923-a32e-4d16-a0a6-56bbc73680f2
md"""
### Morbit Example
"""

# ╔═╡ 3feed8b1-1338-47c7-960f-892481f96a5c
function unique_with_indices( x :: AbstractVector{T} ) where T
	unique_elems = T[]
	indices = Int[]
	for elem in x
		i = findfirst( e -> all( isequal.(e,elem) ), unique_elems )
		if isnothing(i)
			push!(unique_elems, elem)
			push!(indices, length(unique_elems) )
		else
			push!(indices, i)
		end
	end
	return unique_elems, indices
end

# ╔═╡ d41308c0-b7f1-4ed9-b27e-7c6fe87080e6
md"---"

# ╔═╡ f294835d-ee9a-42b1-b2b7-5816d65e0f18
md"`ingredients` thanks to [fonsp](https://github.com/fonsp/Pluto.jl/issues/115)"

# ╔═╡ e9af48fd-ac05-4358-8a33-c14008ca1cb9
#=
function ingredients(path::String)
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
end
=#

# ╔═╡ 6c6e63ae-28ce-4c1c-bb60-7e7826ffefbb
# Trees = ingredients( joinpath(@__DIR__, ".." ,"src", "Trees.jl") )

# ╔═╡ c3fa417a-898f-4273-a75c-974a42e80a79
@with_kw struct FDiffNode{T,X,C} <: Trees.Node
	x_sym :: Vector{Symbolics.Num} = []
	x :: X = Float64[]
	vals :: T = nothing
	cache :: C = nothing
end

# ╔═╡ aabc4fed-e62f-4668-9984-c9b72cad15e9
function val( n :: FDiffNode{<:AbstractVector{Y},X,C}, args...; output_index = 1) where {Y<:Real,X,C}
	if isempty(n.vals) 
		return missing(Y)
	else 
		n.vals[output_index]
	end
end

# ╔═╡ d21fed2f-f61a-42d6-aed3-430bf73fd170
function val( n :: FDiffNode{T,X,C}, indices, stamp; output_index = 1 ) where{T,X,C}
	copied_indices = copy(indices)
	i = popfirst!( copied_indices )
	
	if C <: Nothing
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
end

# ╔═╡ 17347e0e-08ee-4a29-83b3-f337e397f49b
function build_tree( x_sym, stamp, vars, order = 1; x_type = Vector{Float64}, val_type = Vector{Float64}, cache_type = Nothing )
	
	if order <= 0
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
end

# ╔═╡ a09ab593-8efd-4799-8f0f-aa2ceb1e135d
@with_kw struct DiffWrapper{X <: RVec,Y <: RVec,N,S,T,C}
	x0 :: X
	fx0 :: Y
	#vars :: Tuple{Tuple{N,Symbolics.Num}, Symbolics.Num} = _get_vars( x0 )
	vars :: N = _get_vars(x0)
	x0_sym :: Vector{Symbolics.Num} = [vars[1]...]
	stamp :: S = CFDStamp(1,2, stepsize(x0))
	order :: Int = 1
		
	cache_type :: C = Dict{Vector{UInt8}, eltype(vec_typef(fx0, stamp))} 
	tree :: T = build_tree( x0_sym, stamp, vars, order; x_type = vec_typex(x0,stamp), val_type = vec_typef(fx0, stamp), cache_type = cache_type )
end

# ╔═╡ 5253756b-1a03-4d8f-854a-b03c5f396dc9
val( dw :: DiffWrapper, indices; output_index = 1 ) = val( dw.tree, indices, dw.stamp ; output_index)

# ╔═╡ dfc93b7e-980d-4f35-9d72-16cee2c747c2
function gradient( dw :: DiffWrapper; output_index = 1 )
	@assert 1 <= dw.order <= 2 "Gradient retrieval only implemented for DiffWrapper of order 1 and 2."
	return gradient( dw :: DiffWrapper, Val(dw.order); output_index )
end

# ╔═╡ a728763c-8995-4d09-9878-1db83afc9194
function gradient( dw :: DiffWrapper, ::Val{1}; output_index = 1 )
	n = length( dw.x0 )
	g = Vector{eltype(dw.fx0)}(undef, n)
	for i = 1 : n
		g[i] = val( dw, [i,]; output_index )
	end
	return g
end

# ╔═╡ 2f50ce22-7a20-4b24-ac57-5f3c3918079e
function gradient( dw :: DiffWrapper, ::Val{2}; output_index = 2 )
	n = length( dw.x0 )
	g = Vector{eltype(dw.fx0)}(undef, n)
	zi = _zi( dw.stamp )	# index of stamp grid point is zero
	node = dw.tree.vals[1][zi]
	for i = 1 : n
		g[i] = val( node, [i,], dw.stamp; output_index )		
	end
	return g
end

# ╔═╡ 634067d4-15db-4aa5-8b5a-b0794ee88a39
function hessian( dw :: DiffWrapper; output_index = 1)
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
end

# ╔═╡ dbdceb0d-e4bb-406c-b3cf-defffcc1ccf9
function substitute_symbols!(dw :: DiffWrapper)
	substitute_symbols!( dw.tree, dw.x0, _stepsize( dw.stamp ), dw.vars )
end

# ╔═╡ 8c11b8e1-c0fb-48b8-b9f1-8679e3d0e596
function substitute_leaves!(dw :: DiffWrapper)
	substitute_leaves!( dw.tree, dw.x0, _stepsize( dw.stamp ), dw.vars )
end

# ╔═╡ 351d4199-28fa-4733-89d1-266ee16137c4
function _substitute_symbols!( node :: FDiffNode, x0, h, vars )
	x_vars, h_var = vars
	empty!(node.x)
	append!(node.x, Symbolics.value.(substitute.(node.x_sym, (
		Dict((x_vars[i] => x0[i] for i = eachindex(x0))..., 
					h_var=>h),
	))))
end

# ╔═╡ 94d3947d-fb1c-4e8d-9b16-bdf14a411193
begin
	Trees.children( n :: FDiffNode{<:RVec} ) = nothing
	Trees.children( n :: FDiffNode ) = Iterators.flatten( n.vals )
end

# ╔═╡ 98051445-7ffd-4a92-af38-07507efab445
function jacobian( dw :: DiffWrapper )
	@assert 1 <= dw.order <= 2 "Gradient retrieval only implemented for DiffWrapper of order 1 and 2."
	k = isempty( dw.fx0 ) ? length(first(Trees.Leaves(dw.tree)).vals) : length(dw.fx0)
	return transpose( hcat( collect(gradient(dw; output_index = m) for m = 1 : k )...) )
end

# ╔═╡ 0be3107e-a3c9-46f5-837d-c7f6d2b2d595
function substitute_symbols!(root :: FDiffNode, x0, h, vars )
	_substitute_symbols!( Trees.PreOrderDFS( root ), x0, h, vars )
end

# ╔═╡ 42e6c63e-0525-4ec6-88b9-b077130603ff
function substitute_leaves!(root :: FDiffNode, x0, h, vars )
	_substitute_symbols!( Trees.Leaves(root), x0, h, vars )
end

# ╔═╡ df8bd829-5c28-46c1-b171-1b5dc0358c76
function collect_leave_sites( dw :: DiffWrapper )
	return [ node.x for node in Trees.Leaves( dw.tree ) ]
end

# ╔═╡ 7b725d45-f6fa-4bd8-99d7-fcad166320e5
function set_leave_values!(dw :: DiffWrapper, f :: Function )
	for node in Trees.Leaves(dw.tree)
		_set_node_values!(node, f)
	end
end

# ╔═╡ 9cb6592b-ba7d-4b46-9171-1816fd7dff69
function set_leave_values!(dw :: DiffWrapper, leave_vals :: AbstractVector )
	for (i,node) in enumerate(Trees.Leaves(dw.tree))
		empty!(node.vals)
		append!(node.vals, leave_vals[i] )
	end
end

# ╔═╡ 8a1443d4-0beb-4f78-8ab4-ed47adf45e8c
#=
begin
	func = x -> [ sum(x.^2); exp( sum( x ) ) ]
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
end
=#

# ╔═╡ bf982874-9ec2-4aa2-87ed-27f3d26d5d51
function prepare_tree!( dw :: DiffWrapper, f :: Function )
	x0 = dw.x0
	vars = dw.vars
	h = _stepsize( dw.stamp )
	
	for node in Trees.Leaves( dw.tree )
		_substitute_symbols!(node, x0, h, vars)
		_set_node_values!( node, f )
	end
end

# ╔═╡ e0154be3-4c57-4ab9-9218-13ab3a4f44b4
#=
begin
	# I used this cell for fiddling around.
	
	f = x -> [ 2 * x[1]^2 + x[2]^3; sum( x ) ]
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
end
=#

# ╔═╡ 90b565b5-0826-4674-836a-376430823cc5
# jacobian(dw)

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
BenchmarkTools = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
Parameters = "d96e819e-fc66-5662-9728-84c9c7592b0a"
StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
Symbolics = "0c5d862f-8b57-4792-8d23-62f2024744c7"

[compat]
BenchmarkTools = "~1.1.1"
Parameters = "~0.12.2"
StaticArrays = "~1.2.7"
Symbolics = "~1.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

[[AbstractTrees]]
git-tree-sha1 = "03e0550477d86222521d254b741d470ba17ea0b5"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.3.4"

[[ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"

[[ArrayInterface]]
deps = ["IfElse", "LinearAlgebra", "Requires", "SparseArrays", "Static"]
git-tree-sha1 = "a71d224f61475b93c9e196e83c17c6ac4dedacfa"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "3.1.18"

[[Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[BenchmarkTools]]
deps = ["JSON", "Logging", "Printf", "Statistics", "UUIDs"]
git-tree-sha1 = "c31ebabde28d102b602bada60ce8922c266d205b"
uuid = "6e4b80f9-dd63-53aa-95a3-0cdb28fa8baf"
version = "1.1.1"

[[ChainRulesCore]]
deps = ["Compat", "LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "f53ca8d41e4753c41cdafa6ec5f7ce914b34be54"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "0.10.13"

[[Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[CommonSolve]]
git-tree-sha1 = "68a0743f578349ada8bc911a5cbd5a2ef6ed6d1f"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.0"

[[Compat]]
deps = ["Base64", "Dates", "DelimitedFiles", "Distributed", "InteractiveUtils", "LibGit2", "Libdl", "LinearAlgebra", "Markdown", "Mmap", "Pkg", "Printf", "REPL", "Random", "SHA", "Serialization", "SharedArrays", "Sockets", "SparseArrays", "Statistics", "Test", "UUIDs", "Unicode"]
git-tree-sha1 = "dc7dedc2c2aa9faf59a55c622760a25cbefbe941"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "3.31.0"

[[CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"

[[CompositeTypes]]
git-tree-sha1 = "d5b014b216dc891e81fea299638e4c10c657b582"
uuid = "b152e2b5-7a66-4b01-a709-34e65c35f657"
version = "0.1.2"

[[ConstructionBase]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f74e9d5388b8620b4cee35d4c5a618dd4dc547f4"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.3.0"

[[DataAPI]]
git-tree-sha1 = "ee400abb2298bd13bfc3df1c412ed228061a2385"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.7.0"

[[DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4437b64df1e0adccc3e5d1adbc3ac741095e4677"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.9"

[[DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[DelimitedFiles]]
deps = ["Mmap"]
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"

[[DiffRules]]
deps = ["NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "85d2d9e2524da988bffaf2a381864e20d2dae08d"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.2.1"

[[Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"

[[Distributions]]
deps = ["FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SparseArrays", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3889f646423ce91dd1055a76317e9a1d3a23fff1"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.11"

[[DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "a32185f5428d3986f47c2ab78b1f216d5e6cc96f"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.8.5"

[[DomainSets]]
deps = ["CompositeTypes", "IntervalSets", "LinearAlgebra", "StaticArrays", "Statistics", "Test"]
git-tree-sha1 = "6cdd99d0b7b555f96f7cb05aa82067ee79e7aef4"
uuid = "5b8099bc-c8ec-5219-889f-1d9e522a28bf"
version = "0.5.2"

[[Downloads]]
deps = ["ArgTools", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"

[[DynamicPolynomials]]
deps = ["DataStructures", "Future", "LinearAlgebra", "MultivariatePolynomials", "MutableArithmetics", "Pkg", "Reexport", "Test"]
git-tree-sha1 = "5e47c4d652ea67652b7c5945c79c46472397d47f"
uuid = "7c1d4256-1411-5781-91ec-d7bc3513ac07"
version = "0.3.18"

[[EllipsisNotation]]
deps = ["ArrayInterface"]
git-tree-sha1 = "8041575f021cba5a099a456b4163c9a08b566a02"
uuid = "da5c29d0-fa7d-589e-88eb-ea29b0a81949"
version = "1.1.0"

[[ExprTools]]
git-tree-sha1 = "b7e3d17636b348f005f11040025ae8c6f645fe92"
uuid = "e2ba6199-217a-4e67-a87a-7c52f15ade04"
version = "0.1.6"

[[FillArrays]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "Statistics"]
git-tree-sha1 = "8c8eac2af06ce35973c3eadb4ab3243076a408e7"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "0.12.1"

[[Formatting]]
deps = ["Printf"]
git-tree-sha1 = "8339d61043228fdd3eb658d86c926cb282ae72a8"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.2"

[[Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"

[[IfElse]]
git-tree-sha1 = "28e837ff3e7a6c3cdb252ce49fb412c8eb3caeef"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.0"

[[InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[IntervalSets]]
deps = ["Dates", "EllipsisNotation", "Statistics"]
git-tree-sha1 = "3cc368af3f110a767ac786560045dceddfc16758"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.5.3"

[[IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "642a199af8b68253517b80bd3bfd17eb4e84df6e"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.3.0"

[[JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "81690084b6198a2e1da36fcfda16eeca9f9f24e4"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.1"

[[LaTeXStrings]]
git-tree-sha1 = "c7f1c695e06c01b95a67f0cd1d34994f3e7db104"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.2.1"

[[LabelledArrays]]
deps = ["ArrayInterface", "LinearAlgebra", "MacroTools", "StaticArrays"]
git-tree-sha1 = "5e38cfdd771c34821ade5515f782fe00865d60b3"
uuid = "2ee39098-c373-598a-b85f-a56591580800"
version = "1.6.2"

[[Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "Printf", "Requires"]
git-tree-sha1 = "a4b12a1bd2ebade87891ab7e36fdbce582301a92"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.6"

[[LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"

[[LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"

[[LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"

[[Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[LinearAlgebra]]
deps = ["Libdl"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[LogExpFunctions]]
deps = ["DocStringExtensions", "LinearAlgebra"]
git-tree-sha1 = "7bd5f6565d80b6bf753738d2bc40a5dfea072070"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.2.5"

[[Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "6a8a2a625ab0dea913aba95c11370589e0239ff0"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.6"

[[Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"

[[Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "4ea90bd5d3985ae1f9a908bd4500ae88921c5ce7"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.0.0"

[[Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"

[[MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"

[[MultivariatePolynomials]]
deps = ["DataStructures", "LinearAlgebra", "MutableArithmetics"]
git-tree-sha1 = "45c9940cec79dedcdccc73cc6dd09ea8b8ab142c"
uuid = "102ac46a-7ee4-5c85-9060-abc95bfdeaa3"
version = "0.3.18"

[[MutableArithmetics]]
deps = ["LinearAlgebra", "SparseArrays", "Test"]
git-tree-sha1 = "3927848ccebcc165952dc0d9ac9aa274a87bfe01"
uuid = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"
version = "0.2.20"

[[NaNMath]]
git-tree-sha1 = "bfe47e760d60b82b66b61d2d44128b62e3a369fb"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "0.3.5"

[[NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"

[[OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "4dd403333bcf0909341cfe57ec115152f937d7d8"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.1"

[[Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "2276ac65f1e236e0a6ea70baff3f62ad4c625345"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.2"

[[Parsers]]
deps = ["Dates"]
git-tree-sha1 = "c8abc88faa3f7a3950832ac5d6e690881590d6dc"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "1.1.0"

[[Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"

[[Preferences]]
deps = ["TOML"]
git-tree-sha1 = "00cfd92944ca9c760982747e9a1d0d5d86ab1e5a"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.2.2"

[[Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "12fbe86da16df6679be7521dfb39fbc861e1dc7b"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.4.1"

[[REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[Random]]
deps = ["Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[RecipesBase]]
git-tree-sha1 = "b3fb709f3c97bfc6e948be68beeecb55a0b340ae"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.1.1"

[[RecursiveArrayTools]]
deps = ["ArrayInterface", "ChainRulesCore", "DocStringExtensions", "LinearAlgebra", "RecipesBase", "Requires", "StaticArrays", "Statistics", "ZygoteRules"]
git-tree-sha1 = "a4bd5d7c4bf7effc1e7ab75d503928082f63bd71"
uuid = "731186ca-8d62-57ce-b412-fbd966d074cd"
version = "2.16.0"

[[Reexport]]
git-tree-sha1 = "5f6c21241f0f655da3952fd60aa18477cf96c220"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.1.0"

[[Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "4036a3bd08ac7e968e27c203d45f5fff15020621"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.1.3"

[[Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "bf3188feca147ce108c76ad82c2792c57abe7b1f"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.7.0"

[[Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "68db32dff12bb6127bac73c209881191bf0efbb7"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.3.0+0"

[[RuntimeGeneratedFunctions]]
deps = ["ExprTools", "SHA", "Serialization"]
git-tree-sha1 = "5975a4f824533fa4240f40d86f1060b9fc80d7cc"
uuid = "7e49a35a-f44a-4d26-94aa-eba1b4ca6b47"
version = "0.5.2"

[[SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"

[[SciMLBase]]
deps = ["ArrayInterface", "CommonSolve", "ConstructionBase", "Distributed", "DocStringExtensions", "IteratorInterfaceExtensions", "LinearAlgebra", "Logging", "RecipesBase", "RecursiveArrayTools", "StaticArrays", "Statistics", "Tables", "TreeViews"]
git-tree-sha1 = "f0bf114650476709dd04e690ab2e36d88368955e"
uuid = "0bca4576-84f4-4d90-8ffe-ffa030f20462"
version = "1.18.2"

[[Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "Requires"]
git-tree-sha1 = "d5640fc570fb1b6c54512f0bd3853866bd298b3e"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "0.7.0"

[[SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"

[[Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "b3363d7460f7d098ca0912c69b082f75625d7508"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.0.1"

[[SparseArrays]]
deps = ["LinearAlgebra", "Random"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[SpecialFunctions]]
deps = ["ChainRulesCore", "LogExpFunctions", "OpenSpecFun_jll"]
git-tree-sha1 = "a50550fa3164a8c46747e62063b4d774ac1bcf49"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "1.5.1"

[[Static]]
deps = ["IfElse"]
git-tree-sha1 = "62701892d172a2fa41a1f829f66d2b0db94a9a63"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "0.3.0"

[[StaticArrays]]
deps = ["LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1b9a0f17ee0adde9e538227de093467348992397"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.2.7"

[[Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[StatsAPI]]
git-tree-sha1 = "1958272568dc176a1d881acb797beb909c785510"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.0.0"

[[StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2f6792d523d7448bbe2fec99eca9218f06cc746d"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.8"

[[StatsFuns]]
deps = ["LogExpFunctions", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "30cd8c360c54081f806b1ee14d2eecbef3c04c49"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "0.9.8"

[[SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[SymbolicUtils]]
deps = ["AbstractTrees", "ChainRulesCore", "Combinatorics", "ConstructionBase", "DataStructures", "DynamicPolynomials", "IfElse", "LabelledArrays", "LinearAlgebra", "MultivariatePolynomials", "NaNMath", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "TimerOutputs"]
git-tree-sha1 = "17fecd52e9a82ca52bfc9d28a2d31b33458a6ce0"
uuid = "d1185830-fcd6-423d-90d6-eec64667417b"
version = "0.13.1"

[[Symbolics]]
deps = ["ConstructionBase", "DiffRules", "Distributions", "DocStringExtensions", "DomainSets", "IfElse", "Latexify", "Libdl", "LinearAlgebra", "MacroTools", "NaNMath", "RecipesBase", "Reexport", "Requires", "RuntimeGeneratedFunctions", "SciMLBase", "Setfield", "SparseArrays", "SpecialFunctions", "StaticArrays", "SymbolicUtils", "TreeViews"]
git-tree-sha1 = "dae26a27018d0cad7efd585a9a0012c6a2752a88"
uuid = "0c5d862f-8b57-4792-8d23-62f2024744c7"
version = "1.4.2"

[[TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"

[[TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "LinearAlgebra", "TableTraits", "Test"]
git-tree-sha1 = "8ed4a3ea724dac32670b062be3ef1c1de6773ae8"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.4.4"

[[Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"

[[Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[TimerOutputs]]
deps = ["ExprTools", "Printf"]
git-tree-sha1 = "209a8326c4f955e2442c07b56029e88bb48299c7"
uuid = "a759f4b9-e2f1-59dc-863e-4aeb61b1ea8f"
version = "0.5.12"

[[TreeViews]]
deps = ["Test"]
git-tree-sha1 = "8d0d7a3fe2f30d6a7f833a5f19f7c7a5b396eae6"
uuid = "a2a6695c-b41b-5b7d-aed9-dbfdeacea5d7"
version = "0.3.0"

[[UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"

[[ZygoteRules]]
deps = ["MacroTools"]
git-tree-sha1 = "9e7a1e8ca60b742e508a315c17eef5211e7fbfd7"
uuid = "700de1a5-db45-46bc-99cf-38207098b444"
version = "0.2.1"

[[nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"

[[p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
"""

# ╔═╡ Cell order:
# ╟─7b914a4f-89b4-4284-b8c6-d629157b9a7c
# ╠═e7d76f0e-eea5-11eb-1bdf-9196c513ff0a
# ╠═44f2f27b-2ac6-4df1-b222-4cfa37f41985
# ╟─10b9b062-450e-43a2-9c7a-2fe6c7121ab7
# ╟─bd83fc2a-d7bd-489e-83f3-ef5b327d5cca
# ╠═a7d81ac5-0124-441a-8eae-f83cee2b81c3
# ╟─fdcf00a4-836e-4404-ba1c-6058ae85d9ce
# ╠═aeac724f-b87a-4f37-aa57-893d1d131f6d
# ╟─07e698bb-fac0-4951-85dc-5adb208cf5dd
# ╠═9462b310-0842-4373-ac2c-f2eaef5e6e7f
# ╠═c19325b6-7516-4acf-a5f1-fe7084bc113d
# ╠═2426f92a-21f3-4455-aeb7-fe6674aafe02
# ╠═4ebb80a7-539d-49a0-981f-166f0bbf3964
# ╟─a5e02956-c0f9-44a6-80da-abfb497bcfb9
# ╠═09d78bbc-f568-459a-8bd0-05de84b04af2
# ╟─04861d1b-38d9-4ba7-9fde-b10150f39280
# ╟─20907685-2bb6-49d5-9ff0-a175435fa591
# ╟─65d0f18a-6da9-4836-bf4c-8ff10e6a265f
# ╠═c3fa417a-898f-4273-a75c-974a42e80a79
# ╟─6cdc6c1e-790e-4d2a-984c-36f5f0c98f63
# ╠═94d3947d-fb1c-4e8d-9b16-bdf14a411193
# ╟─dc4d3c61-f2ec-4fb1-b84b-2c1957da3a7e
# ╠═c2cb0dd1-e36d-44ff-aa59-6365a15e7ab6
# ╠═aabc4fed-e62f-4668-9984-c9b72cad15e9
# ╟─3f67de97-2a12-4ca7-82bd-16851f958989
# ╠═18e51761-98ae-4814-b575-3bb4a40a06cb
# ╠═d21fed2f-f61a-42d6-aed3-430bf73fd170
# ╟─a5565a82-2b6d-4c43-909c-9bfeca9890cb
# ╠═17347e0e-08ee-4a29-83b3-f337e397f49b
# ╟─1db29942-3ff0-49e8-b5c6-2480eadd608f
# ╠═af650705-0636-4728-a643-ab658d362290
# ╠═7c091ff3-7d34-4ad8-b959-fe601397f091
# ╠═a09ab593-8efd-4799-8f0f-aa2ceb1e135d
# ╟─59211507-8901-4f4c-b2ab-c5d4fb950bf1
# ╟─eb61f081-bd96-4a25-b2ba-9a3bc9e87934
# ╠═5253756b-1a03-4d8f-854a-b03c5f396dc9
# ╟─f77f8375-5b8f-4a3e-b571-28ebccb204c4
# ╟─fe8171dc-7efd-4acb-b1f9-0da5f1c6dc9d
# ╠═dfc93b7e-980d-4f35-9d72-16cee2c747c2
# ╠═a728763c-8995-4d09-9878-1db83afc9194
# ╠═2f50ce22-7a20-4b24-ac57-5f3c3918079e
# ╠═98051445-7ffd-4a92-af38-07507efab445
# ╠═634067d4-15db-4aa5-8b5a-b0794ee88a39
# ╟─5dd33903-f4a4-455c-8348-c3c98e5f6404
# ╠═e0154be3-4c57-4ab9-9218-13ab3a4f44b4
# ╠═90b565b5-0826-4674-836a-376430823cc5
# ╟─8ad29cb3-9bb5-48d6-b0a6-ceb46e30293e
# ╠═351d4199-28fa-4733-89d1-266ee16137c4
# ╠═fef9850c-3cbd-47f8-a786-e420f01d9141
# ╠═0be3107e-a3c9-46f5-837d-c7f6d2b2d595
# ╠═42e6c63e-0525-4ec6-88b9-b077130603ff
# ╠═dbdceb0d-e4bb-406c-b3cf-defffcc1ccf9
# ╠═8c11b8e1-c0fb-48b8-b9f1-8679e3d0e596
# ╠═df8bd829-5c28-46c1-b171-1b5dc0358c76
# ╠═ac3d101b-c9d1-4fdb-80a4-645464298ba3
# ╠═7b725d45-f6fa-4bd8-99d7-fcad166320e5
# ╠═9cb6592b-ba7d-4b46-9171-1816fd7dff69
# ╠═bf982874-9ec2-4aa2-87ed-27f3d26d5d51
# ╟─5d586923-a32e-4d16-a0a6-56bbc73680f2
# ╠═3feed8b1-1338-47c7-960f-892481f96a5c
# ╠═8a1443d4-0beb-4f78-8ab4-ed47adf45e8c
# ╟─d41308c0-b7f1-4ed9-b27e-7c6fe87080e6
# ╠═6c6e63ae-28ce-4c1c-bb60-7e7826ffefbb
# ╟─f294835d-ee9a-42b1-b2b7-5816d65e0f18
# ╟─e9af48fd-ac05-4358-8a33-c14008ca1cb9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
end