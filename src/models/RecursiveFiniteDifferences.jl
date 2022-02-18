module RecursiveFiniteDifferences

# This is a stripped down version of the Pluto notebook found in the docs.
include(joinpath(@__DIR__, "Trees.jl"))
using .Trees

using StaticArrays
using Symbolics
using Symbolics: Sym
using Parameters

const RVec = AbstractVector{<:Real}
const RVecOrR = Union{Real, RVec}

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


begin
	stepsize( x :: F ) where F <: AbstractFloat = 10 * sqrt(eps(F))
	stepsize( F :: Type{<:AbstractFloat} ) = 10 * sqrt(eps(F))
	stepsize( x :: AbstractVector{F} ) where F<:AbstractFloat = stepsize(F)
	stepsize( x ) = stepsize(Float64)
end

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
"Evaluate all outputs of `f` with respect to ``x_i``." 
function dx(stamp :: FiniteDiffStamp, f :: Function, x0 :: RVec, i = 1)
	h = _stepsize(stamp)
	grid = [ [x0[1:i-1];x0[i] + h*g; x0[i+1:end]] for g in _grid(stamp) ]
	evals = f.(grid)
	return stamp( evals )
end

begin
	cfd = CFDStamp(1,2)
	Func = x -> x[1]*x[2]^2
	
	# ∂₁Func = x[2]^2
	# ∂₂Func = x[1]
	dx( cfd, Func, [1,2], 1)
end





begin
	missing(T::Type{<:AbstractFloat}) = T(NaN)
	missing(T) = Float16(NaN)
end

function val( subnode_list, indices, stamp, output_index )
	coeff = _coeff( stamp )
	h = _stepsize( stamp )
	return stamp( val(sub_node, indices, stamp; output_index) for sub_node in subnode_list ) 
end


begin
	"Helper function to get Symbolic variables for `x_1,…,x_n` and the stepsize `h`." 
	function _get_vars(n :: Int)
		return Tuple(Num.(Symbolics.variable.(:x, 1:n))), Num(Symbolics.variable(:h))
	end
	
	_get_vars( x :: AbstractVector ) = _get_vars(length(x))
end
begin
	vec_typex( x :: AbstractVector{Y}, stamp ) where Y = Vector{ Base.promote_op( +, Y, typeof(_stepsize(stamp))) }
	vec_typef( fx :: AbstractVector{Y}, stamp ) where Y = Vector{ Base.promote_op( *, Y, typeof(_coeff(stamp)[1])) }
end


function _substitute_symbols!(root_iterator, x0, h, vars)
	for node in root_iterator
		_substitute_symbols!(node, x0, h, vars)
	end
end

function _set_node_values!( node, f :: Function )
	empty!(node.vals)
	append!(node.vals, f( node.x ))
end

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
# Trees = ingredients( joinpath(@__DIR__, ".." ,"src", "Trees.jl") )
@with_kw struct FDiffNode{T,X,C} <: Trees.Node
	x_sym :: Vector{Symbolics.Num} = []
	x :: X = Float64[]
	vals :: T = nothing
	cache :: C = nothing
end

function val( n :: FDiffNode{<:AbstractVector{Y},X,C}, args...; output_index = 1) where {Y<:Real,X,C}
	if isempty(n.vals) 
		return missing(Y)
	else 
		n.vals[output_index]
	end
end

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
val( dw :: DiffWrapper, indices; output_index = 1 ) = val( dw.tree, indices, dw.stamp ; output_index)
function gradient( dw :: DiffWrapper; output_index = 1 )
	@assert 1 <= dw.order <= 2 "Gradient retrieval only implemented for DiffWrapper of order 1 and 2."
	return gradient( dw :: DiffWrapper, Val(dw.order); output_index )
end

function gradient( dw :: DiffWrapper, ::Val{1}; output_index = 1 )
	n = length( dw.x0 )
	g = Vector{eltype(dw.fx0)}(undef, n)
	for i = 1 : n
		g[i] = val( dw, [i,]; output_index )
	end
	return g
end

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

function substitute_symbols!(dw :: DiffWrapper)
	substitute_symbols!( dw.tree, dw.x0, _stepsize( dw.stamp ), dw.vars )
end

function substitute_leaves!(dw :: DiffWrapper)
	substitute_leaves!( dw.tree, dw.x0, _stepsize( dw.stamp ), dw.vars )
end

function _substitute_symbols!( node :: FDiffNode, x0, h, vars )
	x_vars, h_var = vars
	empty!(node.x)
	append!(node.x, Symbolics.value.(substitute.(node.x_sym, (
		Dict((x_vars[i] => x0[i] for i = eachindex(x0))..., 
					h_var=>h),
	))))
end
begin
	Trees.children( n :: FDiffNode{<:RVec} ) = nothing
	Trees.children( n :: FDiffNode ) = Iterators.flatten( n.vals )
end

function jacobian( dw :: DiffWrapper )
	@assert 1 <= dw.order <= 2 "Gradient retrieval only implemented for DiffWrapper of order 1 and 2."
	k = isempty( dw.fx0 ) ? length(first(Trees.Leaves(dw.tree)).vals) : length(dw.fx0)
	return transpose( hcat( collect(gradient(dw; output_index = m) for m = 1 : k )...) )
end

function substitute_symbols!(root :: FDiffNode, x0, h, vars )
	_substitute_symbols!( Trees.PreOrderDFS( root ), x0, h, vars )
end

function substitute_leaves!(root :: FDiffNode, x0, h, vars )
	_substitute_symbols!( Trees.Leaves(root), x0, h, vars )
end

function collect_leave_sites( dw :: DiffWrapper )
	return [ node.x for node in Trees.Leaves( dw.tree ) ]
end

function set_leave_values!(dw :: DiffWrapper, f :: Function )
	for node in Trees.Leaves(dw.tree)
		_set_node_values!(node, f)
	end
end

function set_leave_values!(dw :: DiffWrapper, leave_vals :: AbstractVector )
	for (i,node) in enumerate(Trees.Leaves(dw.tree))
		empty!(node.vals)
		append!(node.vals, leave_vals[i] )
	end
end
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
function prepare_tree!( dw :: DiffWrapper, f :: Function )
	x0 = dw.x0
	vars = dw.vars
	h = _stepsize( dw.stamp )
	
	for node in Trees.Leaves( dw.tree )
		_substitute_symbols!(node, x0, h, vars)
		_set_node_values!( node, f )
	end
end
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
# jacobian(dw)
end