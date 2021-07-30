# This file has an abstract interface for a Tree structure 
# and some traversal iterators.
# I wrote it for another project, but use it for the Finite-Difference Taylor models here.

module Trees 

abstract type Node end

# methods that can/should be customized
children(n :: N ) where{N<:Node}= isdefined( n, :children ) ? n.children : N[];
parent( n :: Node ) = isdefined( n, :parent ) ? n.parent : nothing;

@doc """
	depth(node)

Return depth of `node` in a tree.
Root node is assigned depth 0, its children have depth 1 and so forth.

This default implementation goas up the tree starting at `node` as long 
as there is a parent node. \n 
It can be overwritten for custom Node/Tree types, e.g. when depth is 
stored explicitly.
"""
function depth( node :: Node )
	depth = 0;
	iter_node = parent(node);
	while !isnothing( iter_node )
		depth += 1;	
		iter_node = parent(iter_node)
	end
	return depth
end

# DFS Iterator 
@doc """
	PreOrderDFS{T}
		start_node :: T
		traverse_subtree_filter :: Union{Nothing, F where F<:Function};
	end

An iterator to perform a depth first traversal of the subtree that has 
`start_node` as its root.
PreOrder means that parent nodes are visited before their respective children.
The field `traverse_subtree_filter` defaults to `nothing`.
If a function is provided it should accept an argument of type `T` (a parent node)
and return `true` if its children should be pushed to the stack for visiting.
"""
struct PreOrderDFS{T}
	start_node :: T
	traverse_subtree_filter :: Union{Nothing, F where F<:Function};
end
# default constructor without a filter: 
PreOrderDFS( sn :: T ) where T = PreOrderDFS{T}( sn, nothing );

Base.IteratorSize( ::PreOrderDFS{T} ) where T = Base.SizeUnknown();

function _children_to_stack( node :: T ) where T
    node_children = children( node )
    if isnothing( node_children )
        return T[]
    else
        return reverse(collect(node_children))
    end
end

function Base.iterate( iter :: PreOrderDFS{T} ) where T
	# in first iteration, collect all shildren as candidates to visit 
	# `reverse` to imitate stack behavior
	init_stack = _children_to_stack( iter.start_node )
	(iter.start_node, init_stack); 
end

function Base.iterate( iter :: PreOrderDFS{T}, stack :: AbstractVector ) where T
	if isempty( stack )
		return nothing
	else
		return_node = pop!(stack)
		if ( 
				isnothing( iter.traverse_subtree_filter ) || 
				iter.traverse_subtree_filter(return_node) 
			)
			push!( stack, _children_to_stack(return_node)... );
		end
		return (return_node, stack );
	end
end

@doc "An abstract super type for iterators that are based on the 
PreOrderDFS iterator."
abstract type PreOrderDFSSubIterator end;
Base.IteratorSize( ::PreOrderDFSSubIterator ) = Base.SizeUnknown();

# `node_filter` should be implemented by Iterators subtyping PreOrderDFSSubIterator
function node_filter( ::PreOrderDFSSubIterator ) end;

# Iteration procedure for every `PreOrderDFSSubIterator`
# (including the special cases like `Leaves` and `DepthFilterNodes` etc.)
function Base.iterate( iter::PreOrderDFSSubIterator, stack = nothing )
	if isnothing(stack)
		# first iteration …
		# initilize using the base iterator `iter.dfs_iterator`
		node, stack = iterate( iter.dfs_iterator );
	else
		if isempty(stack)
			# no more nodes left to visit; return
			return nothing
		else
			# let `iter.dfs_iterator` handle iteration 
			# this takes care of "subtree filtering"
			node, stack = iterate( iter.dfs_iterator, stack);
		end
	end
	next = (node, stack);
	# as long as there is a non-nothing *next* …
	while !isnothing( next )
		node, stack = next;
		if node_filter( iter, node )
			# … return the node if it is deemed suitable …
			return node, stack 
		end
		# … and progress iteration
		next = iterate( iter.dfs_iterator, stack )
	end
	return nothing
end

@doc """
	PreOrderDFSFiltered{T} <: PreOrderDFSSubIterator

An iterator that is based on the `PreOrderDFS` iterator and allows
for filtering out nodes that are returned.
The most basic constructor is 
```
iter = PreOrderDFSFiltered( 
	start_node :: T, 
	func :: F where F <: Function )
```
The function `func` should take one argument of type `T` (a node) 
and return true if it should be returned by `iter`.
"""
struct PreOrderDFSFiltered{T} <: PreOrderDFSSubIterator
	start_node :: T
	dfs_iterator :: Union{Nothing, PreOrderDFS{T}}
	node_filter :: F where F<:Function
end

# outer constructor without `dfs_iterator` provided
function PreOrderDFSFiltered( sn::T, fn :: F where F<:Function ) where T 
	PreOrderDFSFiltered{T}( sn, PreOrderDFS(sn), fn )
end

# set default behavior for `node_filter` for type `PreOrderDFSFiltered`
node_filter( iter::PreOrderDFSFiltered, node ) = iter.node_filter( node )

@doc """
	Leaves{T} <: PreOrderDFSSubIterator

Initialize as per
```iter = Leaves( start_node )```
`iter` is an iterator that returns the leaves of a subtree 
starting at `start_node`. 
It is based on a PreOrderDFS traversal.
"""
struct Leaves{T} <: PreOrderDFSSubIterator
	start_node :: T
	dfs_iterator :: Union{Nothing, PreOrderDFS{T}}
end
Leaves( sn :: T ) where T = Leaves{T}( sn, PreOrderDFS( sn ) );

# determine a `node` a leave it it has no children
node_filter( :: Leaves, node ) = let childs = children(node);
    return isnothing(childs) || isnothing( iterate( childs ) )
end

@doc """
	DepthFilterNodes{T} <: PreOrderDFSSubIterator 

An iterator based on `PreOrderDFS` that returns only nodes
with a certain depth(s). Traversal starts at `start_node` but depth 
is measured by `depth( any_node )` and hence in relation to the 
root of the tree (which possibly does not equal `start_node`) by default.\n
A subtree is only visited for traversal if the depth of its respective
parent is below the maximum value of desired return depths.

Initialize via 
```
dfs_iterator = DepthFilterNodes( start_node, desired_depths :: Vector{Int} );
```
"""
struct DepthFilterNodes{T} <: PreOrderDFSSubIterator
	start_node :: T
	dfs_iterator :: Union{Nothing, PreOrderDFS{T}}
	desired_depths :: Union{Nothing, Vector{Int}}
end
function DepthFilterNodes( sn :: T ) where T 
	return PreOrderDFS(sn)
end
function DepthFilterNodes( sn :: T, desired_depths :: Vector{Int64} ) where T 
	max_depth = maximum( desired_depths )

	# setup a filter function for the base iterator so that 
	# only substrees are visited where the respective parent 
	# has sufficiently small depth
	children_filter = function( parent_node :: T )
		return depth(parent_node) < max_depth
	end
	base_iterator = PreOrderDFS( sn, children_filter );

	DepthFilterNodes{T}( sn, base_iterator, desired_depths );
end

function node_filter(iter::DepthFilterNodes, node )
	depth(node) in iter.desired_depths 
end

#=
begin
	struct MyTree <: Node
		data :: Int64
		parent :: Union{Nothing, MyTree}
		children :: Vector{MyTree}
	end
	
	children( n::MyTree ) = n.children;
	parent( n :: MyTree ) = n.parent;
	
	function add_child!( parent :: MyTree, child_data :: Int64 )
		child_node = MyTree( child_data, parent, [] );
		push!( parent.children, child_node )
		return child_node
	end
	
	root_node = MyTree( 0, nothing, [] )
	
	child_l1_1 = add_child!( root_node, 1 )
	child_l1_2 = add_child!( root_node, 2 )
	child_l2_1 = add_child!( child_l1_1, 3 )
	child_l2_2 = add_child!( child_l1_1, 4 )
	
	@show [n.data for n in DepthFilterNodes(root_node, collect(0:2))];
end
=#

end