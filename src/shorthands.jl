const Vec = AbstractVector{<:Real}
const VecVec = AbstractVector{<:AbstractVector}
const NumOrVec = Union{Real, Vec}
const VecOrNum = NumOrVec 
const Mat = AbstractMatrix{<:Real}

const VecF = AbstractVector{<:AbstractFloat}
const MatF = AbstractMatrix{<:AbstractFloat}

const NumOrVecF = Union{AbstractFloat, VecF}
const MIN_PRECISION = Float32

#%%

# used instead of list comprehension
# works with vectors of vectors too:
flatten_vecs( x :: Number) = [x,]

function flatten_vecs(x)
	return [ e for e in Iterators.flatten(x) ]  # TODO check performance
end

function mat_from_row_vecs( row_vecs )
	return copy( transpose( hcat(row_vecs...) ) )
end

# ## Concrete Types 
# These types do not depend on anything else, but are important for 
# all the other stuff:

import MathOptInterface as MOI
const VarInd = MOI.VariableIndex

struct ObjectiveIndex
    value :: Int
    num_out :: Int 

    ObjectiveIndex( val :: Int, num_out :: Int = 1 ) = new(val, num_out)
end

struct ConstraintIndex
    value :: Int
    num_out :: Int
    
    type :: Symbol
    
    function ConstraintIndex( val :: Int, num_out :: Int = 1, type :: Symbol = :eq )
        @assert type in [:eq, :ineq, :nl_eq, :nl_ineq]
        new(val, num_out, type)
    end
end

struct NLIndex
    value :: Int
    num_out :: Int
end 

const FunctionIndex = Union{ObjectiveIndex, ConstraintIndex}
const AnyIndex = Union{FunctionIndex, NLIndex}

Base.broadcastable( ind :: AnyIndex ) = Ref(ind)

#Base.isless(fi :: FunctionIndex, fi2 :: FunctionIndex) = isless(fi.value, fi2.value)

const FunctionIndexTuple = Tuple{Vararg{<:FunctionIndex}}
const FunctionIndexIterable = Union{FunctionIndexTuple, Vector{<:FunctionIndex}}
const AnyIndexTuple = Tuple{Vararg{<:AnyIndex}}
const AnyIndexIterable = Union{AnyIndexTuple, AbstractVector{<:AnyIndex}}
const NLIndexTuple = Tuple{Vararg{<:NLIndex}}

num_outputs( fi :: AnyIndex ) = fi.num_out
function num_outputs( indices :: AnyIndexIterable)
    isempty(indices) && return 0
    return sum( num_outputs(fi) for fi in indices )
end

#=
function _split( indices :: FunctionIndexIterable )
    arr1 = ObjectiveIndex[]
    arr2 = ConstraintIndex[]
    arr3 = ConstraintIndex[]
    for ind in indices 
        if ind isa ObjectiveIndex 
            push!(arr1, ind)
        else
            if ind.type == :nl_eq
                push!(arr2, ind)
            elseif ind.type == :nl_ineq
                push!(arr3, ind)
            end
        end
    end
    return arr1, arr2, arr3
end
=#

#=
function func_index_and_relative_position_from_func_indices( func_indices :: FunctionIndexIterable, out_pos :: Int )
    counter = 1
    for ind in func_indices
        next_counter = counter + num_outputs(ind)
        if counter <= out_pos < next_counter
            return ind, out_pos - counter + 1
        end
        counter = next_counter
    end
    return nothing
end
=#
###################################################
_ensure_vec( x :: Number ) = [x,]
_ensure_vec( x :: AbstractVector{<:Number} ) = x

struct CountedFunc{T,F<:Function} <: Function
    func :: F
    counter :: Base.RefValue{Int64}
    
    function CountedFunc( func :: F; can_batch = false ) where F<:Function
        return new{can_batch, F}( func, Ref(0))
    end
end

_can_batch( fn :: CountedFunc{T,F} ) where{T,F} = T 

# evaluation of a BatchObjectiveFunction
function (F::CountedFunc)(x :: Vec)
    F.counter[] += 1
    return _ensure_vec(F.func(x))
end

# overload broadcasting for BatchObjectiveFunction's
# that are assumed to handle arrays themselves
function Broadcast.broadcasted( F :: CountedFunc{true,<:Function}, X :: VecVec)
    F.counter[] += length(X)
    return _ensure_vec.(F.func( X ))
end

################################################
_typename( T :: DataType ) = T.name.name
_typename( T :: UnionAll ) = _typename( T.body )

##################################################

# linear constraints
# inspired by MathOptInterface