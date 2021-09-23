const Vec = AbstractVector{<:Real}
const VecVec = AbstractVector{<:AbstractVector}
const NumOrVec = Union{Real, Vec}
const VecOrNum = NumOrVec 
const Mat = AbstractMatrix{<:Real}

const Vec64 = Vector{Float64}
const NumOrVec64 = Union{Float64, Vec64}

const VecF = AbstractVector{<:AbstractFloat}
const MatF = AbstractMatrix{<:AbstractFloat}

const NumOrVecF = Union{AbstractFloat, VecF}
const MIN_PRECISION = Float32

#%%

# used instead of list comprehension
# works with vectors of vectors too:
flatten_vecs( x :: Number) = [x,]

function flatten_vecs(x)
	return [ e for e in Iterators.flatten(x) ]
end

function mat_from_row_vecs( row_vecs )
	return copy( transpose( hcat(row_vecs...) ) )
end

# ## Concrete Types 
# These types do not depend on anything else, but are important for 
# all the other stuff:

struct VarInd
	val :: Int 
end

struct ObjectiveIndex
    val :: Int
    num_out :: Int 

    ObjectiveIndex( val :: Int, num_out :: Int = 1 ) = new(val, num_out)
end

struct EqConstraintIndex
    val :: Int
    num_out :: Int
    
    EqConstraintIndex( val :: Int, num_out :: Int = 1 ) = new(val, num_out)
end

struct IneqConstraintIndex
    val :: Int
    num_out :: Int
    
    IneqConstraintIndex( val :: Int, num_out :: Int = 1 ) = new(val, num_out)
end

const FunctionIndex = Union{ObjectiveIndex, EqConstraintIndex, IneqConstraintIndex}
Base.broadcastable( ind :: FunctionIndex ) = Ref(ind)

Base.isless(fi :: FunctionIndex, fi2 :: FunctionIndex) = isless(fi.val, fi2.val)

const FunctionIndexTuple = Tuple{Vararg{<:FunctionIndex}}
const FunctionIndexIterable = Union{FunctionIndexTuple, Vector{<:FunctionIndex}}

num_outputs( fi :: FunctionIndex ) = fi.num_out
function num_outputs( indices :: FunctionIndexIterable)
    isempty(indices) && return 0
    return sum( num_outputs(fi) for fi in indices )
end

function _split( indices :: FunctionIndexIterable )
    arr1 = ObjectiveIndex[]
    arr2 = EqConstraintIndex[]
    arr3 = IneqConstraintIndex[]
    for ind in indices 
        if ind isa ObjectiveIndex 
            push!(arr1, ind)
        elseif ind isa EqConstraintIndex
            push!(arr2, ind)
        else
            push!(arr3, ind)
        end
    end
    return arr1, arr2, arr3
end

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

###################################################
_ensure_vec( x :: Number ) = [x,]
_ensure_vec( x :: AbstractVector{<:Number} ) = x

struct VecFuncWrapper{T,F<:Function} <: Function
    func :: F
    counter :: Base.RefValue{Int64}
    
    function VecFuncWrapper( func :: F; can_batch = false ) where F<:Function
        return new{can_batch, F}( func, Ref(0))
    end
end

# evaluation of a BatchObjectiveFunction
function (F::VecFuncWrapper)(x :: Vec)
    F.counter[] += 1
    return _ensure_vec(F.func(x))
end

# overload broadcasting for BatchObjectiveFunction's
# that are assumed to handle arrays themselves
function Broadcast.broadcasted( F :: VecFuncWrapper{true,<:Function}, X :: VecVec)
    F.counter[] += length(X)
    return _ensure_vec.(F.func( X ))
end

################################################
_typename( T :: DataType ) = T.name.name
_typename( T :: UnionAll ) = _typename(T.body)