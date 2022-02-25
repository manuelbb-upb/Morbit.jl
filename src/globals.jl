const Vec = AbstractVector{<:Real}
const VecVec = AbstractVector{<:AbstractVector}
const NumOrVec = Union{Real, Vec}
const VecOrNum = NumOrVec 
const Mat = AbstractMatrix{<:Real}

const VecF = AbstractVector{<:AbstractFloat}
const MatF = AbstractMatrix{<:AbstractFloat}
const NumOrVecF = Union{AbstractFloat, VecF}

const MIN_PRECISION = Float32

const LP_OPTIMIZER = OSQP.Optimizer
# ## Concrete Types 
# These types do not depend on anything else, but are important for 
# all the other stuff:

import MathOptInterface
const MOI = MathOptInterface # import X as Y syntax not supported by Julia 1.5
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

struct ModelGrouping{T}
    indices :: Vector{NLIndex}
    cfg :: T
end

function _contains_index( m :: ModelGrouping, ind )
    return ind in m.indices
end

struct CountedFunc{T,F} <: Function
    func :: F
    counter :: Base.RefValue{Int64}
    
    function CountedFunc( func :: F; can_batch = false ) where F
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

# ### Enums

# These codes should be availabe everywhere and that is why we 
# define them here:

@enum ITER_TYPE begin
    ACCEPTABLE     # accept trial point, shrink radius 
    SUCCESSFULL    # accept trial point, grow radius 
    MODELIMPROVING # reject trial point, keep radius 
    INACCEPTABLE   # reject trial point, shrink radius (much)
    RESTORATION    # apart from the above distinction: a restoration step has been computed and used as the next iterate
    FILTER_FAIL    # trial point is not acceptable for filter
    FILTER_ADD     # trial point acceptable to filter with large constraint violation
    EARLY_EXIT
#    CRIT_LOOP_EXIT
    INITIALIZATION
end

@enum STOP_CODE begin
    CONTINUE = 1
    MAX_ITER = 2
    BUDGET_EXHAUSTED = 3
    CRITICAL = 4
    TOLERANCE = 5 
    INFEASIBLE = 6
end

@enum RADIUS_UPDATE begin 
    LEAVE_UNCHANGED 
    GROW
    SHRINK
    SHRINK_MUCH 
end

const NLOPT_SUCCESS_CODES = [
    :SUCCESS, 
    :STOPVAL_REACHED, 
    :FTOL_REACHED,
    :XTOL_REACHED,
    :MAXEVAL_REACHED, 
    :MAXTIME_REACHED
]

const NLOPT_FAIL_CODES = [
    :FAILURE,
    :INVALID_ARGS,
    :OUT_OF_MEMORY,
    :ROUNDOFF_LIMITED,
    :FORCED_STOP
]