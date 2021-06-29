# Custom function to allow for batch evaluation outside of julia
# or to exploit parallelized objective functions
# (included from "AbstractObjectiveInterface")
struct BatchObjectiveFunction{F<:Function} <: Function
    function_handle :: F
end

# evaluation of a BatchObjectiveFunction
function (objf::BatchObjectiveFunction)(x :: Vec)
    vec(objf.function_handle(x))
end

# overload broadcasting for BatchObjectiveFunction's
# that are assumed to handle arrays themselves
function Broadcast.broadcasted( objf::BatchObjectiveFunction, X :: VecVec)
    vec.( objf.function_handle( X ) )
end

function _new_batch( func1 :: F, func2 :: T ) where{ F <:Function ,T <: Function }
    return BatchObjectiveFunction(
        function( x :: Union{ VecVec, Vec } )
            if isa( x, Vec )
                [ func1(x); func2(x) ]
            else
                f1 = func1.(x)
                f2 = func2.(x)
                [ [f1[i];f2[i] ] for i = eachindex(x) ]
            end
        end
    )
end

combine( func1 :: Function, func2 :: BatchObjectiveFunction ) = _new_batch( func1, func2)
combine( func1 :: BatchObjectiveFunction, func2 :: Function) = _new_batch( func1, func2 )
combine( func1 :: BatchObjectiveFunction, func2 :: BatchObjectiveFunction ) = _new_batch( func1, func2 )

"Get a new function function handle stacking the output of `func1` and `func2`."
function combine(func1 :: F, func2 :: T ) where{ F <:Function ,T <: Function }
    return function( x :: Vec )
        return vcat( func1(x), func2(x) )
    end
end