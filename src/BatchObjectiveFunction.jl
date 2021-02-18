# Custom function to allow for batch evaluation outside of julia
# or to exploit parallelized objective functions
# (included from "AbstractObjectiveInterface")
@with_kw struct BatchObjectiveFunction <: Function
    function_handle :: Union{Function, Nothing} = nothing
end

# evaluation of a BatchObjectiveFunction
function (objf::BatchObjectiveFunction)(args...)
    objf.function_handle(args...)
end

# overload broadcasting for BatchObjectiveFunction's
# that are assumed to handle arrays themselves
function Broadcast.broadcasted( objf::BatchObjectiveFunction, args... )
    objf.function_handle( args... )
end

function _new_batch( func1 :: F, func2 :: T ) where{ F <:Function ,T <: Function }
    return BatchObjectiveFunction(
        function( x :: Union{ RVecArr, RVec } )
            if isa( x, Rvec )
                [ func1(x); func2(x) ]
            else
                f1 = func1.(x)
                f2 = func2.(x)
                [ [f1[i];f2[i]] for i = eachindex(x) ]
            end
        end
    )
end

combine( func1 :: Function, func2 :: BatchObjectiveFunction ) = _new_batch( func1, func2)
combine( func1 :: BatchObjectiveFunction, func2 :: Function) = _new_batch( func1, func2 )
combine( func1 :: BatchObjectiveFunction, func2 :: BatchObjectiveFunction ) = _new_batch( func1, func2 )

"Get a new function function handle stacking the output of `func1` and `func2`."
function combine(func1 :: F, func2 :: T ) where{ F <:Function ,T <: Function }
    return function( x :: Vector{R} ) where{R <: Real}
        [ func1(x); func2(x) ]
    end
end