# depends on abstract type SurrogateConfig & combinable( :: SurrogateConfig )

Broadcast.broadcastable( objf::AbstractObjective ) = Ref( objf );

include("BatchObjectiveFunction.jl");

# MANDATORY METHODS
"A general constructor."
function _wrap_func( :: Type{<:AbstractObjective}, fn :: Function, 
    model_cfg :: SurrogateConfig, n_vars :: Int, n_out :: Int ) :: AbstractObjective
    nothing
end

"Return a function that evaluates an objective at a **scaled** site."
eval_handle(::AbstractObjective) :: Function = nothing;
# NOTE usually, the user provides a function that takes a 
# vector from the **unscaled** domain, not [0,1]^n.
# That is why `_add_objective!` (in AbstractMOPInterface.jl)
# wraps the user provided function handle in a transform function.

num_vars( :: AbstractObjective ) = nothing :: Int;

"Number of calls to the original objective function."
num_evals( :: AbstractObjective ) = 0 :: Int;
"Set evaluation counter to `N`."
num_evals!( :: AbstractObjective, N :: Int) = nothing :: Nothing;

"Return surrogate configuration used to model the objective internally."
model_cfg( :: AbstractObjective ) = nothing <:SurrogateConfig;

"Combine two objectives. Only needed if `combinable` can return true."
combine( ::AbstractObjective, :: AbstractObjective ) = nothing <:AbstractObjective;

num_outputs( objf :: AbstractObjective ) = nothing :: Int;

# DERIVED methods and defaults
# can_batch( ::AbstractObjective ) = false :: Bool;

"Evaluate the objective at scaled site(s). and increase counter."
function eval_objf(objf :: AbstractObjective, x̂ :: RVec )
    inc_evals!(objf);
    eval_handle(objf)(x̂)
end

function Broadcast.broadcasted( ::typeof(eval_objf), objf :: AbstractObjective, X̂ :: RVecArr)
    inc_evals!(objf, length(X̂))
    eval_handle(objf).(X̂)
end

# Helpers to retrieve function handles that increase the eval count:
# … using Memoization here so that always the same function is returned
# this should speed up automatic differentiation 
@memoize ThreadSafeDict function _eval_handle(objf :: AbstractObjective)
    x -> eval_objf(objf, x)
end

@memoize ThreadSafeDict function _eval_handle( objf :: AbstractObjective, ℓ :: Int)
    return x -> eval_objf( objf, x)[ℓ]
end
# NOTE _eval_handle increases eval count, eval_handle does **not** increase count
    
"(Soft) upper bound on the number of function calls. "
max_evals( objf :: AbstractObjective) = max_evals( model_cfg(objf) );
"Set upper bound of № evaluations to `N`"
max_evals!( :: AbstractObjective, N :: Int ) = max_evals!( model_cfg(objf), N );

"Increase evaluation count by `N`"
function inc_evals!( objf :: AbstractObjective, N :: Int = 1 )
    num_evals!( objf, num_evals(objf) + N )
end

#=
"Evaluate the objective at provided site(s)."
function (objf :: AbstractObjective )(x :: RVec )
    inc_evals!( objf );
    return eval_objf( objf, x );
end
=#

combinable( objf :: AbstractObjective ) = combinable( model_cfg(objf) );
function combinable( objf1 :: AbstractObjective, objf2 :: AbstractObjective )
    return combinable( objf1 ) && combinable( objf2 ) && 
        model_cfg( objf1 ) == model_cfg( objf2 )
end

# generic combine function for abstract objectives
function combine( objf1 :: T, objf2 :: T ) where{T<:AbstractObjective}
    new_fn = combine( eval_handle( objf1 ), eval_handle( objf2 ) );
    n_out = num_outputs( objf1 ) + num_outputs( objf2 );
    new_config = combine( model_cfg(objf1), model_cfg(objf2) )
    return _wrap_func( T, new_fn, new_config, num_vars(objf1), n_out );
end

#= TODO is needed?
function Broadcast.broadcasted( objf :: AbstractObjective, X :: RVecArr )
    inc_evals!( objf, length(X) );
    return eval_objf.( objf, X )
end
=#