# depends on abstract type SurrogateConfig & combinable( :: SurrogateConfig )

Broadcast.broadcastable( objf::AbstractObjective ) = Ref( objf );

include("BatchObjectiveFunction.jl");

# MANDATORY METHODS
"A general constructor."
function _wrap_func( :: T, fn :: Function, model_cfg :: SurrogateConfig, 
    n_vars :: Int, n_out :: Int ) :: T where {T<:Type{<:AbstractObjective}}
    nothing
end

"Return a function that evaluates an objective at an **unscaled** site."
eval_handle(::AbstractObjective) :: Function = nothing;

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

"Evaluate the objective at unscaled site `x`. and increase counter."
function eval_objf(objf :: AbstractObjective, x :: Vec )
    inc_evals!(objf);
    eval_handle(objf)(x)
end

#=
function Broadcast.broadcasted( ::typeof(eval_objf), objf :: AbstractObjective, X̂ :: VecVec)
    inc_evals!(objf, length(X̂))
    eval_handle(objf).(X̂)
end
=#

"(Soft) upper bound on the number of function calls. "
max_evals( objf :: AbstractObjective) = max_evals( model_cfg(objf) );

"Increase evaluation count by `N`"
function inc_evals!( objf :: AbstractObjective, N :: Int = 1 )
    num_evals!( objf, num_evals(objf) + N )
end

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
