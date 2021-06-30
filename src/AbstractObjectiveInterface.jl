# depends on abstract type SurrogateConfig & combinable( :: SurrogateConfig )
using Lazy: @forward

Broadcast.broadcastable( objf::AbstractObjective ) = Ref( objf );

include("BatchObjectiveFunction.jl");

# MANDATORY METHODS
"A general constructor."
function _wrap_func( :: T, fn :: Function, model_cfg :: SurrogateConfig, 
    n_vars :: Int, n_out :: Int ) :: T where {T<:Type{<:AbstractObjective}}
    nothing
end

"Return a function that evaluates an objective at an **unscaled** site."
eval_handle(::AbstractObjective) :: Function = nothing

num_vars( :: AbstractObjective ) :: Int = nothing

"Number of calls to the original objective function."
num_evals( :: AbstractObjective ) :: Int = 0
"Set evaluation counter to `N`."
num_evals!( :: AbstractObjective, N :: Int) :: Nothing = nothing

num_outputs( objf :: AbstractObjective ) :: Int = nothing

model_cfg( objf :: AbstractObjective ) :: SurrogateConfig = nothing

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

#Does the model type allow for combination with models of same type?
combinable( objf :: AbstractObjective ) = combinable( model_cfg(objf) );

#Can objectives of different types be combined?
combinable( ::Type{<:AbstractObjective}, ::Type{<:AbstractObjective} ) = false;

function combinable( objf1 :: T, objf2 :: F ) where {T<:AbstractObjective, F<:AbstractObjective}
    return ( combinable( T,F ) && 
        combinable( objf1 ) && combinable( objf2 ) && 
        model_cfg( objf1 ) == model_cfg( objf2 )
    )
end

# generic combine function for abstract objectives
"Combine two objectives. Only needed if `combinable` can return true."
function combine( objf1 :: T, objf2 :: F ) where{T<:AbstractObjective, F<:AbstractObjective}
    new_fn = combine( eval_handle( objf1 ), eval_handle( objf2 ) );
    n_out = num_outputs( objf1 ) + num_outputs( objf2 );
    new_config = combine( model_cfg(objf1), model_cfg(objf2) )
    return _wrap_func( T, new_fn, new_config, num_vars(objf1), n_out );
end

################
# A wrapper around `AbstractObjective`s that 
# ensures a certain output type `T` when evaluating
struct OutTypeWrapper{T, O <: AbstractObjective} <: AbstractObjective
    objf :: O
end

function OutTypeWrapper( inner_objf :: O, T :: Type{<:Vec} = Vector{Float64}) where O<:AbstractObjective
    return OutTypeWrapper{T,O}(inner_objf)
end

# force a new outtype if input is already wrapped
function OutTypeWrapper( wrapped :: OutTypeWrapper{T,O}, F :: Type{<:Vec} = Vector{Float64}) where{T,O}
    return OutTypeWrapper{F,O}(wrapped.objf)
end

@forward OutTypeWrapper.objf num_vars, num_evals, num_evals!, num_outputs
@forward OutTypeWrapper.objf model_cfg, max_evals, inc_evals!, combinable

function eval_handle( wrapped_objf :: OutTypeWrapper{T,O} ) where {T,O}
    return x -> convert(T, eval_handle(wrapped_objf.objf(x)))
end

function eval_objf( wrapped_objf :: OutTypeWrapper{T,O}, x :: Vec ) :: T where {T,O}
    inc_evals!(wrapped_objf)
    return convert(T, eval_handle(wrapped_objf.objf)(x))
end

"A general constructor."
function _wrap_func( :: Type{<:OutTypeWrapper{T,O}}, fn :: Function, model_cfg :: SurrogateConfig, 
    n_vars :: Int, n_out :: Int ) where {T, O}
    inner = _wrap_func( O, fn, n_vars, n_out )
    return OutTypeWrapper{T, typeof(inner)}(inner)
end

function combinable( ::Type{OutTypeWrapper{T1,O1}}, :: Type{OutTypeWrapper{T2,O2}} ) where{T1,T2,O1,O2}
    return combinable(O1,O2) && promote_type(T1,T2) <: Vec
end

# generic combine function for abstract objectives
function combine( wrapped1 :: OutTypeWrapper{T1,O1}, wrapped2 :: OutTypeWrapper{T2,O2} ) where {T1,T2,O1,O2}
    new_fn = combine( eval_handle( wrapped1.objf ), eval_handle( wrapped2.objf ) );
    n_out = num_outputs( wrapped1.objf ) + num_outputs( wrapped2.objf );
    new_config = combine( model_cfg(wrapped1.objf), model_cfg(wrapped2.objf) )
    T = promote_type(T1,T2)
    return _wrap_func( T, new_fn, new_config, num_vars(wrapped1.objf), n_out );
end