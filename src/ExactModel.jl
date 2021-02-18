# This file defines the required data structures and methods for Exact Models.

@with_kw mutable struct ExactModel <: SurrogateModel
    # reference to mop to have unscaling availabe;
    mop :: AbstractMOP

    # reference to objective(s) to evaluate 
    objf :: AbstractObjective

    diff_fn :: Union{DiffFn,Nothing} = nothing
end

@with_kw mutable struct ExactConfig <: SurrogateConfig
    gradients :: Union{Symbol, Nothing, Vector{<:Function}, Function } = :autodiff

    # alternative keyword, usage discouraged...
    jacobian :: Union{Symbol, Nothing, F where F<:Function} = nothing

    max_evals :: Int64 = typemax(Int64)

    @assert !( isnothing(gradients) && isnothing(jacobian) )
    @assert !( ( isa(gradients, Vector) && isempty( gradients ) ) && isnothing(jacobian) )
end

struct ExactMeta <: SurrogateMeta end   # no construction meta data needed

max_evals( emc :: ExactConfig ) = emc.max_evals
function max_evals!( emc :: ExactConfig, N :: Int )
    emc.max_evals = N 
    nothing 
end

# saveable( em :: ExactMeta ) = ExactMeta();

fully_linear( em :: ExactModel ) = true;

combinable( :: ExactConfig ) = false;

function set_gradients!( mod :: ExactModel, objf :: AbstractObjective )
    cfg = model_cfg(objf);
    if isa( cfg.gradients, Symbol )
        if cfg.gradients == :autodiff
            mod.diff_fn = AutoDiffWrapper( objf )
        elseif cfg.gradients == :fdm 
            mod.diff_fn = FiniteDiffWrapper( obj );
        end
    else 
        mod.diff_fn = GradWrapper( cfg.gradients, cfg.jacobian )
    end
    nothing
end


@doc "Return an ExactModel build from a VectorObjectiveFunction `objf`. 
Model is the same inside and outside of criticality round."
function init_model( objf :: AbstractObjective, mop :: AbstractMOP, ac :: Any )
    em = ExactModel(; mop = mop, objf = objf );
    set_gradients!( em, objf );
    return em, ExactMeta();
end

function update_model( em :: ExactModel, ::AbstractMOP, ac :: Any, crit_flag :: Bool = true)
    return em, ExactMeta();
end

@doc "Evaluate the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: RVec )
    return eval_objf( em.objf, unscale( em.mop, x̂ ) )
end

@doc "Evaluate output `ℓ` of the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: RVec, ℓ :: Int64)
    return eval_models(em,x̂)[ℓ]
end

@doc "Gradient vector of output `ℓ` of `em` at scaled site `x̂`."
function get_gradient( em :: ExactModel, x̂ :: RVec, ℓ :: Int64)
    return get_gradient( em.diff_fn, unscale(em.mop, x̂), ℓ)
end

@doc "Jacobian Matrix of ExactModel `em` at scaled site `x̂`."
function get_jacobian( em :: ExactModel, x̂ :: RVec )
    return get_jacobian( em.diff_fn, unscale(em.mop, x̂) )
end

#=
function set_hessians!( mod :: TaylorModel, objf :: AbstractObjective )
    cfg = model_cfg(objf);
    if isa( cfg.hessians, Symbol )
        if cfg.hessians == :autodiff
            mod.hess_fn = AutoDiffWrapper( objf )
        elseif cfg.hessians == :fdm 
            mod.hess_fn = FiniteDiffWrapper( objf );
        end
    else 
        mod.hess_fn = HessWrapper( cfg.hessians );
    end
    nothing
end
=#