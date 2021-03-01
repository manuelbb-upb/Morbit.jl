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
    jacobian :: Union{Symbol, Nothing, Function} = nothing

    max_evals :: Int64 = typemax(Int64)

    @assert !( ( isa(gradients, Vector) && isempty( gradients ) ) && isnothing(jacobian) ) "Provide either `gradients` or `jacobian`."
    @assert !( isnothing(gradients) && isnothing(jacobian) ) "Provide either `gradients` or `jacobian`."
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

"Modify/initialize thec exact model `mod` so that we can differentiate it later."
function set_gradients!( mod :: ExactModel, objf :: AbstractObjective, mop :: AbstractMOP )
    cfg = model_cfg(objf);
    if isa( cfg.gradients, Symbol )
        if cfg.gradients == :autodiff
            mod.diff_fn = AutoDiffWrapper( objf )
        elseif cfg.gradients == :fdm 
            mod.diff_fn = FiniteDiffWrapper( objf );
        end
    else
        if isa(cfg.gradients, Vector)
            @assert length(cfg.gradients) == num_outputs(objf) "Provide as many gradient functions as the objective has outputs."
        elseif isa(cfg.gradients, Function)
            @assert num_outputs(objf) == 1 "Only one gradient provided for $(num_outputs(objf)) outputs."
        end
        mod.diff_fn = GradWrapper( mop, cfg.gradients, cfg.jacobian )
    end
    nothing
end

@doc "Return an ExactModel build from a VectorObjectiveFunction `objf`. 
Model is the same inside and outside of criticality round."
function _init_model( ::ExactConfig, objf :: AbstractObjective, 
    mop :: AbstractMOP, ::AbstractIterData , :: AbstractConfig)
    em = ExactModel(; mop = mop, objf = objf );
    set_gradients!( em, objf, mop );
    return em, ExactMeta();
end

function update_model( em :: ExactModel, :: AbstractObjective, meta ::ExactMeta,
    ::AbstractMOP, id :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false ) :: Tuple{ ExactModel, ExactMeta }
    return em, meta
end

function improve_model( em :: ExactModel, :: AbstractObjective, meta ::ExactMeta,
    ::AbstractMOP, id :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false ) :: Tuple{ ExactModel, ExactMeta }
    return em, meta
end

@doc "Evaluate the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: RVec )
    return eval_objf( em.objf, x̂ )
end

@doc "Evaluate output `ℓ` of the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: RVec, ℓ :: Int64)
    return eval_models(em,x̂)[ℓ]
end

@doc "Gradient vector of output `ℓ` of `em` at scaled site `x̂`."
function get_gradient( em :: ExactModel, x̂ :: RVec, ℓ :: Int64)
    return get_gradient( em.diff_fn, x̂, ℓ )
end

@doc "Jacobian Matrix of ExactModel `em` at scaled site `x̂`."
function get_jacobian( em :: ExactModel, x̂ :: RVec )
    return get_jacobian( em.diff_fn, x̂ );
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