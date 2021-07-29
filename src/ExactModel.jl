# This file defines the required data structures and methods for Exact Models.

struct ExactModel{
        M <: TransformerFn,
        O <: AbstractObjective,
        D <: DiffFn 
    } <: SurrogateModel
    # reference to mop to have unscaling availabe;
    tfn :: M

    # reference to objective(s) to evaluate 
    objf :: O

    diff_fn :: Union{D,Nothing}
end

@with_kw struct ExactConfig{
        G <: Union{Symbol, Nothing, AbstractVector{<:Function} },
        J <: Union{Nothing, Function}
    } <: SurrogateConfig

    gradients :: G = :autodiff
    # alternative keyword, usage discouraged...
    jacobian :: J = nothing

    max_evals :: Int64 = typemax(Int64)

    @assert !(isnothing(gradients) && isnothing(jacobian)) "Provide either `gradients` or `jacobian`."
    @assert !(gradients isa Symbol) || gradients in [:autodiff, :fdm ] "`gradients` must be `:autodiff` or `:fdm`"
end

struct ExactMeta <: SurrogateMeta end   # no construction meta data needed

max_evals( emc :: ExactConfig ) = emc.max_evals

fully_linear( em :: ExactModel ) = true;

combinable( :: ExactConfig ) = false;

# get a wrapper for when cfg.gradients isa Symbol
function get_DiffFn( cfg :: ExactConfig{G,J}, objf :: AbstractObjective, tfn ) where{G<:Symbol,J}
    if cfg.gradients == :autodiff
        return AutoDiffWrapper( objf, tfn, nothing )
    elseif cfg.gradients == :fdm 
        return FiniteDiffWrapper( objf, tfn, nothing );
    end
end

# else get GradWrapper
function get_DiffFn( cfg :: ExactConfig{G,J}, objf :: AbstractObjective, tfn) where{G,J}
    @assert length(cfg.gradients) == num_outputs(objf) "Provide as many gradient functions as the objective has outputs."
    return GradWrapper( tfn, cfg.gradients, cfg.jacobian )
end

@doc "Return an ExactModel build from a VectorObjectiveFunction `objf`. 
Model is the same inside and outside of criticality round."
function _init_model(cfg ::ExactConfig, objf :: AbstractObjective, 
    mop :: AbstractMOP, ::AbstractIterData, ::AbstractDB, :: AbstractConfig, emeta :: ExactMeta; kwargs...)
    tfn = TransformerFn(mop)
    diff_fn = get_DiffFn( cfg, objf, tfn )
    em = ExactModel(tfn, objf, diff_fn )
    return em, emeta
end

function update_model( em :: ExactModel, :: AbstractObjective, meta ::ExactMeta,
    ::AbstractMOP, :: AbstractIterData, ::AbstractDB, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false, kwargs... )
    return em, meta
end

function improve_model( em :: ExactModel, :: AbstractObjective, meta ::ExactMeta,
    ::AbstractMOP, :: AbstractIterData, ::AbstractDB, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false, kwargs ... )
    return em, meta
end

function prepare_init_model(cfg ::ExactConfig, objf :: AbstractObjective, 
    mop :: AbstractMOP, ::AbstractIterData, ::AbstractDB, :: AbstractConfig; kwargs...)
    return ExactMeta()
end

@doc "Evaluate the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: Vec )
    return eval_objf( em.objf, x̂ )
    # using `eval_objf` will increase the evaluation count of `em.objf`
    # That is why this count might be very high when using backtracking.
    # eval_handle( em.objf )(x̂) would not increase the count.
end

@doc "Evaluate output `ℓ` of the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: Vec, ℓ :: Int64)
    return eval_models(em,x̂)[ℓ]
end

@doc "Gradient vector of output `ℓ` of `em` at scaled site `x̂`."
function get_gradient( em :: ExactModel, x̂ :: Vec, ℓ :: Int64)
    return get_gradient( em.diff_fn, x̂, ℓ )
end

@doc "Jacobian Matrix of ExactModel `em` at scaled site `x̂`."
function get_jacobian( em :: ExactModel, x̂ :: Vec )
    return get_jacobian( em.diff_fn, x̂ );
end