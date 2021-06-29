# This file defines the required data structures and methods for Exact Models.

struct ExactModel{
        M <: AbstractMOP,
        O <: AbstractObjective,
        D <: DiffFn 
    } <: SurrogateModel
    # reference to mop to have unscaling availabe;
    mop :: M

    # reference to objective(s) to evaluate 
    objf :: O

    diff_fn :: Union{D,Nothing}
end

struct ExactConfig{
        G <: Union{Symbol, Nothing, AbstractVector{<:Function} },
        J <: Union{Nothing, Function}
    } <: SurrogateConfig

    gradients :: G
    # alternative keyword, usage discouraged...
    jacobian :: J

    max_evals :: Int64 
end

# outer keyword constructors
# # default: :autodiff, nothing, typemax(Int)
function ExactConfig(; gradients :: Symbol = :autodiff, jacobian :: Union{Nothing, Function} = nothing, max_evals :: Int = typemax(Int64))
    @assert gradients in [:autodiff, :fdm] "`gradients` can take values `autodiff` and `fdm`." 
    return ExactConfig( gradients, jacobian, max_evals )
end

function ExactConfig(; gradients :: Function, jacobian :: Union{Nothing, Function} = nothing, max_evals :: Int = typemax(Int64))
    return ExactConfig([gradients,], jacobian, max_evals)
end

function ExactConfig(; gradients :: AbstractVector{<:Function}, jacobian :: Union{Nothing, Function} = nothing, max_evals ::Int = typemax(Int64))
    return ExactConfig(gradients, jacobian, max_evals)
end

function ExactConfig(; gradients :: Nothing, jacobian :: Union{Nothing, Function}, max_evals = typemax(Int64))
    @assert !(isnothing(gradients) && isnothing(jacobian)) "Provide either `gradients` or `jacobian`."
    return ExactConfig(gradients, jacobian, max_evals)
end

struct ExactMeta <: SurrogateMeta end   # no construction meta data needed

max_evals( emc :: ExactConfig ) = emc.max_evals

fully_linear( em :: ExactModel ) = true;

combinable( :: ExactConfig ) = false;

# get a wrapper for when cfg.gradients isa Symbol
function get_DiffFn( cfg :: ExactConfig{G,J}, objf :: AbstractObjective, mop :: AbstractMOP ) where{G<:Symbol,J}
    tfn = TransformerFn(mop)    # TODO pass `tfn` as argument?
    if cfg.gradients == :autodiff
        return AutoDiffWrapper( objf, tfn, nothing )
    elseif cfg.gradients == :fdm 
        return FiniteDiffWrapper( objf, tfn, nothing );
    end
end

# else get GradWrapper
function get_DiffFn( cfg :: ExactConfig{G,J}, objf :: AbstractObjective, mop :: AbstractMOP ) where{G,J}
    @assert length(cfg.gradients) == num_outputs(objf) "Provide as many gradient functions as the objective has outputs."
    tfn = TransformerFn(mop)
    return GradWrapper( tfn, cfg.gradients, cfg.jacobian )
end

@doc "Return an ExactModel build from a VectorObjectiveFunction `objf`. 
Model is the same inside and outside of criticality round."
function _init_model(cfg ::ExactConfig, objf :: AbstractObjective, 
    mop :: AbstractMOP, ::AbstractIterData , :: AbstractConfig)
    diff_fn = get_DiffFn( cfg, objf, mop )
    em = ExactModel(mop, objf, diff_fn )
    return em, ExactMeta();
end

function update_model( em :: ExactModel, :: AbstractObjective, meta ::ExactMeta,
    ::AbstractMOP, :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false ) :: Tuple{ ExactModel, ExactMeta }
    return em, meta
end

function improve_model( em :: ExactModel, :: AbstractObjective, meta ::ExactMeta,
    ::AbstractMOP, :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false ) :: Tuple{ ExactModel, ExactMeta }
    return em, meta
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