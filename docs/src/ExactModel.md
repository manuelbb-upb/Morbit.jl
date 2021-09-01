```@meta
EditURL = "<unknown>/../src/ExactModel.jl"
```

Exact models

## Introduction
The `ExactModel` is used to evaluate the objective exactly, without surrogate modelling
(except for internal variable scaling).
The derivatives are either user provided callbacks or can be deterimned using `ForwardDiff`
or `FiniteDiff` automatically.

## Surrogate Model Interface Implementations

````julia
"""
    ExactModel( tfn, objf, diff_fn )

Exact Model type for evaluating the objective function `objf` directly.
Is instantiated by the corresponding `init_model` and `update_model` functions.
"""
struct ExactModel{
        M <: TransformerFn,
        O <: AbstractObjective,
        D <: DiffFn
    } <: SurrogateModel

    # reference to a `TransformerFn` to have unscaling availabe:
    tfn :: M

    # reference to objective(s) to evaluate
    objf :: O

    # a `DiffFn` providing derivative information
    diff_fn :: Union{D,Nothing}
end
````

The can determine the behavior of an `ExactModel` using `ExactConfig`:

````julia
"""
    ExactConfig(; gradients, jacobian = nothing, max_evals = typemax(Int64))

Configuration for an `ExactModel`.
`gradients` should be a vector of callbacks for the objective gradients **or**
a `Symbol`, either `:autodiff` or `fdm`, to define the differentiation method
to use on the objective.
Alternatively, a `jacobian` handle can be provided.
"""
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
````

There is no need for custom meta information:

````julia
struct ExactMeta <: SurrogateMeta end   # no construction meta data needed
````

The remaining implementations are straightforward:

````julia
max_evals( emc :: ExactConfig ) = emc.max_evals
````

We always deem the models fully linear:

````julia
fully_linear( em :: ExactModel ) = true
````

They are not combinable to have individiual gradients availabe:

````julia
combinable( :: ExactConfig ) = false
````

## Construction

When `cfg.gradients` is a `Symbol` we make use of the `DiffWrapper`s defined
in `src/diff_wrappers.jl`:

````julia
function get_DiffFn( cfg :: ExactConfig{G,J}, objf :: AbstractObjective, tfn ) where{G<:Symbol,J}
    if cfg.gradients == :autodiff
        return AutoDiffWrapper( objf, tfn, nothing )
    elseif cfg.gradients == :fdm
        return FiniteDiffWrapper( objf, tfn, nothing );
    end
end
````

Else we use a `GradWrapper`:

````julia
function get_DiffFn( cfg :: ExactConfig{G,J}, objf :: AbstractObjective, tfn) where{G,J}
    @assert length(cfg.gradients) == num_outputs(objf) "Provide as many gradient functions as the objective has outputs."
    return GradWrapper( tfn, cfg.gradients, cfg.jacobian )
end
````

All "construction" work is done in the `_init_model` function:

````julia
function prepare_init_model(cfg ::ExactConfig, objf :: AbstractObjective,
    mop :: AbstractMOP, ::AbstractIterData, ::AbstractDB, :: AbstractConfig; kwargs...)
    return ExactMeta()
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
````

All the other functions simply return the input:

````julia
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
````

## Evaluation

````julia
@doc "Evaluate the ExactModel `em` at scaled site `x̂`."
function eval_models( em :: ExactModel, x̂ :: Vec )
    return eval_objf( em.objf, em.tfn, x̂ )
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
````

## Quick Usage Example

```julia
using Morbit
Morbit.print_all_logs()
mop = MixedMOP(3)

f1 = x -> sum( ( x .- 1 ).^2
f2 = x -> sum( ( x .+ 1 ).^2
g1 = x -> 2 .* ( x .- 1 )
g2 = x -> 2 .* ( x .+ 1 )

add_objective!( mop, f1, ExactConfig(; gradients = [g1,]) )
add_objective!( mop, f2, ExactConfig(; gradients = [g2,]) )

x_fin, f_fin, _ = optimize( mop, [-π, ℯ, 0])
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

