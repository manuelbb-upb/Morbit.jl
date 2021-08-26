```@meta
EditURL = "<unknown>/src/TaylorModel.jl"
```

# Taylor Polynomial Models

We provide vector valued polynomial Taylor models of degree 1 or 2.
They implement the `SurrogateModel` interface.

We allow the user to either provide gradient and hessian callback handles
or to request finite difference approximations.
For using callbacks, we have `TaylorConfigCallbacks`. \
There are two ways to use finite differences. The old (not recommended way) is to
use `TaylorConfigFiniteDiff`. This uses `FiniteDiff.jl` and could potentially
require more evaluations. \
To make use of the new 2-phase construction procedure, use `TaylorConfig` and
set the fields `gradients` and `hessians` to an `RFD.FiniteDiffStamp`.
If they use the same stamp (default: `RFD.CFDStamp(1,3) :: CFDStamp{3,Float64}`),
it should be the most efficient, because we get the gradients for free from computing the hessians.

````julia
include("RecursiveFiniteDifferences.jl")

using .RecursiveFiniteDifferences
const RFD = RecursiveFiniteDifferences
````

The actual model is defined only by the gradient vectors at `x₀` and the Hessians (if applicable).

````julia
@with_kw struct TaylorModel{
    XT <: AbstractVector{<:Real}, FXT <: AbstractVector{<:Real},
    G <: AbstractVector{<:AbstractVector{<:Real}},
    HT <: Union{Nothing,AbstractVector{<:AbstractMatrix{<:Real}}},
    } <: SurrogateModel

    # expansion point and value
    x0 :: XT
    fx0 :: FXT

    # gradient(s) at x0
    g :: G
    H :: HT = nothing
end

fully_linear( :: TaylorModel ) = true
````

Note, that the derivative approximations are actually constructed for the function(s)
```math
    f_ℓ ∘ s^{-1}
```
if some internal transformation ``s`` has happened before.
If the problem is unbounded then ``s = \operatorname{id} = s^{-1}``.

## Model Construction

Because of all the possibilities offered to the user, we actually have several
(sub-)implementiations of `SurrogateConfig` for Taylor Models.

````julia
abstract type TaylorCFG <: SurrogateConfig end
````

We make sure, that all subtypes have a field `max_evals`:

````julia
max_evals( cfg :: TaylorCFG ) = cfg.max_evals
````

### Recursive Finite Difference Models

Let's start by defining the recommended way of using Taylor approximations.
The derivative information is approximated using a dynamic programming approach
and we take care to avoid unnecessary objective evaluations.

````julia
@doc """
    TaylorConfig(; degree, gradients :: RFD.CFDStamp, hessians :: RFD.CFDStamp, max_evals)

Configuration for a polynomial Taylor model using finite difference approximations of the derivatives.
By default we have `degree = 2` and `gradients == hessians == RFD.CFDStamp(1,2)`, that is,
a first order central difference scheme of accuracy order 3 is recursed to compute the Hessians
and the gradients.
In this case, the finite difference scheme is the same for both Hessians and gradients and we profit
from caching intermediate results.
"""
@with_kw struct TaylorConfig{
        S1 <: RFD.FiniteDiffStamp,
        S2 <: Union{Nothing,RFD.FiniteDiffStamp}
    } <: TaylorCFG

    degree :: Int64 = 2

    gradients :: S1 = RFD.CFDStamp(1,2)
    hessians :: S2 = gradients

    max_evals :: Int64 = typemax(Int64)

    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
end

combinable( :: TaylorConfig ) = true
````

The new meta type only stores database indices of sites used for a finite diff approximation
in the actual construction call and is filled in the `prepare_XXX` methods:

````julia
@with_kw struct TaylorIndexMeta{W1, W2} <: SurrogateMeta
    database_indices :: Vector{Int} = Int[]
    grad_setter_indices :: Vector{Int} = Int[]
    hess_setter_indices :: Vector{Int} = Int[]
    hess_wrapper :: W1 = nothing
    grad_wrapper :: W2 = nothing
end
````

The end user won't be interested in the wrappers, so we put `nothing` in there:

````julia
saveable_type( :: TaylorIndexMeta ) = TaylorIndexMeta{Nothing, Nothing}
saveable( meta :: TaylorIndexMeta ) = TaylorIndexMeta(;
    grad_setter_indices = meta.grad_setter_indices,
    hess_setter_indices = meta.hess_setter_indices
)
````

The new construction process it is a bit complicated.
We set up a recursive finite diff tree and
need this little helper:

````julia
"Return `unique_elems, indices = unique_with_indices(arr)` such that
`unique_elems[indices] == arr` (and `unique_elems == unique(arr)`)."
function unique_with_indices( x :: AbstractVector{T} ) where T
	unique_elems = T[]
	indices = Int[]
	for elem in x
		i = findfirst( e -> all( isequal.(e,elem) ), unique_elems )
		if isnothing(i)
			push!(unique_elems, elem)
			push!(indices, length(unique_elems) )
		else
			push!(indices, i)
		end
	end
	return unique_elems, indices
end
````

Now, if the polynomial degree equals 2 we construct a tree for the Hessian calculation.
In any case, we need a tree for the gradients/jacobian.
If the `RFD.FiniteDiffStamp` for the gradients is the same as for the Hessians, we can re-use the
Hessian tree for this purpose. Else, we need to construct a new one.

````julia
function _get_RFD_trees( x, fx, grad_stamp, hess_stamp = nothing, deg = 2)
    if deg >= 2
        @assert !isnothing(hess_stamp)
        # construct tree for hessian first
        hess_wrapper = RFD.DiffWrapper(; x0 = x, fx0 = fx, stamp = hess_stamp, order = 2 )
    else
        hess_wrapper = nothing
    end

    if !isnothing(hess_wrapper) && grad_stamp == hess_stamp
        grad_wrapper = hess_wrapper
    else
        grad_wrapper = RFD.DiffWrapper(; x0 = x, fx0 = fx, stamp = grad_stamp, order = 1 )
    end

    return grad_wrapper, hess_wrapper
end


function prepare_init_model(cfg :: TaylorConfig, objf :: AbstractObjective,
    mop :: AbstractMOP, iter_data ::AbstractIterData, db :: AbstractDB, algo_cfg :: AbstractConfig; kwargs...)

    return prepare_update_model( nothing, objf, TaylorIndexMeta(), mop, iter_data, db, algo_cfg; kwargs... )
end
````

The actual database preparations are delegated to the `prepare_update_model` function.

````julia
function prepare_update_model( mod :: Union{Nothing, TaylorModel}, objf, meta :: TaylorIndexMeta, mop,
    iter_data, db, algo_cfg; kwargs... )

    x = get_x( iter_data )
    fx = get_fx( iter_data )
    x_index = get_x_index( iter_data )

    cfg = model_cfg( objf )

    grad_wrapper, hess_wrapper = _get_RFD_trees( x, fx, cfg.gradients, cfg.hessians, cfg.degree )

    XT = typeof(x)
    FXT = typeof(fx)

    lb, ub = full_bounds_internal( mop )

    if cfg.degree >= 2
        RFD.substitute_leaves!(hess_wrapper)
        # We project into the scaled variable boundaries to avoid violations:
        hess_sites = [ _project_into_box(s,lb,ub) for s in RFD.collect_leave_sites( hess_wrapper ) ]
    else
        hess_sites = XT[]
    end

    # collect leave sites for gradients
    if grad_wrapper == hess_wrapper
        grad_sites = hess_sites
    else
        RFD.substitute_leaves!( grad_wrapper )
        grad_sites = [ _project_into_box(s, lb,ub) for s in RFD.collect_leave_sites( grad_wrapper ) ]
    end

    combined_sites = [ [x,]; hess_sites; grad_sites ]

    unique_new, unique_indices = unique_with_indices(combined_sites)
    # now: `combined_sites == unique_new[unique_indices]`

    num_hess_sites = length(hess_sites)
    hess_setter_indices = unique_indices[ 2 : num_hess_sites + 1]
    grad_setter_indices = unique_indices[ num_hess_sites + 2 : end ]
    # now: `hess_sites == unique_new[ hess_setter_indices ]` and
    # `grad_sites == unique_new[ grad_setter_indices ]`

    db_indices = [ [x_index,]; [ new_result!(db, ξ, FXT()) for ξ in unique_new[ 2:end ] ] ]
    # now: `unique_new == get_site.(db, db_indices)`

    # we return a new meta object in each iteration, so that the node cache is reset in between.
    return TaylorIndexMeta(;
        database_indices = db_indices,
        grad_setter_indices,
        hess_setter_indices,
        grad_wrapper,
        hess_wrapper
    )
end
````

If the meta data is set correctly, we only have to set the value vectors for the
RFD trees and then ask for the right matrices:

````julia
function _init_model( cfg :: TaylorConfig, objf :: AbstractObjective, mop :: AbstractMOP,
    iter_data :: AbstractIterData, db :: AbstractDB, algo_config :: AbstractConfig, meta :: TaylorIndexMeta; kwargs... )
    return update_model( nothing, objf, meta, mop, iter_data, db, algo_config; kwargs...)
end

function update_model( mod :: Union{Nothing, TaylorModel}, objf :: AbstractObjective, meta :: TaylorIndexMeta,
    mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, algo_config :: AbstractConfig; kwargs...)

    all_leave_vals = get_value.( db, meta.database_indices )

    if !isnothing( meta.hess_wrapper )
        hess_leave_vals = all_leave_vals[ meta.hess_setter_indices ]
        RFD.set_leave_values!( meta.hess_wrapper, hess_leave_vals )
        H = [ RFD.hessian( meta.hess_wrapper; output_index = ℓ ) for ℓ = 1 : num_outputs(objf) ]
    else
        H = nothing
    end

    # calculate gradients
    if meta.hess_wrapper != meta.grad_wrapper
        grad_leave_vals = all_leave_vals[ meta.grad_setter_indices ]
        RFD.set_leave_values!( meta.grad_wrapper, grad_leave_vals )
    end

    # if hessians have been calculated before and `grad_wrapper == hess_wrapper` we profit from caching
    J = RFD.jacobian( meta.grad_wrapper )
    g = copy.( eachrow( J ) )

    return TaylorModel(;
        x0 = get_x( iter_data ),
        fx0 = get_fx( iter_data ),
        g, H
    ), meta
end
````

### Callback Models with Derivatives, AD or Adaptive Finite Differencing

The old way of defining Taylor Models was to provide an objective callback function
and either give callbacks for the derivatives too or ask for automatic differencing.
This is very similar to the `ExactModel`s, with the notable difference that the
gradient and Hessian information is only used to construct models
``m_ℓ = f_0 + \mathbf g^T \mathbf h + \mathbf h^T \mathbf H \mathbf h`` **once** per iteration
and then use these ``m_ℓ`` for all subsequent model evaluations/differentiation.

````julia
"""
    TaylorCallbackConfig(;degree=1,gradients,hessians=nothing,max_evals=typemax(Int64))

Configuration for a linear or quadratic Taylor model where there are callbacks provided for the
gradients and -- if applicable -- the Hessians.
The `gradients` keyword point to an array of callbacks where each callback evaluates
the gradient of one of the outputs.
"""
@with_kw struct TaylorCallbackConfig{
        G <:Union{Nothing,AbstractVector{<:Function}},
        J <:Union{Nothing,Function},
        H <:Union{Nothing,AbstractVector{<:Function}},
    } <: TaylorCFG

    degree :: Int64 = 1
    gradients :: G
    jacobian :: J = nothing
    hessians :: H = nothing

    max_evals :: Int64 = typemax(Int64)

    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
    @assert !(isnothing(gradients) && isnothing(jacobian)) "Provide either `gradients` or `jacobian`."
    @assert isa( gradients, AbstractVector ) && !isempty( gradients ) || !isnothing(jacobian) "Provide either `gradients` or `jacobian`."
    @assert !(isnothing(gradients) || isnothing(hessians)) || length(gradients) == length(hessians) "Provide same number of gradients and hessians."
end

"""
    TaylorApproximateConfig(;degree=1,mode=:fdm,max_evals=typemax(Int64))

Configure a linear or quadratic Taylor model where the gradients and Hessians are constructed
either by finite differencing (`mode = :fdm`) or automatic differencing (`mode = :autodiff`).
"""
@with_kw struct TaylorApproximateConfig <: TaylorCFG
    degree :: Int64 = 1

    mode :: Symbol = :fdm

    max_evals :: Int64 = typemax(Int64)

    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
    @assert mode in [:fdm, :autodiff] "Use `mode = :fdm` or `mode = :autodiff`."
end
````

For these models, it is not advisable to combine objectives:

````julia
combinable( :: Union{TaylorCallbackConfig, TaylorApproximateConfig}) = false
````

In both cases we transfer the finalized callbacks to the same meta structs.
In fact, `GW` and `HW` are `DiffFn`s as defined in `src/diff_wrappers.jl` (or `nothing`):

````julia
struct TaylorMetaCallbacks{GW, HW} <: SurrogateMeta
    gw :: GW
    hw :: HW
end
````

Again, we have a `tfn::TransformerFn` that represents the (un)scaling. \
If callbacks are provided, then we use the `GradWrapper` and the `HessWrapper`.

````julia
function init_meta( cfg :: TaylorCallbackConfig, objf, tfn )
    gw = GradWrapper( tfn, cfg.gradients, cfg.jacobian )
    hw = cfg.degree == 2 ? isa( cfg.hessians, AbstractVector{<:Function} ) ?
        HessWrapper(tfn, cfg.hessians ) : nothing : nothing;
    return TaylorMetaCallbacks( gw, hw )
end
````

If no callbacks are provided, we inspect the `mode` and use the corresponding wrappers:

````julia
function init_meta( cfg :: TaylorApproximateConfig, objf, tfn )
    if cfg.mode == :fdm
        gw = FiniteDiffWrapper(objf, tfn, nothing)
    else
        gw = AutoDiffWrapper( objf, tfn, nothing )
    end
    hw = cfg.degree == 2 ? HessFromGrads(gw) : nothing
    return TaylorMetaCallbacks( gw, hw )
end
````

The initialization for the legacy config types is straightforward as they don't use
the new 2-phase process:

````julia
function prepare_init_model(cfg :: Union{TaylorCallbackConfig, TaylorApproximateConfig}, objf :: AbstractObjective,
    mop :: AbstractMOP, ::AbstractIterData, ::AbstractDB, :: AbstractConfig; kwargs...)
    tfn = TransformerFn(mop)
    return init_meta( cfg, objf, tfn )
end
````

The model construction happens in the `update_model` method and makes use of the `get_gradient` and `get_hessian`
methods for the wrappers stored in `meta`:

````julia
function _init_model(cfg :: Union{TaylorCallbackConfig, TaylorApproximateConfig}, objf :: AbstractObjective,
    mop :: AbstractMOP, iter_data ::AbstractIterData, db ::AbstractDB, algo_config :: AbstractConfig,
    meta :: TaylorMetaCallbacks; kwargs...)
    return update_model(nothing, objf, meta, mop, iter_data, db, algo_config; kwargs...)
end

function update_model( ::Union{Nothing,TaylorModel}, objf :: AbstractObjective, meta :: TaylorMetaCallbacks,
    mop :: AbstractMOP, iter_data :: AbstractIterData, db :: AbstractDB, algo_config :: AbstractConfig; kwargs...)

    x = get_x(iter_data)
    fx = get_fx( iter_data )

    num_out = num_outputs( objf )
    g = [ get_gradient(meta.gw , x , ℓ ) for ℓ = 1 : num_out ]

    if !isnothing(meta.hw)
        H = [ get_hessian(meta.hw, x, ℓ) for ℓ = 1 : num_out ]
    else
        H = nothing
    end

    return TaylorModel(; x0 = x, fx0 = fx, g, H ), meta
end
````

## Model Evaluation

The evaluation of a Taylor model of form

```math
    m_ℓ(\mathbf x) = f_ℓ(\mathbf x_0) +
    \mathbf g^T ( \mathbf x - \mathbf x_0 ) + ( \mathbf x - \mathbf x_0 )^T \mathbf H_ℓ ( \mathbf x - \mathbf x_0)
```
is straightforward:

````julia
"Evaluate (internal) output `ℓ` of TaylorModel `tm`, provided a difference vector `h = x - x0`."
function _eval_models( tm :: TaylorModel, h :: Vec, ℓ :: Int )
    ret_val = tm.fx0[ℓ] + tm.g[ℓ]'h
    if !isnothing(tm.H)
        ret_val += .5 * h'tm.H[ℓ]*h
    end
    return ret_val
end

"Evaluate (internal) output `ℓ` of `tm` at scaled site `x̂`."
function eval_models( tm :: TaylorModel, x̂ :: Vec, ℓ :: Int )
    h = x̂ .- tm.x0
    return _eval_models( tm, h, ℓ)
 end
````

For the vector valued model, we iterate over all (internal) outputs:

````julia
function eval_models( tm :: TaylorModel, x̂ :: Vec )
    h = x̂ .- tm.x0
    return [ _eval_models(tm, h, ℓ) for ℓ = eachindex(tm.g)]
end
````

The gradient of ``m_ℓ`` can easily be determined:

````julia
function get_gradient( tm :: TaylorModel, x̂ :: Vec, ℓ :: Int)
    if isnothing(tm.H)
        return tm.g[ℓ]
    else
        h = x̂ .- tm.x0
        return tm.g[ℓ] .+ .5 * ( tm.H[ℓ]' + tm.H[ℓ] ) * h
    end
end
````

And for the Jacobian, we again iterate:

````julia
function get_jacobian( tm :: TaylorModel, x̂ :: Vec )
    grad_list = [ get_gradient(tm, x̂, ℓ) for ℓ=eachindex( tm.g ) ]
    return transpose( hcat( grad_list... ) )
end
````

## Summary & Quick Examples

1. The recommended way to use Finite Difference Taylor models is to define them
   with TaylorConfig, i.e.,
   ```julia
   add_objective!(mop, f, TaylorConfig)
   ```
2. To use `FiniteDiff.jl` instead, do
   ```julia
   add_objective!(mop, f, TaylorApproximateConfig(; mode = :fdm))
   ```
3. Have callbacks for the gradients and the Hessians? Great!
   ```julia
   add_objective!(mop, f, TaylorCallbackConfig(; degree = 1, gradients = [g1,g2]))
   ```
4. No callbacks, but you want the correct matrices anyways? `ForwardDiff` to the rescue:
   ```julia
   add_objective!(mop, f, TaylorApproximateConfig(; degree = 2, mode = :autodiff)
   ```

### Complete usage example
```julia
using Morbit
Morbit.print_all_logs()
mop = MixedMOP(3)

add_objective!( mop, x -> sum( ( x .- 1 ).^2 ), Morbit.TaylorApproximateConfig(;degree=2,mode=:fdm) )
add_objective!( mop, x -> sum( ( x .+ 1 ).^2 ), Morbit.TaylorApproximateConfig(;degree=2,mode=:autodiff) )

x_fin, f_fin, _ = optimize( mop, [-π, ℯ, 0])
```

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

