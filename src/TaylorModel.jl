# # Taylor Polynomial Models 
#
# We provide vector valued polynomial Taylor models of degree 1 or 2.
# They implement the `SurrogateModel` interface.

# We allow the user to either provide gradient and hessian callback handles 
# or to request finite difference approximations.
# For using callbacks, we have `TaylorConfigCallbacks`. \
# There are two ways to use finite differences. The old (not recommended way) is to 
# use `TaylorConfigFiniteDiff`. This uses `FiniteDiff.jl` and could potentially
# require more evaluations. \
# To make use of the new 2-phase construction procedure, use `TaylorConfig` and 
# set the fields `gradients` and `hessians` to an `RFD.FiniteDiffStamp`.
# If they use the same stamp (default: `RFD.CFDStamp(1,3) :: CFDStamp{3,Float64}`), 
# it should be the most efficient, because we get the gradients for free from computing the hessians.

include("RecursiveFiniteDifferences.jl")

using .RecursiveFiniteDifferences
const RFD = RecursiveFiniteDifferences

# Because of all these possibilities, we actually have several 
# (sub-)implementiations of `SurrogateConfig` for Taylor Models:
abstract type TaylorCFG <: SurrogateConfig end 

@with_kw struct TaylorConfigCallbacks{
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

@with_kw struct TaylorConfigFiniteDiff <: TaylorCFG
    degree :: Int64 = 1

    max_evals :: Int64 = typemax(Int64)
    
    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
end

@with_kw struct TaylorConfig{
        S1 <: RFD.FiniteDiffStamp,
        S2 <: RFD.FiniteDiffStamp
    } <: TaylorCFG 
    
    degree :: Int64 = 1

    gradients :: S1 = RFD.CFDStamp(1,3)
    hessians :: S2 = gradients

    max_evals :: Int64 = typemax(Int64)
    
    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models." 
end

# The actual model is defined only by the gradient vectors at `x₀` and maybe Hessians.
struct TaylorModel{
    XT <: AbstractVector{<:Real}, FXT <: AbstractVector{<:Real}, 
    G <: AbstractVector{<:AbstractVector{<:Real}}, 
    H <: AbstractVector{<:AbstractMatrix{<:Real}},
    T <: Union{Nothing,TransformerFn}
    } <: SurrogateModel
    
    # expansion point and value 
    x0 :: XT
    fx0 :: FXT
    
    # gradient(s) at x0
    g :: G
    H :: H
end

# There are two types of meta. The legacy meta type stores callbacks for gradients and hessians.
# If a list of functions is provided, the file `src/diff_wrappers.jl` provides the same methods 
# as for `DiffFn`s. The legacy meta does not exploit the 2-phase construction process.
# The new meta type only stores database indices of sites used for a finite diff approximation in the actual
# construction call.

struct TaylorMetaCallbacks{GW, HW}
    gw :: GW
    hw :: HW
end

# If actual callback handles are provided, we construct the wrappers here, similar to how its done for 
# `ExactModel`s:
function init_meta( cfg :: TaylorConfigCallbacks, tfn )
    gw = FiniteDiffWrapper( tfn, cfg.gradients, cfg.jacobian )
    hw = isa( cfg.hessians, AbstractVector{<:Function} ) ? 
        HessWrapper(tfn, cfg.hessians ) : HessFromGrads( gw );
    return TaylorMetaCallbacks( gw, hw )
end

# If no callbacks are provided:
function init_meta( cfg :: TaylorConfigFiniteDiff, tfn )
    gw = FiniteDiffWrapper( tfn, cfg.gradients, cfg.jacobian )
    hw = HessFromGrads(gw)
    return TaylorMetaCallbacks( gw, hw )
end

# The other type of meta data is filled in the `prepare_XXX` methods:
@with_kw struct TaylorIndexMeta 
    grad_eval_indices :: Vector{Int} = Int[]
    grad_setter_indices :: Vector{Int} = Int[]
    hess_eval_indices :: Vector{Int} = Int[]
    hess_setter_indices :: Vector{Int} = Int[]
end

function init_meta( :: TaylorConfig )
    return TaylorIndexMeta()
end

#=
struct TaylorMeta <: SurrogateMeta end   # no construction meta data needed

max_evals( cfg :: TaylorConfig ) = cfg.max_evals;

fully_linear( tm :: TaylorModel ) = true;
combinable( :: TaylorConfig ) = false;      # TODO think about this 

# Same method as for ExactModel; duplicated for tidyness...
"Modify/initialize thec exact model `mod` so that we can differentiate it later."
function set_gradients!( mod :: TaylorModel, objf :: AbstractObjective, mop :: AbstractMOP ) :: Nothing
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
        mod.diff_fn = GradWrapper(mop, cfg.gradients, cfg.jacobian )
    end
    nothing
end

function set_hessians!( mod :: TaylorModel, objf :: AbstractObjective, mop :: AbstractMOP) :: Nothing
    cfg = model_cfg(objf);
    if isa( cfg.hessians, Symbol )
        if isa( mod.diff_fn, GradWrapper )
            mod.hess_fn = HessFromGrads( mod.diff_fn, cfg.hessians );
        else 
            if cfg.hessians == :autodiff
                if isa( mod.diff_fn, AutoDiffWrapper )
                    mod.hess_fn = mod.diff_fn 
                else
                    mod.hess_fn = AutoDiffWrapper(objf);
                end
            elseif cfg.hessians == :fdm 
                if isa( mod.diff_fn, FiniteDiffWrapper)
                    mod.hess_fn = mod.diff_fn
                else
                    mod.hess_fn = FiniteDiffWrapper( objf );
                end
            end
        end
    else
        if isa(cfg.hessians, Vector)
            @assert length(cfg.hessians) == num_outputs(objf) "Provide as many hessian functions as the objective has outputs."
        elseif isa(cfg.hessians, Function)
            @assert num_outputs(objf) == 1 "Only one hessian function provided for $(num_outputs(objf)) outputs."
        end
        mod.hess_fn = HessWrapper(mop, cfg.hessians )
    end
    nothing
end

@doc "Return a TaylorModel build from a VectorObjectiveFunction `objf`."
function _init_model( cfg ::TaylorConfig, objf :: AbstractObjective, 
    mop :: AbstractMOP, id :: AbstractIterData, ac :: AbstractConfig ) :: Tuple{TaylorModel, TaylorMeta}
    tm0 = TaylorModel(; mop = mop, objf = objf );
    set_gradients!( tm0, objf, mop );
    if cfg.degree >= 2
        set_hessians!( tm0, objf, mop );
    end
    tmeta0 = TaylorMeta()
    return update_model( tm0, objf, tmeta0, mop, id, ac);    
end

function update_model( tm :: TaylorModel, objf :: AbstractObjective, tmeta :: TaylorMeta,
    mop :: AbstractMOP, id :: AbstractIterData, :: AbstractConfig; ensure_fully_linear :: Bool = false ) :: Tuple{TaylorModel,TaylorMeta}
    @info "Building Taylor model(s)."
    tm.x0 = xᵗ(id);
    tm.fx0 = fxᵗ(id)[output_indices(objf,mop)];
    
    # set gradients
    empty!(tm.g)
    for ℓ = 1 : num_outputs(objf)
        push!(tm.g, get_gradient(tm.diff_fn, tm.x0, ℓ))
    end
    
    # and hessians if needed
    if !isnothing(tm.hess_fn)
        empty!(tm.H)
        for ℓ = 1 : num_outputs(objf)
            hess_mat = Matrix(get_hessian(tm.hess_fn, tm.x0, ℓ));
            push!(tm.H, hess_mat);
        end
    end
    @info "Done building Taylor model(s)."
    return tm, tmeta
end

function improve_model(tm::TaylorModel, ::AbstractObjective, tmeta :: TaylorMeta,
    ::AbstractMOP, id :: AbstractIterData, :: AbstractConfig;
    ensure_fully_linear :: Bool = false ) :: Tuple{TaylorModel, TaylorMeta}
    tm, tmeta 
end

function _eval_models( tm :: TaylorModel, h :: RVec, ℓ :: Int ) :: Real
    ret_val = tm.fx0[ℓ] + tm.g[ℓ]'h
    if !isempty(tm.H)
        ret_val += .5 * h'tm.H[ℓ]*h 
    end
    return ret_val
end

function eval_models( tm :: TaylorModel, x̂ :: RVec, ℓ :: Int ) :: Real
    h = x̂ .- tm.x0;
    return _eval_models( tm, h, ℓ);
 end

function eval_models( tm :: TaylorModel, x̂ :: RVec ) :: RVec
    h = x̂ .- tm.x0;
    return vcat( [_eval_models(tm, h, ℓ) for ℓ=1:num_outputs(tm.objf)]... )
end

function get_gradient( tm :: TaylorModel, x̂ :: RVec, ℓ :: Int) :: RVec
    if isempty(tm.H)
        return tm.g[ℓ]
    else
        h = x̂ .- tm.x0;
        return tm.g[ℓ] .+ .5 * ( tm.H[ℓ]' + tm.H[ℓ] ) * h
    end
end

function get_jacobian( tm :: TaylorModel, x̂ :: RVec )
    grad_list = [get_gradient(tm, x̂, ℓ) for ℓ=1:num_outputs( tm.objf ) ]
    return transpose( hcat( grad_list... ) )
end

=#