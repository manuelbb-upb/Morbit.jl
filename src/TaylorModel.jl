# # Taylor Polynomial Models 
#
# We provide vector valued polynomial Taylor models of degree 1 or 2.
# They implement the `SurrogateModel` interface.
#
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

# The actual model is defined only by the gradient vectors at `x₀` and the Hessians (if applicable).
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

# Note, that the derivative approximations are actually constructed for the function(s)
# ```math
#     f_ℓ ∘ s^{-1}
# ```
# if some internal transformation ``s`` has happened before.
# If the problem is unbounded then ``s = id = s^{-1}``.

# ## Model Construction

# Because of all the possibilities offered to the user, we actually have several 
# (sub-)implementiations of `SurrogateConfig` for Taylor Models.
abstract type TaylorCFG <: SurrogateConfig end 

# ### Recursive Finite Difference Models

# Let's start by defining the recommended way of using Taylor approximations.
# The derivative information is approximated using a dynamic programming approach 
# and we take care to avoid unnecessary objective evaluations. 

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

# The new meta type only stores database indices of sites used for a finite diff approximation 
# in the actual construction call and is filled in the `prepare_XXX` methods:
@with_kw struct TaylorIndexMeta{W1, W2} <: SurrogateMeta
    database_indices :: Vector{Int} = Int[]
    grad_setter_indices :: Vector{Int} = Int[]
    hess_setter_indices :: Vector{Int} = Int[]
    hess_wrapper :: W1 = nothing 
    grad_wrapper :: W2 = nothing
end

# The end user won't be interested in the wrappers, so we put `nothing` in there:
saveable_type( :: TaylorIndexMeta ) = TaylorIndexMeta{Nothing, Nothing}
saveable( meta :: TaylorIndexMeta ) = TaylorIndexMeta(; 
    grad_setter_indices = meta.grad_setter_indices,
    hess_setter_indices = meta.hess_setter_indices
)

# The new construction process it is a bit complicated: We set up a recursive finite diff tree and 
# need this little helper:
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

# Now, if the polynomial degree equals 2 we construct a tree for the Hessian calculation.
# In any case, we need a tree for the gradients/jacobian.
# If the `RFD.FiniteDiffStamp` for the gradients is the same as for the hessians, we can re-use the 
# hessian tree for this purpose. Else, we need to construct a new one.

function _get_RFD_trees( x, fx, grad_stamp, hess_stamp = nothing, deg = 2)
    if deg >= 2 
        @assert !isnothing(hess_stamp)
        ## construct tree for hessian first 
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

# The actual database preparations are delegated to the `prepare_update_model` function.
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
        ## We project into the scaled variable boundaries to avoid violations:
        hess_sites = [ _project_into_box(s,lb,ub) for s in RFD.collect_leave_sites( hess_wrapper ) ]
    else
        hess_sites = XT[]
    end
    
    ## collect leave sites for gradients
    if grad_wrapper == hess_wrapper 
        grad_sites = hess_sites
    else
        RFD.substitute_leaves!( grad_wrapper )
        grad_sites = [ _project_into_box(s, lb,ub) for s in RFD.collect_leave_sites( grad_wrapper ) ]
    end

    combined_sites = [ [x,]; hess_sites; grad_sites ]
  
    unique_new, unique_indices = unique_with_indices(combined_sites)
    ## now: `combined_sites == unique_new[unique_indices]` 
    
    num_hess_sites = length(hess_sites)
    hess_setter_indices = unique_indices[ 2 : num_hess_sites + 1]
    grad_setter_indices = unique_indices[ num_hess_sites + 2 : end ]
    ## now: `hess_sites == unique_new[ hess_setter_indices ]` and 
    ## `grad_sites == unique_new[ grad_setter_indices ]`

    db_indices = [ [x_index,]; [ new_result!(db, ξ, FXT()) for ξ in unique_new[ 2:end ] ] ]
    ## now: `unique_new == get_site.(db, db_indices)`

    ## we return a new meta object in each iteration, so that the node cache is reset in between.
    return TaylorIndexMeta(;
        database_indices = db_indices,
        grad_setter_indices,
        hess_setter_indices,
        grad_wrapper,
        hess_wrapper
    )
end 

# If the meta data is set correctly, we only have to set the value vectors for the 
# RFD trees and then ask for the right matrices:
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

    ## calculate gradients
    if meta.hess_wrapper != meta.grad_wrapper
        grad_leave_vals = all_leave_vals[ meta.grad_setter_indices ]
        RFD.set_leave_values!( meta.grad_wrapper, grad_leave_vals )
    end

    ## if hessians have been calculated before and `grad_wrapper == hess_wrapper` we profit from caching
    J = RFD.jacobian( meta.grad_wrapper )
    g = copy.( eachrow( J ) )
    
    return TaylorModel(;
        x0 = get_x( iter_data ),
        fx0 = get_fx( iter_data ),
        g, H
    ), meta
end

# ## Model Evaluation

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

function eval_models( tm :: TaylorModel, x̂ :: Vec )
    h = x̂ .- tm.x0
    return [ _eval_models(tm, h, ℓ) for ℓ = eachindex(tm.g)]
end

function get_gradient( tm :: TaylorModel, x̂ :: Vec, ℓ :: Int) 
    if isnothing(tm.H)
        return tm.g[ℓ]
    else
        h = x̂ .- tm.x0
        return tm.g[ℓ] .+ .5 * ( tm.H[ℓ]' + tm.H[ℓ] ) * h
    end
end

function get_jacobian( tm :: TaylorModel, x̂ :: Vec )
    grad_list = [ get_gradient(tm, x̂, ℓ) for ℓ=eachindex( tm.g ) ]
    return transpose( hcat( grad_list... ) )
end
#=

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

# There are two types of meta. The legacy meta type stores callbacks for gradients and hessians.
# If a list of functions is provided, the file `src/diff_wrappers.jl` provides the same methods 
# as for `DiffFn`s. The legacy meta does not exploit the 2-phase construction process.

struct TaylorMetaCallbacks{GW, HW}
    gw :: GW
    hw :: HW
end

# If actual callback handles are provided, we construct the wrappers here, similar to how its done for 
# `ExactModel`s:
function init_meta( cfg :: TaylorConfigCallbacks, tfn )
    gw = FiniteDiffWrapper( tfn, cfg.gradients, cfg.jacobian )
    hw = cfg.degree == 2 ? (isa( cfg.hessians, AbstractVector{<:Function} ) ? 
        HessWrapper(tfn, cfg.hessians ) : HessFromGrads( gw ) ) : nothing
    return TaylorMetaCallbacks( gw, hw )
end

# If no callbacks are provided:
function init_meta( cfg :: TaylorConfigFiniteDiff, tfn )
    gw = FiniteDiffWrapper( tfn, cfg.gradients, cfg.jacobian )
    hw = cfg.degree == 2 ? HessFromGrads(gw) : nothing
    return TaylorMetaCallbacks( gw, hw )
end

# The initialization for the legacy config types is straightforward as they don't use 
# the 2-phase process:
function prepare_init_model(cfg :: Union{TaylorConfigCallbacks, TaylorConfigFiniteDiff}, objf :: AbstractObjective, 
    mop :: AbstractMOP, ::AbstractIterData, ::AbstractDB, :: AbstractConfig; kwargs...)
    tfn = TransformerFn(mop)
    return init_meta( cfg, tfn )
end


=#

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


=#