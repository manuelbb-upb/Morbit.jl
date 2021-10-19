# # Taylor Polynomial Models 

#src We generate a documentation page from this source file. 
#src Let's tell the user:
#md # This file is automatically generated from source code. 
#md # For usage examples refer to [Summary & Quick Examples](@ref taylor_summary).

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
    
    ## expansion point and value 
    x0 :: XT
    fx0 :: FXT
    
    ## gradient(s) at x0
    g :: G
    H :: HT = nothing
end

fully_linear( :: TaylorModel ) = true

# Note, that the derivative approximations are actually constructed for the function(s)
# ```math
#     f_ℓ ∘ s^{-1}
# ```
# if some internal transformation ``s`` has happened before.
# If the problem is unbounded then ``s = \operatorname{id} = s^{-1}``.

# ## Model Construction

# Because of all the possibilities offered to the user, we actually have several 
# (sub-)implementiations of `SurrogateConfig` for Taylor Models.
abstract type TaylorCFG <: SurrogateConfig end 

# We make sure, that all subtypes have a field `max_evals`:
max_evals( cfg :: TaylorCFG ) = cfg.max_evals

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
function get_saveable_type( :: TaylorConfig )
    return TaylorIndexMeta{Nothing, Nothing}
end
get_saveable( meta :: TaylorIndexMeta ) = TaylorIndexMeta(; 
    grad_setter_indices = meta.grad_setter_indices,
    hess_setter_indices = meta.hess_setter_indices
)

# The new construction process it is a bit complicated. 
# We set up a recursive finite diff tree and 
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
# If the `RFD.FiniteDiffStamp` for the gradients is the same as for the Hessians, we can re-use the 
# Hessian tree for this purpose. Else, we need to construct a new one.

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


function prepare_init_model(cfg :: TaylorConfig, func_indices :: FunctionIndexIterable,
    mop :: AbstractMOP, scal :: AbstractVarScaler, 
	id :: AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; 
	kwargs...)
    return prepare_update_model(nothing, TaylorIndexMeta(), cfg, func_indices, mop, scal, id, sdb, ac ; kwargs... )
end

# The actual database preparations are delegated to the `prepare_update_model` function.
function prepare_update_model( 
        mod :: Union{Nothing, TaylorModel}, meta :: TaylorIndexMeta, 
		cfg :: TaylorConfig, func_indices :: FunctionIndexIterable, mop :: AbstractMOP, scal :: AbstractVarScaler, 
		iter_data :: AbstractIterData, sdb :: AbstractSuperDB, algo_config :: AbstractConfig; kwargs... )

    db = get_sub_db( sdb, func_indices )
    x = get_x_scaled( iter_data )
    x_index = get_x_index( iter_data, func_indices )
    fx = get_value( db, x_index )

    grad_wrapper, hess_wrapper = _get_RFD_trees( x, fx, cfg.gradients, cfg.hessians, cfg.degree )
    
    XT = typeof(x)
    
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
        grad_sites = [ _project_into_box(s,lb,ub) for s in RFD.collect_leave_sites( grad_wrapper ) ]
    end

    combined_sites = [ [x,]; hess_sites; grad_sites ]
  
    unique_new, unique_indices = unique_with_indices(combined_sites)
    ## now: `combined_sites == unique_new[unique_indices]` 
    
    num_hess_sites = length(hess_sites)
    hess_setter_indices = unique_indices[ 2 : num_hess_sites + 1]
    grad_setter_indices = unique_indices[ num_hess_sites + 2 : end ]
    ## now: `hess_sites == unique_new[ hess_setter_indices ]` and 
    ## `grad_sites == unique_new[ grad_setter_indices ]`

    db_indices = [ [x_index,]; [ new_result!(db, ξ, []) for ξ in unique_new[ 2:end ] ] ]
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
function init_model(meta :: TaylorIndexMeta, cfg :: TaylorConfig, func_indices :: FunctionIndexIterable,
	mop :: AbstractMOP, scal :: AbstractVarScaler, iter_data :: AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; kwargs... )
    
    return update_model( nothing, meta, cfg, func_indices, mop, scal, iter_data, sdb, ac; kwargs...)
end

# Note, that we only perform updates if the iterate has changed, `x != mod.x0`, because 
# we don't change the differencing parameters.
function update_model( mod::Union{Nothing,TaylorModel}, meta :: TaylorIndexMeta, cfg :: TaylorConfig,
	func_indices :: FunctionIndexIterable, mop :: AbstractMOP, scal :: AbstractVarScaler, iter_data :: AbstractIterData, 
	sdb :: AbstractSuperDB, ac :: AbstractConfig; 
	kwargs... )
    
    db = get_sub_db( sdb, func_indices )
    x = get_x_scaled(iter_data)
    if isnothing(mod) || (x != mod.x0)
        all_leave_vals = get_value.( db, meta.database_indices )
            
        if !isnothing( meta.hess_wrapper )
            hess_leave_vals = all_leave_vals[ meta.hess_setter_indices ]
            RFD.set_leave_values!( meta.hess_wrapper, hess_leave_vals )
            H = [ RFD.hessian( meta.hess_wrapper; output_index = ℓ ) for ℓ = 1 : num_outputs(func_indices) ]
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
            x0 = x,
            fx0 = get_fx( iter_data ),
            g, H
        ), meta
    else
        return mod,meta
    end
end

# ### Callback Models with Derivatives, AD or Adaptive Finite Differencing 

# The old way of defining Taylor Models was to provide an objective callback function 
# and either give callbacks for the derivatives too or ask for automatic differencing.
# This is very similar to the `ExactModel`s, with the notable difference that the 
# gradient and Hessian information is only used to construct models 
# ``m_ℓ = f_0 + \mathbf g^T \mathbf h + \mathbf h^T \mathbf H \mathbf h`` **once** per iteration
# and then use these ``m_ℓ`` for all subsequent model evaluations/differentiation.

"""
    TaylorCallbackConfig(;degree=1,max_evals=typemax(Int64))

Configuration for a linear or quadratic Taylor model where there are callbacks provided for the 
gradients and -- if applicable -- the Hessians.
"""
@with_kw struct TaylorCallbackConfig <: TaylorCFG
    
    degree :: Int64 = 1

    max_evals :: Int64 = typemax(Int64)

    @assert 1 <= degree <= 2 "Can only construct linear and quadratic polynomial Taylor models."
end

needs_gradients( cfg :: TaylorCallbackConfig ) = true
needs_hessians( cfg :: TaylorCallbackConfig ) = cfg.degree >= 2

# For these models, it is not advisable to combine objectives:
combinable( :: TaylorCallbackConfig ) = false

# The meta structs are just for show:
struct TaylorCallbackMeta <: SurrogateMeta end

# The initialization for the legacy config types is straightforward as they don't use 
# the new 2-phase process:
function prepare_init_model(cfg :: TaylorCallbackConfig, func_indices :: FunctionIndexIterable,
    mop :: AbstractMOP, scal :: AbstractVarScaler, id ::AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; kwargs...)
    return TaylorCallbackMeta()
end

# The model construction happens in the `update_model` method and makes use of the `get_gradient` and `get_hessian`
# methods of the `AbstractVectorObjective`.

function init_model(meta :: TaylorCallbackMeta, cfg :: TaylorCallbackConfig, func_indices :: FunctionIndexIterable,
    mop :: AbstractMOP, scal :: AbstractVarScaler, id ::AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; kwargs...)
    return update_model(nothing, meta, cfg, func_indices, mop, scal, id, sdb, ac; kwargs... )
end

function update_model( mod :: Union{Nothing,TaylorModel}, meta :: TaylorCallbackMeta, cfg :: TaylorCallbackConfig, func_indices :: FunctionIndexIterable,
    mop :: AbstractMOP, scal :: AbstractVarScaler, id ::AbstractIterData, sdb :: AbstractSuperDB, ac :: AbstractConfig; kwargs...)

    x0 = get_x_scaled(id)
    x0_unscaled = get_x(id)

    if isnothing(mod) || (x0 != mod.x0)
        fx0 = get_vals( id, sdb, func_indices )

        J = jacobian_of_unscaling(scal)
        Jᵀ = transpose(J)

        g = collect( Iterators.flatten( 
            [ let func = _get(mop, ind), func_jac = get_objf_jacobian(func, x0_unscaled); 
                [ _ensure_vec( Jᵀ * func_jac[ℓ,:] ) for ℓ = 1 : num_outputs(func) ]
            end for ind = func_indices ] 
        ) )
       
        H = if cfg.degree >= 2
            collect( Iterators.flatten( 
                [ let func = _get(mop, ind); 
                    [ (Jᵀ * get_objf_hessian(func, x0_unscaled, ℓ) * J) for ℓ = 1 : num_outputs(func) ]
                end for ind = func_indices ] 
            ) )
        else
            nothing 
        end
        
        return TaylorModel(; x0, fx0, g, H ), meta
    else
        return mod, meta
    end

end

# ## Model Evaluation

# The evaluation of a Taylor model of form 
# 
# ```math 
#     m_ℓ(\mathbf x) = f_ℓ(\mathbf x_0) + 
#     \mathbf g^T ( \mathbf x - \mathbf x_0 ) + ( \mathbf x - \mathbf x_0 )^T \mathbf H_ℓ ( \mathbf x - \mathbf x_0)  
# ```
# is straightforward:
"Evaluate (internal) output `ℓ` of TaylorModel `tm`, provided a difference vector `h = x - x0`."
function _eval_models( tm :: TaylorModel, h :: Vec, ℓ :: Int )
    ret_val = tm.fx0[ℓ] + tm.g[ℓ]'h
    if !isnothing(tm.H)
        ret_val += .5 * h'tm.H[ℓ]*h 
    end
    return ret_val
end

"Evaluate (internal) output `ℓ` of `tm` at scaled site `x̂`."
function eval_models( tm :: TaylorModel, scal :: AbstractVarScaler, x̂ :: Vec, ℓ :: Int )
    h = x̂ .- tm.x0
    return _eval_models( tm, h, ℓ)
 end

# For the vector valued model, we iterate over all (internal) outputs:
function eval_models( tm :: TaylorModel, scal :: AbstractVarScaler, x̂ :: Vec )
    h = x̂ .- tm.x0
    return [ _eval_models(tm, h, ℓ) for ℓ = eachindex(tm.g)]
end

# The gradient of ``m_ℓ`` can easily be determined:
function get_gradient( tm :: TaylorModel, scal :: AbstractVarScaler, x̂ :: Vec, ℓ :: Int) 
    if isnothing(tm.H)
        return tm.g[ℓ]
    else
        h = x̂ .- tm.x0
        return tm.g[ℓ] .+ .5 * ( tm.H[ℓ]' + tm.H[ℓ] ) * h
    end
end

# And for the Jacobian, we again iterate:
function get_jacobian( tm :: TaylorModel, scal :: AbstractVarScaler, x̂ :: Vec )
    grad_list = [ get_gradient(tm, x̂, ℓ) for ℓ=eachindex( tm.g ) ]
    return transpose( hcat( grad_list... ) )
end

# ## [Summary & Quick Examples](@id taylor_summary)

# 1. The recommended way to use Finite Difference Taylor models is to define them 
#    with TaylorConfig, i.e.,  
#    ```julia
#    add_objective!(mop, f, TaylorConfig())
#    ```
# 2. To use `FiniteDiff.jl` instead, do 
#    ```julia
#    add_objective!(mop, f, TaylorApproximateConfig(; mode = :fdm))
#    ```
# 3. Have callbacks for the gradients and the Hessians? Great! 
#    ```julia
#    add_objective!(mop, f, TaylorCallbackConfig(; degree = 1, gradients = [g1,g2]))
#    ```
# 4. No callbacks, but you want the correct matrices anyways? `ForwardDiff` to the rescue:
#    ```julia 
#    add_objective!(mop, f, TaylorApproximateConfig(; degree = 2, mode = :autodiff)
#    ```

# ### Complete usage example 
# ```julia
# using Morbit
# Morbit.print_all_logs()
# mop = MixedMOP(3)
#
# add_objective!( mop, x -> sum( ( x .- 1 ).^2 ), Morbit.TaylorApproximateConfig(;degree=2,mode=:fdm) )
# add_objective!( mop, x -> sum( ( x .+ 1 ).^2 ), Morbit.TaylorApproximateConfig(;degree=2,mode=:autodiff) )
#
# x_fin, f_fin, _ = optimize( mop, [-π, ℯ, 0])
# ```
