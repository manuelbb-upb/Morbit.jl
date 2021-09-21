# This File is included from the Main script `Morbit.jl`
# It does not depend on user-defined concrete types, 
# but `AbstractMOP` and `AbstractConfig` should be defined.
#
# using LinearAlgebra: norm
#
# Contains:
# * variable scaling methods 
# * methods for stopping 
# * methods for prettier printing/logging
###################################################################

function _contains_index( m :: ModelGrouping, ind :: FunctionIndex )
    return ind in m.indices
end

function do_groupings( mop :: AbstractMOP, ac :: AbstractConfig )
    if !_combine_models_by_type(ac)
        return [ ModelGrouping( [ind,], get_cfg(_get(mop,ind)) ) for ind in get_function_indices(mop) ]
    end
    groupings = ModelGrouping[]
    for objf_ind1 in get_function_indices( mop )
        objf1 = _get( mop, objf_ind1 )

        # check if there already is a group that `objf1`
        # belongs to and set `group_index` to its position in `groupings`
        group_index = -1
        for (gi, group) in enumerate(groupings)
            if _contains_index( group, objf_ind1 )
                group_index = gi
                break
            end
        end
        # if there is no group with `objf1` in it, 
        # then create a new one and set `group_index` to new, last position
        if group_index < 0
            push!( groupings, ModelGrouping(FunctionIndex[objf_ind1,], model_cfg(objf1)) )
            group_index = length(groupings)
        end
        
        group = groupings[group_index]
        
        # now, for every remaining function index, check if we can add 
        # it to the group of `objf_ind1`
        for objf_ind2 in get_function_indices( mop )
            objf2 = _get( mop, objf_ind2 )
            if objf_ind1 != objf_ind2 && combinable( objf1, objf2 ) && !_contains_index(group, objf_ind2)
                push!( group.indices, objf_ind2 )
            end
        end
    end
    return groupings
end

function build_super_db( groupings :: Vector{<:ModelGrouping}, x_scaled :: XT, eval_res ) where XT <: VecF
    n_vars = length(x_scaled)

    sub_dbs = Dict{FunctionIndexTuple, ArrayDB}()
    x_index_mapping = Dict{FunctionIndexTuple, Int}()
    for group in groupings 
        index_tuple = Tuple(group.indices)

        _group_vals = eval_result_to_vector( eval_res, group.indices )
        group_vals = (Base.promote_eltype( _group_vals, MIN_PRECISION )).(_group_vals)

        res = Result(; x = SVector{n_vars}(x_scaled), y = MVector{length(group_vals)}(group_vals) )
        config_saveable_type = get_saveable_type( group.cfg, x_scaled, group_vals )
        
        sub_db = init_db( ArrayDB, typeof(res), config_saveable_type )
        set_transformed!(sub_db, true)
        x_index = ensure_contains_values!( sub_db, x_scaled, group_vals )
        
        sub_dbs[ index_tuple ] = sub_db
        x_index_mapping[ index_tuple ] = x_index
    end

    return sub_dbs, x_index_mapping
end
###################################################################
function ensure_precision( x :: X ) where X<:Real 
    _X = promote_type( X, MIN_PRECISION )
    return _X(x)
end

function ensure_precision( x :: AbstractVector{X} ) where X<:Real
    _X = promote_type( X, MIN_PRECISION )
    return _X.(x)
end

# Scaling
function _scale!( x, lb, ub )
    for (i,var_bounds) ∈ enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            x[i] -= var_bounds[1]
            x[i] /= ( var_bounds[2] - var_bounds[1] )
        end
    end
    nothing
end

function _scale( x, lb, ub )
    χ = copy(x);
    _scale!(χ, lb, ub);
    return χ
end

function _unscale!( x_scaled, lb, ub )
    for (i,var_bounds) ∈ enumerate(zip( lb, ub ))
        if !(isinf(var_bounds[1]) || isinf(var_bounds[2]))
            # TODO: Make the component scaling memoized?
            x_scaled[i] *= (var_bounds[2] - var_bounds[1]) 
            x_scaled[i] += var_bounds[1]
        end
    end
    nothing
end

function _unscale( x_scaled, lb, ub )
    χ̂ = copy(x_scaled)
    _unscale!(χ̂, lb, ub)
    return χ̂
end

function _project_into_box( z, lb, ub)
    return min.( max.( z, lb ), ub )
end

"Return smallest positive and biggest negative and `σ₊` and `σ₋` so that `x .+ σ± .* d` stays within bounds."
function _intersect_bounds( x, d, lb, ub )
   d_scaled = (ub .- lb ) .* d ./ norm( d, Inf )
   
   σ_pos = norm( _project_into_box( x .+ d_scaled, lb, ub ) - x, 2 )
   σ_neg = norm( _project_into_box( x .- d_scaled, lb, ub ) - x, 2 )

   return σ_pos, σ_neg
end

function intersect_bounds( x, d, lb, ub ; return_vals :: Symbol = :both )
    σ_pos, σ_neg = _intersect_bounds( x, d, lb, ub )

    if return_vals == :both 
        return σ_pos, σ_neg
    elseif return_vals == :pos
        return σ_pos 
    elseif return_vals == :neg 
        return σ_neg 
    elseif return_vals == :absmax
        if abs(σ_pos) >= abs(σ_neg)
            return σ_pos
        else
            return σ_neg 
        end 
    end
end

"Return lower and upper bound vectors combining global and trust region constraints."
function _local_bounds( x, Δ, lb, ub )
    lb_eff = max.( lb, x .- Δ );
    ub_eff = min.( ub, x .+ Δ );
    return lb_eff, ub_eff 
end


# use for finite (e.g. local) bounds only
function _rand_box_point(lb, ub, type :: Type{<:Real} = MIN_PRECISION)
    return lb .+ (ub .- lb) .* rand(type, length(lb))
end


######## Stopping

function _budget_okay( mop :: AbstractMOP, ac :: AbstractConfig ) :: Bool
    max_conf_evals = max_evals( ac )
    for objf ∈ list_of_objectives(mop)
        if num_evals(objf) >= min( max_evals(objf), max_conf_evals ) - 1
            return false;
        end
    end
    return true
end

function f_tol_rel_test( fx :: Vec, fx⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = f_tol_rel(ac)
    if isa(tol, Real)
        ret = norm( fx .- fx⁺, Inf ) <= tol * norm( fx, Inf )
    else
        ret = all( abs.( fx .- fx⁺ ) .<= tol .* fx )
    end
    ret && @logmsg loglevel1 "Relative (objective) stopping criterion fulfilled."
    ret
end

function x_tol_rel_test( x :: Vec, x⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = x_tol_rel(ac)
    if isa(tol, Real)
        ret = norm( x .- x⁺, Inf ) <= tol * norm( x, Inf )
    else
        ret = all( abs.( x .- x⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Relative (decision) stopping criterion fulfilled."
    ret
end

function f_tol_abs_test( fx :: Vec, fx⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = f_tol_abs(ac)
    if isa(tol, Real)
        ret = norm( fx .- fx⁺, Inf ) <= tol 
    else
        ret = all( abs.( fx .- fx⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Absolute (objective) stopping criterion fulfilled."
    ret
end

function x_tol_abs_test( x :: Vec, x⁺ :: Vec, ac :: AbstractConfig ) :: Bool
    tol = x_tol_abs(ac)
    if isa(tol, Real)
        ret =  norm( x .- x⁺, Inf ) <= tol 
    else
        ret = all( abs.( x .- x⁺ ) .<= tol )
    end
    ret && @logmsg loglevel1 "Absolute (decision) stopping criterion fulfilled."
    ret
end

function ω_Δ_rel_test( ω :: Real, Δ :: VecOrNum, ac :: AbstractConfig )
    ω_tol = omega_tol_rel( ac )
    Δ_tol = delta_tol_rel( ac )
    ret = ω <= ω_tol && all( Δ .<= Δ_tol )
    ret && @logmsg loglevel1 "Realtive criticality stopping criterion fulfilled."
    ret
end

function Δ_abs_test( Δ :: VecOrNum, ac :: AbstractConfig )
    tol = delta_tol_abs( ac )
    ret = all( Δ .<= tol )
    ret && @logmsg loglevel1 "Absolute radius stopping criterion fulfilled."
    ret
end

function ω_abs_test( ω :: Real, ac :: AbstractConfig )
    tol = omega_tol_abs( ac )
    ret = ω .<= tol
    ret && @logmsg loglevel1 "Absolute criticality stopping criterion fulfilled."
    ret
end

function _stop_info_str( ac :: AbstractConfig, mop :: Union{AbstractMOP,Nothing} = nothing )
    ret_str = "Stopping Criteria:\n"
    if isnothing(mop)
        ret_str *= "No. of objective evaluations ≥ $(max_evals(ac)).\n"
    else
        for f_ind ∈ get_function_indices(mop)
            func = _get(mop, f_ind)
            ret_str *= "• No. of. evaluations of $(f_ind) ≥ $(min( max_evals(func), max_evals(ac) )).\n"
        end
    end
    ret_str *= "• No. of iterations is ≥ $(max_iter(ac)).\n"
    ret_str *= @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g ⋅ ‖ fx ‖,\n", f_tol_rel(ac) )
    ret_str *= @sprintf("• ‖ x - x⁺ ‖ ≤ %g ⋅ ‖ x ‖,\n", x_tol_rel(ac) )
    ret_str *= @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g,\n", f_tol_abs(ac) )
    ret_str *= @sprintf("• ‖ x - x⁺ ‖ ≤ %g,\n", x_tol_abs(ac) )
    ret_str *= @sprintf("• ω ≤ %g AND Δ ≤ %g,\n", omega_tol_rel(ac), delta_tol_rel(ac))
    ret_str *= @sprintf("• Δ ≤ %g OR", delta_tol_abs(ac))
    ret_str *= @sprintf(" ω ≤ %g.", omega_tol_abs(ac))
end

function get_return_values(iter_data, mop )
    ret_x = unscale( get_x(iter_data), mop )
	ret_fx = get_fx( iter_data )
    return ret_x, ret_fx 
end

function _fin_info_str(data_base, iter_data :: AbstractIterData, mop, stopcode = nothing )
    ret_x, ret_fx = get_return_values( iter_data, mop )
    return """\n
        |--------------------------------------------
        | FINISHED ($stopcode)
        |--------------------------------------------
        | No. iterations:  $(get_num_iterations(iter_data)) 
    """ * (isnothing(mop) ? "" :
        "    | No. evaluations: $(num_evals(mop))" ) *
    """ 
        | final unscaled vectors:
        | iterate: $(_prettify(ret_x, 10))
        | value:   $(_prettify(ret_fx, 10))
    """
end

using Printf: @sprintf
function _prettify( vec :: Vec, len :: Int = 5) :: AbstractString
    return string(
        "[",
        join( 
            [@sprintf("%.5f",vec[i]) for i = 1 : min(len, length(vec))], 
            ", "
        ),
        length(vec) > len ? ", …" : "",
        "]"
    )
end