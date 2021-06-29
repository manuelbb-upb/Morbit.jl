
# This File is included from the Main script `Morbit.jl`
# It does not depend on user-defined concrete types, 
# but `AbstractMOP` and `AbstractConfig` should be defined.
#
# using LinearAlgebra: norm
#
# Contains:
# * methods for stopping 
# * methods for prettier printing/logging

function _budget_okay( mop :: AbstractMOP, ac :: AbstractConfig ) :: Bool
    for objf ∈ list_of_objectives(mop)
        if num_evals(objf) >= min( max_evals(objf), max_evals(ac) ) - 1
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
    ω_tol = ω_tol_rel( ac )
    Δ_tol = Δ_tol_rel( ac )
    ret = ω <= ω_tol && all( Δ .<= Δ_tol )
    ret && @logmsg loglevel1 "Realtive criticality stopping criterion fulfilled."
    ret
end

function Δ_abs_test( Δ :: VecOrNum, ac :: AbstractConfig )
    tol = Δ_tol_abs( ac )
    ret = all( Δ .<= tol )
    ret && @logmsg loglevel1 "Absolute radius stopping criterion fulfilled."
    ret
end

function ω_abs_test( ω :: Real, ac :: AbstractConfig )
    tol = ω_tol_abs( ac )
    ret = ω .<= tol
    ret && @logmsg loglevel1 "Absolute criticality stopping criterion fulfilled."
    ret
end

function _stop_info_str( ac :: AbstractConfig, mop :: Union{AbstractMOP,Nothing} = nothing )
    ret_str = "Stopping Criteria:\n"
    if isnothing(mop)
        ret_str *= "No. of objective evaluations ≥ $(max_evals(ac)).\n"
    else
        for objf ∈ list_of_objectives(mop)
            ret_str *= "• No. of. evaluations of objective(s) $(output_indices(objf,mop)) ≥ $(min( max_evals(objf), max_evals(ac) )).\n"
        end
    end
    ret_str *= "• No. of iterations is ≥ $(max_iter(ac)).\n"
    ret_str *= @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g ⋅ ‖ fx ‖,\n", f_tol_rel(ac) )
    ret_str *= @sprintf("• ‖ x - x⁺ ‖ ≤ %g ⋅ ‖ x ‖,\n", x_tol_rel(ac) )
    ret_str *= @sprintf("• ‖ fx - fx⁺ ‖ ≤ %g,\n", f_tol_abs(ac) )
    ret_str *= @sprintf("• ‖ x - x⁺ ‖ ≤ %g,\n", x_tol_abs(ac) )
    ret_str *= @sprintf("• ω ≤ %g AND Δ ≤ %g,\n", ω_tol_rel(ac), Δ_tol_rel(ac))
    ret_str *= @sprintf("• Δ ≤ %g OR", Δ_tol_abs(ac))
    ret_str *= @sprintf(" ω ≤ %g.", ω_tol_abs(ac))
end

function _fin_info_str( iter_data :: AbstractIterData, mop :: Union{AbstractMOP, Nothing} = nothing )
    ret_x, ret_fx = get_return_values( iter_data, mop )
    return """\n
        |--------------------------------------------
        | FINISHED
        |--------------------------------------------
        | No. iterations:  $(num_iterations(iter_data)) 
    """ + (isnothing(mop) ? "" :
        "    | No. evaluations: $(num_evals(mop))" )+ 
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