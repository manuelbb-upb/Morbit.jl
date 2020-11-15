
@doc """true_ω( cg :: ConfigStruct, sc: SurrogateContainer)

Calculate **true** criticality value using the original objective functions.
Method as in `descent.jl` using linear program from Fliege et. al.
For Debugging and Testing only.
"""
function true_ω( cg :: AlgoConfig, sc :: SurrogateContainer)
    @unpack n_vars, iter_data = cg;
    @unpack x = iter_data;

    ∇m = get_jacobian( sc, x );

    prob = JuMP.Model(OSQP.Optimizer);
    JuMP.set_silent(prob)

    set_optimizer_attribute(prob,"eps_rel",1e-5)
    set_optimizer_attribute(prob,"polish",true)

    @variable(prob, α )     # negative of marginal problem value
    @objective(prob, Min, α)

    @variable(prob, d[1:n_vars] )   # direction vector
    @constraint(prob, descent_contraints, ∇m*d .<= α)
    @constraint(prob, norm_constraints, -1.0 .<= d .<= 1.0);
    if cg.problem.is_constrained
        @constraint(prob, box_constraints, 0.0 .<= x .+ d .<= 1.0 )
    end

    JuMP.optimize!(prob)
    #@show x .+ value.(d)
    return -value(α)
end

function create_shadow_surrogates( cg :: AlgoConfig )
    mop = cg.problem;
    shadow_mop = MixedMOP( lb = mop.lb, ub = mop.ub )
    cfg = ExactConfig( gradients = :autodiff );
    for func_tuple in mop.original_functions
        if func_tuple[2] == 1
            add_objective!( shadow_mop, func_tuple[1], cfg )            
        else
            add_vector_objective!( shadow_mop, func_tuple[1], cfg; n_out = func_tuple[2] )
        end
    end

    shadow_sc = SurrogateContainer(
        objf_list = shadow_mop.vector_of_objectives,
        n_objfs = shadow_mop.n_objfs,
    )
    
    set_output_objf_list!(shadow_sc)
    for objf in shadow_sc.objf_list
        prepare!(objf, objf.model_config, cg)
    end
    build_models!( shadow_sc, cg )

    return shadow_sc
end

function true_ω_small( ::Val{true}, cg :: AlgoConfig, shadow_sc :: SurrogateContainer)
    @info "Calculating **true** ω..."

    @show ω = true_ω( cg, shadow_sc )

    if ω < cg.true_ω_stop
        return true, ω 
    else 
        return false, ω
    end
end

true_ω_small( ::Val{false}, ::AlgoConfig, :: SurrogateContainer) = false, nothing