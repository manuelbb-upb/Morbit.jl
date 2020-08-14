using Morbit
function get_model_from_index( opt::AlgoConfig, ind :: Int, shape_parameter = 1.0 )
    id = opt.iter_data
    site_indices = vcat(
        id.iterate_indices[ind],
        id.model_info_array[ind].round1_indices,
        id.model_info_array[ind].round2_indices,
        id.model_info_array[ind].round3_indices,
    )
    t_s = id.sites_db[site_indices]
    t_v = [val[1:opt.n_exp] for val in id.values_db[site_indices]]

    rbf = Morbit.RBFModel( training_sites = t_s, training_values = t_v, kernel = opt.rbf_kernel, shape_parameter = shape_parameter )
    Morbit.train!(rbf)
    return rbf
end

using JuMP
using OSQP

function get_problem( opt::AlgoConfig, ind :: Int, m :: Morbit.RBFModel )
    n_vars = opt.n_vars
    id = opt.iter_data
    x = id.sites_db[ id.iterate_indices[ind] ]

    ∇f = Morbit.eval_jacobian(opt.problem, m, x)
    prob = JuMP.Model(OSQP.Optimizer);
    #JuMP.set_silent(prob)
    @variable(prob, d[1:n_vars] )

    @variable(prob, α )

    @objective(prob, Min, α)
    @constraint(prob, ∇con, ∇f*d .<= α)
    @constraint(prob, unit_con, -1.0 .<= d .<= 1.0);
    if opt.problem.is_constrained
        @constraint(prob, global_const, 0.0 .<=  x .+ Δ .* d .<= 1 )
    end
    return prob
end
