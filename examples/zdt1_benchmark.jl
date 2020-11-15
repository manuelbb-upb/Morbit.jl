using Morbit
using HaltonSequences
using Plots, StatsPlots
#using LaTeXStrings

using JLD2, Dates
#%%

NVAR_ARRAY = [ 2, 3, 5, 8, 13 ]
NUM_HALTON = 10;
ZDT = 3;
KERNEL = :cubic;

function run_and_store_results( opt_template, mop_template, x0, func, cfg;
        num_eval_array, true_ω_array )
    opt = deepcopy(opt_template);
    mop = deepcopy(mop_template);
    add_objective!(mop, func, cfg)
    _,_, true_ω = optimize!(opt, mop, x0 )
    push!(num_eval_array, opt.problem.vector_of_objectives[2].n_evals);
    push!(true_ω_array, true_ω);
    return nothing
end

results = Dict{Int, Vector{Vector{Int64}}}();
ω_dict = Dict{Int, Vector{Vector{Float64}}}();
for n_vars ∈ NVAR_ARRAY
    these_results = Vector{Vector{Int64}}();
    these_omegas = Vector{Vector{Float64}}();
    for x_0 ∈ HaltonPoint(n_vars)[ 1 : NUM_HALTON ]
        evals = Int64[];
        true_omegas = Float64[];

        # setup the MOP properties
        lb = zeros(n_vars)
        ub = ones(n_vars);

        if ZDT == 1
            global h(f1, g) = 1.0 - sqrt(f1 / g);
            global g(x) = 1 + 9 / (n_vars - 1) * sum(x[2:end]);
        else
        # zdt 3            
            global h(f1, g) = 1.0 - sqrt(f1 / g) - (f1 / g) * sin(10 * pi * f1);
            global g(x) = 1 + 9 / (n_vars - 1) * sum(x[2:end]);
        end

        f1(x) = x[1];
        f2(x) = g(x) * h(f1(x), g(x));

        x_0 = rand(n_vars);

        OPT_settings = AlgoConfig(
            max_iter = n_vars * 30,
            ε_crit = 1e-10,
            ν_accept = 1e-8,
            ν_success = 0.25,
            Δ₀ = 0.1,
            Δ_max = .4,
            descent_method = :steepest,
            Δ_critical = 0.0,
            true_ω_stop = 25e-3,
        )
        
        MOP = MixedMOP( lb = lb, ub = ub )
        add_objective!(MOP, f1, :cheap)   
         
        # model f2 with RBF, small θ_pivot
        f2_conf = RbfConfig(
            kernel = KERNEL,
            θ_enlarge_1 = 2.0,
            θ_pivot = 1/4,
            sampling_algorithm = :orthogonal
        )
        run_and_store_results( OPT_settings, MOP, x_0, f2, f2_conf ; 
            num_eval_array = evals, true_ω_array = true_omegas );

        #=
        # model f2 with RBF, largest θ_pivot
        rbf_conf_big_theta = RbfConfig(
            kernel = :cubic,
            θ_enlarge_1 = 2.0,
            θ_pivot = 1/2,
            sampling_algorithm = :orthogonal
        )
        i += 1;
        evals[i] += num_evals( OPT_settings, MOP, x_0, f2, rbf_conf_big_theta)
        =#

        # model f2 with linear polynomial
        f2_conf = LagrangeConfig(
            degree = 1,
            Λ = 10,
        )
        run_and_store_results( OPT_settings, MOP, x_0, f2, f2_conf ; 
        num_eval_array = evals, true_ω_array = true_omegas );

        # model f2 with quadratic polynomial
        lagrange_conf_quad10 = LagrangeConfig(
            degree = 2,
            Λ = 100,
        )
        run_and_store_results( OPT_settings, MOP, x_0, f2, f2_conf ; 
            num_eval_array = evals, true_ω_array = true_omegas );
        
        #=
        # model f2 with quadratic polynomial
        lagrange_conf_quad500 = LagrangeConfig(
            degree = 2,
            Λ = 500,
        )
        i += 1;
        evals[i]  += num_evals( OPT_settings, MOP, x_0, f2, lagrange_conf_quad500);
        =#

        # model f2 with quadratic polynomial
        f2_conf = TaylorConfig(
            degree = 1,
            gradients = :fdm,
        )
        run_and_store_results( OPT_settings, MOP, x_0, f2, f2_conf ; 
            num_eval_array = evals, true_ω_array = true_omegas );

        # model f2 with quadratic polynomial
        f2_conf = TaylorConfig(
            degree = 2,
            gradients = :fdm,
        )
        run_and_store_results( OPT_settings, MOP, x_0, f2, f2_conf ; 
            num_eval_array = evals, true_ω_array = true_omegas );
        
        push!(these_results, evals);
        push!(these_omegas, true_omegas);
    end
    
    results[n_vars] = these_results;
    ω_dict[n_vars] = these_omegas;
end

#%%
plotlyjs() 
Plots.scalefontsizes();
Plots.scalefontsizes(1.5);

dims = sort(collect(keys(results)))
#labels = ["RBF" "RBF_2" "LP1" "LP2" "LP2_2" "TP1" "TP2"]
labels = ["RBF" "LP1" "LP2_2" "TP1" "TP2"];

averages = hcat( [ sum( results[d] )/length(results[d]) for d ∈ dims ]...)';
p1 = plot( dims, averages; xticks = dims, label = labels,
    xlabel = "n_vars", ylabel = "evals", title = "Avg. Expensive Evaluations ($(length(results[dims[1]])) runs).",
    size = (960, 700), linewidth = 2) 


#Plots.scalefontsizes();

p2 = boxplot( repeat( 1:length(results[ dims[end] ][1]); outer=length(results[ dims[end] ])), 
        vcat( results[ dims[end] ]... ) ; legend=:none,
        xticks = ( 1:length(results[ dims[end] ][1]), labels[:]), 
        size = (960, 700), title = "Expensive Evaluations for $(dims[end]) Variables.",
        ) 

#%%

date_suffix = Dates.format(now(), "d_u_Y__H:M:S");

basedir = joinpath( ENV["HOME"], "zdt_benchmarks" )
if !isdir(basedir)
    mkdir(basedir)
end

this_dir = joinpath( basedir, date_suffix )
if !isdir(this_dir)
    mkdir(this_dir)
end

println( "Saving results." )
savefig( p1, joinpath( this_dir, "P1.png") )
savefig( p2, joinpath( this_dir, "P2.png" ))
savefig( p1, joinpath( this_dir, "P1.html") )
savefig( p2, joinpath( this_dir, "P2.html"))

fn = joinpath( this_dir, "ZDT$(ZDT)_$(KERNEL)_results.jld2" )
@save fn results ω_dict;