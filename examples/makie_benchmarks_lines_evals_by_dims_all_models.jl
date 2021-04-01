using Pkg;
Pkg.activate( @__DIR__ );

using AbstractPlotting, CairoMakie
using AlgebraOfGraphics
using Statistics
using Colors;
using DataFrames;
using FileIO, JLD2

# disable inline plotting
#AbstractPlotting.inline!(true);

#%% load helpers 
include("plot_helpers/upb_colors.jl");
include("plot_helpers/loading_saving.jl")

#%% settings 
res_file = joinpath(
    ENV["HOME"], "MORBIT_BENCHMARKS", 
#    "results_WoJEMTIw_3600x8_29_Mar_2021__18_13_49.jld2"
#   "results_XIHJrRkp_4480x8_30_Mar_2021__02_22_26.jld2"
#   "results_QHvrOuJQ_6720x8_30_Mar_2021__19_14_54.jld2",
#"results_XwSD7zMO_960x8_31_Mar_2021__20_32_36.jld2"
"results_32sZVhKT_6720x8_31_Mar_2021__23_47_11.jld2"
);

SIZE = (1450, 550)

#%% Unique values
results = load_results( res_file )

res_file_ws = string(splitext(res_file)[1], "_WS", ".jld2")
if isfile(res_file_ws)
    results_ws = load( res_file_ws )["results"]
    results_ws[:, :model] .= "WS"
    results_ws[:, :method] .= "COBYLA"
    results = vcat( results, results_ws )
end

res_file_ws_orbit = string(splitext(res_file)[1], "_WS_ORBIT", ".jld2")
if isfile(res_file_ws_orbit)
    results_ws = load( res_file_ws_orbit )["results"]
    results_ws[:, :model] .= "WS"
    results_ws[:, :method] .= "ORBIT"
    results = vcat( results, results_ws )
end

all_methods = unique( results[!,:method] )
all_models = unique( results[!, :model] )
all_n_vars = unique( results[!, :n_vars ] )

#%% PLOT 1 - Average Number of Evaluations per Run by Decision Space Dimension 

methods = sort(all_methods);
models = sort(all_models);

filtered_data = results[ 
    ( (res_method -> res_method ∈ methods).(results.method) ) .& 
    ( (res_model -> res_model ∈ models).(results.model) ),
:];

# split and average over different runs
grouped = groupby( filtered_data, [:model, :method, :n_vars] );
combined = unique( 
    combine( 
        grouped, 
        :model, :method, :n_vars, 
        :n_evals => mean => :avg_evals 
    ) 
);
#%%

mod_colors = Dict( 
    "cubic" => wong_black,
    "LP1" => wong_orange, 
    "TP1" => wong_sky_blue,
    "LP2" => wong_yellow,
    "WS" => wong_bluish_green
)
fig = Figure( resolution = (1450, 550) )

ax = Axis( fig[1,1] )
for (i,mod) ∈ enumerate( models )
    col = mod_colors[mod]
    mod_data = combined[ combined.model .== mod, :]
    sd_evals = mod_data[ mod_data.method .== "steepest_descent", 
        [:n_vars, :avg_evals]
    ]
    ps_evals = mod_data[ mod_data.method .== "ps", 
        [:n_vars, :avg_evals]
    ]
    scatterlines!(ax, sd_evals[:, :n_vars], sd_evals[:, :avg_evals]; 
        linewidth = 3, color = col, linestyle = nothing,
        markercolor = col, marker = :circle, markersize = 8 )
    scatterlines!(ax, ps_evals[:, :n_vars], ps_evals[:, :avg_evals]; 
        linewidth = 3, color = col, linestyle = [0,2,3],
        markercolor = col, marker = :circle, markersize = 8 )
end

col = mod_colors["WS"]
ws_data = combined[ combined.model .== "WS" , :]
ws_cobyla = ws_data[ ws_data.method .== "COBYLA", [:n_vars, :avg_evals]]
scatterlines!(ax, ws_cobyla[:, :n_vars], ws_cobyla[:, :avg_evals];
    linewidth = 3, color = col, linestyle = :dash,
    strokecolor = col, markercolor = col, marker = :star5, markersize = 20 )

ws_orbit = ws_data[ ws_data.method .== "ORBIT", [:n_vars, :avg_evals]]
scatterlines!(ax, ws_orbit[:, :n_vars], ws_orbit[:, :avg_evals];
    linewidth = 3, color = col, linestyle = :dash,
    strokecolor = col, markercolor = col, marker = :rect, markersize = 15)

leg_layout = GridLayout()

# model color legend
leg1 = leg_layout[1,1] = Legend(fig,
    [ LineElement( color = mod_colors[mod], linewidth = 5, linestyle = nothing ) for mod ∈ models ],
    [ string(mod) for mod ∈ models ],
    "Model";
    patchsize = (40,20)
)

leg2 = leg_layout[2,1] = Legend(fig,
    [ 
        [
            LineElement( color = :gray, linestyle = nothing, linewidth = 3 ),
            MarkerElement( color = :gray, marker = :circle, strokecolor = :black, markersize = 8 )
        ],
        [
            LineElement( color = :gray, linestyle = [0,2,3], linewidth = 3 ),
            MarkerElement( color = :gray, marker = :circle, strokecolor = :black, markersize = 8 )
        ],
        [
            LineElement( color = mod_colors["WS"], linestyle = :dash, linewidth = 3 ),
            MarkerElement( color = mod_colors["WS"], marker = :rect, strokecolor = mod_colors["WS"], markersize = 15 )
        ],
        [
            LineElement( color = mod_colors["WS"], linestyle = :dash, linewidth = 3 ),
            MarkerElement( color = mod_colors["WS"], marker = :star5, strokecolor = mod_colors["WS"], markersize = 15 )
        ]
    ],
    [ "SD", "PS", "ORBIT", "COBYLA" ],
    "Method";
    patchsize = (45,20)
)

fig[1,2] = leg_layout

title = Label(fig[0,:], "Expensive Evaluations by № of Decision Variables.",
    textsize = 30)

ax.xticks[] = all_n_vars
ylims!(ax, (0, 400))
xlims!(ax, (minimum(all_n_vars) - .2, maximum(all_n_vars) + .2 ))
ax.xlabel[] = "№ of decision variables";
ax.ylabel[] = "avg. № of evals";
ax.xlabelsize[] = ax.ylabelsize[] = 28;
fig

#%% Box Plots
filtered_data[ (filtered_data.model .== "WS") .& (filtered_data.method .== "COBYLA"), :model ] .= "WS_C"
filtered_data[ (filtered_data.model .== "WS") .& (filtered_data.method .== "ORBIT"), :model ] .= "WS_O"
    
function eval_crit_boxplots(dim, prefix)

    fig25 = Figure(resolution = (740, 420));

    ax11 = Axis(fig25[1,1])

    filtered_data[ isinf.(filtered_data.ω) , :ω] .= 0
    dim_data = filtered_data[ 
        (filtered_data.n_vars .== dim), : ]
    
    dim_models = unique( dim_data.model )
    N = length(dim_models)

    for (i,mod) ∈ enumerate(dim_models)
        y = Int.(dim_data[ dim_data.model .== mod , :n_evals ])
        if !isempty(y)
            boxplot!(fill(i, length(y)), 
                y,
                #rand(collect((i*10) : (i*10)+10), length(y)),
                markersize = 3,
                show_outliers = false,
            )
        end
    end

    ax11.xticks[] = collect(1:N)
    ax11.xtickformat[] = i -> dim_models[Int.(i)]
    ax11.xticklabelrotation[] = π/2
    ax11.ylabel[] = "№ evals (no outliers)"
    
    ax12 = fig25[1,2] = Axis(fig25) 

    for (i,mod) ∈ enumerate(dim_models)
        y = Float64.(dim_data[ dim_data.model .== mod , :ω ])
        if !isempty(y)
            boxplot!(ax12, fill(i, length(y)), y, markersize = 3,
            show_outliers = false
            )
        end
    end

    ax12.xticks[] = collect(1:N)
    ax12.xtickformat[] = i -> dim_models[Int.(i)]
    ax12.yaxisposition[] = :right
    ax12.xticklabelrotation[] = π/2
    ax12.ylabel[] = "ω (no outliers)"

    fig25[0,:] = Label(fig25, "($(prefix)) № of Evals & Criticality ($(dim) Vars)", textsize = 24)

    plot_file2 = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
    "omega_criticality_$(prefix)_$(dim).png"
    )
    saveplot(plot_file2, fig25);
    
    fig25
end
f = eval_crit_boxplots(5, "a")

#%% Plot3
ω_threshold = .1;
g = groupby(filtered_data, [:n_vars,:model]);
omg = unique(combine(g, :ω => (ω -> 100*sum(ω .< ω_threshold)/length(ω) ) => :solved));
dat = data(omg);

fig3 = Figure(resolution = (1450,600) );

plot_data = (
    dat * 
    visual( color = wong_array ) *
    mapping(
        :n_vars,
        :solved
    ) *
    mapping(dodge = :model => categorical, color = :model => categorical ) *
    visual(BarPlot)
);

AlgebraOfGraphics.draw!(fig3, plot_data);

ax = content(fig3[1,1][1,1]);

#xlims!(ax,[2,15]);
ax.xticks[] = all_n_vars;
ax.title[] = "Percentage of Solved Problems (ω < $(ω_threshold))."
ax.titlesize[] = 30f0;
ax.ylabel[] = "% of solved problems"
ax.xlabel[] = "№ of decision variables"
ax.xlabelsize[] = ax.ylabelsize[] = 24.0;

# Solved problem percentages
sol_gr = groupby( filtered_data, [:n_vars,:problem_str])
sol_pr = unique(combine( sol_gr, :ω => (ω -> 100*sum(ω .< ω_threshold)/length(ω) ) => :solved))

all_problems = unique( sol_pr.problem_str )
GRAYS = ([RGB(i) for i = LinRange(0,.9, length(all_problems))])

dat = data(sol_pr)
plot_data = (
    dat * 
    visual( color = GRAYS, markercolor = GRAYS ) *
    mapping(
        :n_vars,
        :solved
    ) *
    (
        mapping( color = :problem_str => categorical, 
        markercolor = :problem_str => categorical )
    ) *
    visual( ScatterLines,
        linewidth = 2,
        markersize = 8,
        strokewidth = 2,
    )
);
AlgebraOfGraphics.draw!(fig3, plot_data)

ax1 = contents(fig3[1,1][1,1])[1]
ax2 = contents(fig3[1,1][1,1])[2]

# move legends
leg1, leg2 = contents(fig3[1,2])
leg_sub = GridLayout()
leg_sub[1,1] = leg1
leg_sub[2,1] = leg2
delete!.(contents(fig3[1,2]))
fig3[1,2] = leg_sub
leg1.elements[:titletexts][1].text[] = "Model"
leg2.elements[:titletexts][1].text[] = "Problem"
leg2.patchsize[] = (40, 20)

hidespines!(ax2)
hidedecorations!(ax2)
linkaxes!(ax1, ax2)

lower_y = .95 * min( minimum( omg.solved ), minimum( sol_pr.solved ) );
upper_y = 1.01 *max( maximum( omg.solved ), maximum( sol_pr.solved) );
ylims!(ax1,[lower_y,upper_y]);

plot_file3 = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
 "percentage_of_solved_problems.png"
)
saveplot(plot_file3, fig3);
fig3