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
"results_XwSD7zMO_960x8_31_Mar_2021__20_32_36.jld2"
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

#ls = Dict( "ps" => nothing, "steepest_descent" => :dashdot, "COBYLA" => :dash )
#transform!( combined, :method => ByRow( x -> ls[x] ) => :ls) 

plot_data = (
    data( combined ) *
    visual( color = wong_array ) *
    visual( markercolor = wong_array ) *
    visual( linestyle = [:dash, nothing, :dashdot]) *
    visual( linewidth = 3 ) *
    #visual( linestyle = ls ) *
    mapping( 
        color = :model => categorical,
        markercolor = :model => categorical,
        linestyle = :method => categorical, 
    ) *
    mapping(
        :n_vars,
        :avg_evals,
    ) *    
    visual(ScatterLines, markersize = 10)
);

#%%
fig = Figure(resolution = (1500, 550));
AlgebraOfGraphics.draw!(fig, plot_data)

# styling
# delete legend 
legend = content(fig[1,2])

legend.elements[:titletexts][1].text[] = "Model"
legend.elements[:titletexts][2].text[] = "Method"

for elem ∈ legend.elements[:entrytexts][2]
    if elem.text[] == "steepest_descent"
        elem.text[] = "sd"
    end
end

legend.patchsize[] = (40,20)
legend.patchlabelgap[] = 10;
legend.titlesize[] = 26;
legend.labelsize[] = 22;

# style axes 
ax = content(content(fig[1,1])[1,1]);
ylims!(ax, [0.0, 600]);
ax.xlabel[] = "№ of decision variables";
ax.ylabel[] = "avg. № of evals";
ax.xlabelsize[] = ax.ylabelsize[] = 28;

#ax.ytickformat[] = ys -> ["$(round(y/1e3; digits=1))K" for y ∈ ys];
ax.xticklabelsize[] = ax.yticklabelsize[] = 25;
ax.xticks[] = all_n_vars

# set title
title = fig[0,:] = Label(
    fig.scene, 
    "Expensive Evaluations by № of Decision Variables."; 
    textsize = 30.0
);

plot_file = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
 "line_plots_evals_by_vars_all_models.png"
)
#saveplot(plot_file,fig)
fig

#%% Plot 2
#=
filtered_data2 = filtered_data[ filtered_data.method .== "steepest_descent", : ]
    #(filtered_data.model .!= "ws") , : ]

filtered_data2[ isinf.(filtered_data2[:,:ω]), :ω ] .= 0.0

dim = 2;
prefix = "b"
dim_data = filtered_data2[ filtered_data2.n_vars .== dim, : ]
plot_data_ω = (
    data( dim_data ) * 
    mapping( :model => categorical, :ω ) *
    visual( BoxPlot; markersize = 2.5 )
);

fig2 = Figure(resolution = (740, 420));
AlgebraOfGraphics.draw!(fig2, plot_data_ω)

# move to right side 
fig2[1,2] = contents(fig2[1,1])

plot_data_n = (
    data( dim_data ) * 
    mapping( :model , :n_evals ) *
    mapping( ticks = :model => categorical ) *
    visual( BoxPlot; markersize = 2.5 )
);

AlgebraOfGraphics.draw!(fig2, plot_data_n)

ax1 = content(fig2[1,1][1,1])
ax2 = content(fig2[1,2][1,1])

ax1.ylabel[] = " № of evaluations"

# move y axis of 2nd plot to right
ax2.yaxisposition[] = :right 
ax2.yticklabelalign[] = (:left, :center)

ax1.xticks[] = ax2.xticks[] = LinearTicks( length(unique(dim_data.model)) + 1 )
#ax2.xtickformat[] = ax1.xtickformat[] = xs -> [String(models[i]) for i=eachindex(xs)]

title = fig2[0,:] = Label(fig2.scene, "($(prefix)) № of Evals & Criticality ($(dim) Vars)", textsize = 24)

plot_file2 = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
 "omega_criticality_$(prefix)_$(dim).png"
)
saveplot(plot_file2, fig2);
fig2 
=#

#%%
dim = 5
prefix = "a"
function eval_crit_boxplots(dim, prefix)

    fig25 = Figure(resolution = (740, 420));

    ax11 = fig25[1,1] = Axis(fig25)

    filtered_data[ isinf.(filtered_data.ω) , :ω] .= 0
    dim_data = filtered_data[ filtered_data.n_vars .== dim, : ]

    N = length(models)

    for (i,mod) ∈ enumerate(models)
        y = dim_data[ dim_data.model .== mod , :n_evals ]
        if !isempty(y)
            boxplot!(ax11, i .* ones(length(y)), y, markersize = 3)
        end
    end

    ax11.xticks[] = collect(1:N)
    ax11.xtickformat[] = i -> models[Int.(i)]
    ax11.ylabel[] = "№ of evaluations"

    ax12 = fig25[1,2] = Axis(fig25) 

    N = length(models)
    for (i,mod) ∈ enumerate(models)
        @show y = dim_data[ dim_data.model .== mod , :ω ]
        if !isempty(y)
            boxplot!(ax12, i .* ones(length(y)), y, markersize = 3,
            show_outliers = false
            )
        end
    end

    ax12.xticks[] = collect(1:N)
    ax12.xtickformat[] = i -> models[Int.(i)]
    ax12.yaxisposition[] = :right
    ax12.ylabel[] = "ω (no outliers)"

    fig25[0,:] = Label(fig25, "($(prefix)) № of Evals & Criticality ($(dim) Vars)", textsize = 24)

        
    plot_file2 = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
    "omega_criticality_$(prefix)_$(dim).png"
    )
    saveplot(plot_file2, fig25);
    fig25
end
eval_crit_boxplots(5, "a")

#%% Plot3
ω_threshold = .1;
g = groupby(filtered_data, [:n_vars,:model]);
omg = unique(combine(g, :ω => (ω -> 100*sum(ω .< ω_threshold)/length(ω) ) => :solved));
dat = data(omg);

fig3 = Figure(resolution = SIZE );

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