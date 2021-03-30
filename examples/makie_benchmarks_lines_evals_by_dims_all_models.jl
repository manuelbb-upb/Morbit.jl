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
    "results_XIHJrRkp_4480x8_30_Mar_2021__02_22_26.jld2"
);
plot_file = joinpath(ENV["HOME"], "Desktop", "PaperPlots",
 "line_plots_evals_by_vars_all_models.png"
)

SIZE = (1450, 550)

#%% Unique values
results = load_results( res_file )
all_methods = unique( results[!,:method] )
all_models = unique( results[!, :model] )
all_n_vars = unique( results[!, :n_vars ] )

#%% PLOT 1 - Average Number of Evaluations per Run by Decision Space Dimension 

methods = sort(all_methods);
models = sort(["cubic", "TP1", "LP1", "LP2"])

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

LINECOLORS = [upb_orange, upb_blue, upb_cassis, upb_lightblue ];
LINESTYLES = [ nothing, :dash ];
MARKERS = [:circle, :star5];

fig = Figure(resolution = (1500, 550));

plot_data = (
    data( combined ) *
    visual( color = LINECOLORS ) *
    visual( linewidth = 2 ) *
    mapping( 
        color = :model => categorical,
    ) *
    mapping(
        :n_vars,
        :avg_evals
    ) 
) * (
    visual( linestyle = LINESTYLES ) *
    mapping(linestyle = :method => categorical ) * 
    visual(Lines) + 
    visual( marker = MARKERS ) *
    mapping(marker = :method => categorical) * visual(Scatter) 
);

AlgebraOfGraphics.draw!(fig, plot_data)

# styling
# delete legend 
delete!(content(fig.layout[1,2]))

# symbols for colors 
color_elems = [ PolyElement( color = C, strokecolor = :transparent) for C ∈ LINECOLORS ]
style_elems = [ 
    [
    LineElement( linestyle = LINESTYLES[i], color = :black, strokewidth = 2.0, linepoints = Point2f0[ (0,.5), (2.0, .5)] ),
    MarkerElement( marker = MARKERS[i], strokecolor = :black, color = :black, markerpoints = [Point2f0(1, .5),])
    ] 
    for i = eachindex( methods) 
];

legend = fig[1,2] = Legend(
    fig.scene,
    [color_elems, style_elems ],
    [String.(models), String.(methods)],
    ["Model", "Method"]
);

legend.patchsize[] = (35,20)
legend.patchlabelgap = 45;
legend.titlesize = 26;
legend.labelsize = 22;

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

#saveplot(plot_file,fig)
fig

#%% Plot 2
filtered_data2 = filtered_data[ filtered_data.method .== "steepest_descent", : ]
filtered_data2[ isinf.(filtered_data2[:,:ω]), :ω ] .= 0.0

dim = 10;
prefix = "a"
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
    mapping( :model => categorical, :n_evals ) *
    visual( BoxPlot; markersize = 2.5 )
);

AlgebraOfGraphics.draw!(fig2, plot_data_n)

ax1 = content(fig2[1,1][1,1])
ax2 = content(fig2[1,2][1,1])

ax1.ylabel[] = " № of evaluations"

# move y axis of 2nd plot to right
ax2.yaxisposition[] = :right 
ax2.yticklabelalign[] = (:left, :center)

ax1.xticks[] = ax2.xticks[] = LinearTicks( length(models) + 1 )
ax2.xtickformat[] = ax1.xtickformat[] = xs -> [String(models[i]) for i=eachindex(xs)]

title = fig2[0,:] = Label(fig2.scene, "($(prefix)) № of Evals & Criticality ($(dim) Vars)", textsize = 24)

#savefunc("box_plots_dim_5_evals_omega.png", fig2);
fig2 

#%% Plot 3

#%%
g = groupby(filtered_data2, [:n_vars,:model]);
omg = unique(combine(g, :ω => (ω -> 100*sum(ω .< 0.1)/length(ω) ) => :solved));
dat = data(omg);
LINECOLORS = [upb_orange, upb_blue, upb_cassis, upb_lightblue, upb_lightgray ];

fig3 = Figure(resolution = SIZE );

plot_data = (
    dat * 
    visual( color = LINECOLORS ) *
    mapping(
        :n_vars,
        :solved
    ) *
    mapping(dodge = :model => categorical, color = :model => categorical ) *
    visual(BarPlot)
);

AlgebraOfGraphics.draw!(fig3, plot_data);

ax = content(fig3[1,1][1,1]);
ylims!(ax,[60.0,102]);
#xlims!(ax,[2,15]);
ax.xticks[] = all_n_vars;
ax.title[] = "Percentage of Solved Problems (ω < 0.1)."
ax.titlesize[] = 30f0;
ax.ylabel[] = "% of solved problems"
ax.xlabel[] = "№ of decision variables"
ax.xlabelsize[] = ax.ylabelsize[] = 24.0;

fig3