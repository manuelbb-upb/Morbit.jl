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
    "results_WoJEMTIw_3600x8_29_Mar_2021__18_13_49.jld2"
);
plot_file = joinpath(ENV["HOME"], "Desktop", "PaperPlots", "line_plots_evals_by_vars_all_models.png")

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
ylims!(ax, [0.0, 200]);
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

saveplot(plot_file,fig)
fig