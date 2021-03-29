using Pkg;
Pkg.activate( @__DIR__ );

#using CairoMakie, AbstractPlotting;
#using AbstractPlotting.MakieLayout;
using Gadfly
using Statistics
using Colors;
using DataFrames;
using FileIO, JLD2
using Compose
# disable inline plotting
#AbstractPlotting.inline!(true);

#%% load helpers 
include("plot_helpers/upb_colors.jl");
include("plot_helpers/loading_saving.jl")

#%% settings 
res_file = joinpath(
    ENV["HOME"], "MORBIT_BENCHMARKS", 
    "results_gMszO2yB_6720x8_29_Mar_2021__12_04_54.jld2"
);
plot_file = joinpath(ENV["HOME"], "Desktop", "PaperPlots", "line_plots_evals_by_vars_all_models.svg")
out_svg = SVG(plot_file, 14.5cm, 5.5cm);
set_default_plot_size(14.5cm, 5.5cm)
#SIZE = (1450, 550);

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

upb_theme = Theme(;
    background_color = colorant"white",
    discrete_color_scale = Scale.color_discrete_manual(upb_orange, upb_blue, upb_cassis, upb_lightblue),
    plot_padding=[0.02w, 0pt, 0.04h, 0pt]
);

Gadfly.with_theme(upb_theme) do
    plt = plot( combined, x = :n_vars, y = :avg_evals, Geom.line, Geom.point, 
        color = :model, linestyle= :method, shape = :method,
        Guide.title("Expensive Evaluations by № of Decision Variables."),
        Guide.ylabel("avg. № of evals"), Guide.xlabel("№ of decision variables"),
        Guide.colorkey(title="Model";),
        Guide.manual_discrete_key("Method", ["PS", "SD"];
            color=[colorant"black", colorant"black"], 
            shape = [Shape.circle, Shape.square]),
        Guide.shapekey(nothing),
        Coord.cartesian(;xmin=min(all_n_vars...), xmax =max(all_n_vars...), ymin = 0, ymax = 1500),
        Scale.y_continuous(; labels = x -> "$(round(x/1000;digits=1))K")
    )
end
