using Pkg;
Pkg.activate( @__DIR__ );

using VegaLite
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
    "results_gHb5Xl8h_4320x8_29_Mar_2021__14_55_59.jld2"
);
plot_file = joinpath(ENV["HOME"], "Desktop", "PaperPlots", "line_plots_evals_by_vars_all_models.svg")

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

LINECOLORS = [upb_orange_str, upb_blue_str, upb_cassis_str, upb_lightblue_str ];

plt = combined |> @vlplot(
    {:line,
        clip = true,
        strokeWidth = 3,
        strokeCap = "round",
        point ={
            size = 60,
        }
    },
    x={
        :n_vars, 
        title="№ of decision variables", 
        axis = {
            grid=true,
            values = all_n_vars
        }
    },
    y={
        :avg_evals,
        axis = {
            grid=true,
            title="avg № of evaluations", 
            values = collect(0:500:1500),
        },
        scale = {
            domain = [0,1500]
        }
    },
    
    strokeDash={
        field=:method, 
        title="Method",
        type="nominal",
        scale = {
            domain = ["ps", "steepest_descent"],
            range = [ [1,0], [10,6] ]
        },
    },
    # NOTE dashed line legend needs newer vegal lite version
    shape = {
        field = :method,
        title = "Method",
        type = "nominal",
        scale = {
            domain = ["ps", "steepest_descent"],
        },
    },    
    color={:model, 
        type="nominal",
        title = "Model",
        scale= {
            domain = ["LP1", "LP2", "TP1", "cubic"],
            range = LINECOLORS,
        }
    },
    
    width=1450,
    height=550,
    title={
        text = "Expensive Evaluations by № of Decision Variables.",
        fontSize = 30,
        font = "Open Sans",
    },
    config = {
        axis = {
            titleFontSize = 20,
            titleFont = "Open Sans",
        }
    },
)
#save(joinpath(ENV["HOME"], "Desktop", "lines.vegalite"), plt)