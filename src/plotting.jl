# plotting recipes
using RecipesBase
import Plots: palette, plot

export plot_decision_space, plot_objective_space

default_line_color = :cornflowerblue
default_pareto_color = :mediumseagreen
default_data_color = :lightgoldenrod
default_palette(n) = palette( :oslo, n; rev = false )
markersizes_fn(n) = map( x -> 3 + 8 * x^1.1, range(1,0; length = n ) )

# Decision Space Plotting
@userplot PlotDecisionSpace
@recipe function f( d :: PlotDecisionSpace )

    if !( isa( d.args[1], AlgoConfig ) )
        error("Plot of Decision Space needs an 'AlgoConfig' object as the only argument.")
    end

    if length(d.args) >= 2 && isa( d.args[2], Union{ Tuple{Int64, Int64}, Vector{Int64} } )
        ind_1, ind_2 = d.args[2][1:2]
    else
        ind_1, ind_2 = 1,2
    end

    opt_obj = d.args[1];
    iter_data = opt_obj.iter_data;
    all_sites = iter_data.sites_db;
    iter_ind = iter_data.iterate_indices;

    decision_x = [ site[ind_1] for site ∈ all_sites ];
    decision_y = [ site[ind_2] for site ∈ all_sites ];
    d_iter_x = reverse(decision_x[ iter_ind ]);
    d_iter_y = reverse(decision_y[ iter_ind ]);

    framestyle := :axes
    grid := true
    legend := false

    title --> "Decision Space."
    xguide --> "x_$ind_1"
    yguide --> "x_$ind_2"

    markershape := :circle

    @series begin
        markercolor := default_data_color
        markerstrokewidth := 0.0
        seriestype := :scatter
        markersize --> 5
        decision_x, decision_y
    end

    @series begin
        markercolor := default_palette( length(iter_ind) );
        markerstrokewidth := 0.1
        seriestype := :path
        linecolor := default_line_color
        linewidth := 1.5
        markersize --> markersizes_fn( length(iter_ind ) )
        d_iter_x, d_iter_y
    end
end

@recipe function f( pset :: ParetoSet, ind :: Vector{Int64} = [1,2] )
    markercolor := default_pareto_color
    markerstrokewidth := 0.0
    markersize --> 3

    seriestype := :scatter
    pset_x, pset_y = pset.coordinate_arrays[ ind ];
    return pset_x, pset_y
end

function plot_decision_space( opt_obj :: AlgoConfig, pset :: Union{Nothing, ParetoSet} = nothing; ind::Union{Tuple,Vector{Int64}} = (1,2) )
    ind = [ ind... ]    # turn tuple into list
    if !isnothing( pset )
        plot( pset, ind )
        plotdecisionspace!(opt_obj, ind )
    else
        plotdecisionspace(opt_obj,ind)
    end
end

# Objective Space Plotting

@userplot PlotObjectiveSpace
@recipe function f( d :: PlotObjectiveSpace )

    if !( isa( d.args[1], AlgoConfig ) )
        error("Plot of Objectives Space needs an 'AlgoConfig' object as the only argument.")
    end

    if length(d.args) >= 2 && isa( d.args[2], Union{ Tuple{Int64, Int64}, Vector{Int64} } )
        ind_1, ind_2 = d.args[2][1:2]
    else
        ind_1, ind_2 = 1,2
    end

    opt_obj = d.args[1];
    iter_data = opt_obj.iter_data;
    all_values = iter_data.values_db;
    iter_ind = iter_data.iterate_indices;

    f_1 = [ val[ind_1] for val ∈ all_values ];
    f_2 = [ val[ind_2] for val ∈ all_values ];
    f_iter_1 = reverse(f_1[ iter_ind ]);
    f_iter_2 = reverse(f_2[ iter_ind ]);

    framestyle := :axes
    grid := true
    legend := false

    title --> "Objective Space."
    xguide --> "f_$ind_1"
    yguide --> "f_$ind_2"

    # if Pareto Frontier is given for Comparison, Plot it first
    if length(d.args) >= 2
        pfront_1, pfront_2 = d.args[2];
        @series begin
            seriestype := :scatter
            markercolor := default_pareto_color
            markerstrokewidth := 0.0
            markersize --> 3
            pfront_1, pfront_2
        end
    end

    markershape := :circle

    @series begin
        markercolor := default_data_color;
        markerstrokewidth := 0.0
        seriestype := :scatter
        markersize --> 5
        f_1, f_2
    end

    @series begin
        markercolor := default_palette( length(iter_ind) )
        markerstrokewidth := 0.1
        seriestype := :path
        linecolor := default_line_color
        linewidth := 1.5
        markersize --> markersizes_fn( length(iter_ind) )
        f_iter_1, f_iter_2
    end
end


@recipe function f( pfront :: ParetoFrontier, ind :: Vector{Int64} = [1,2] )
    markercolor := default_pareto_color
    markerstrokewidth := 0.0
    markersize --> 3

    seriestype := :scatter
    pfront_1, pfront_2 = pfront.objective_arrays[ ind ];
    return pfront_1, pfront_2
end

function plot_objective_space( opt_obj :: AlgoConfig, pfront :: Union{Nothing, ParetoFrontier} = nothing; ind::Union{Tuple,Vector{Int64}} = (1,2) )
    ind = [ ind... ]    # turn tuple into list
    if !isnothing( pfront )
        plot( pfront, ind )
        plotobjectivespace!(opt_obj, ind )
    else
        plotobjectivespace(opt_obj,ind)
    end
end
