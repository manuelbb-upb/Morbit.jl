# plotting recipes
using RecipesBase
using RecipesBase: plot
using ColorSchemes: oslo, grayC, RGB

import Base: +
+(c::RGB, v :: Float64) = RGB( min(c.r + v, 1.0), min(c.g + v, 1.0), min(c.b + v,1.0) )

export plot_decision_space, plot_objective_space

default_line_color = :cornflowerblue
default_pareto_color = :mediumseagreen
default_data_color = :lightgoldenrod
default_palette(n) = n > 1 ? get(oslo, range(0,1;length=n)) : [RGB(zeros(3)...)]
markersizes_fn(n) = n > 1 ? map( x -> 3 + 3 * x^1.1, range(1,0; length = n ) ) : 3;

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

    n_vars = length( all_sites[1] )

    decision_x = [ site[ind_1] for site ∈ all_sites ];
    d_iter_x = reverse(decision_x[ iter_ind ]);
    if n_vars >= 2
        decision_y = [ site[ind_2] for site ∈ all_sites ];
        d_iter_y = reverse(decision_y[ iter_ind ]);
    else
        decision_y = zeros( length(all_sites) );
        d_iter_y = zeros( length(iter_ind) );
    end

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

    n_out = length( all_values[1] )

    f_1 = [ val[ind_1] for val ∈ all_values ];
    f_iter_1 = reverse(f_1[ iter_ind ]);

    if n_out >= 2
        f_2 = [ val[ind_2] for val ∈ all_values ];
        f_iter_2 = reverse(f_2[ iter_ind ]);
    else
        f_2 = zeros( length(all_values) )
        f_iter_2 = zeros(length(iter_ind))
    end

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

@userplot PlotStepSizes
@recipe function f( d :: PlotStepSizes )

    if !( isa( d.args[1], AlgoConfig ) )
        error("plotstepsizes needs an 'AlgoConfig' object as the only argument.")
    end

    iter_data = d.args[1].iter_data;
    stepsizes = iter_data.stepsize_array;
    ρs = iter_data.ρ_array;
    linear_flags = [mi.fully_linear for mi ∈ iter_data.model_meta.model_info_array];
    iterations = 1:length(stepsizes)

    title := "Step Sizes."
    @series begin
        seriestype := :path
        markerstrokewidth := 0.1
        linecolor := default_line_color
        label := ""
        iterations, stepsizes
    end

    green_points = findall( ρs .>= d.args[1].ν_success )
    blue_points = findall( d.args[1].ν_success .> ρs .>= d.args[1].ν_accept )
    red = setdiff( iterations, [green_points;blue_points]);
    red_points = red[ linear_flags[red] ]
    red_diamonds = red[ .!(linear_flags)[red] ]

    @series begin
        seriestype := :scatter
        markercolor := :green
        label := "successfull"
        green_points, stepsizes[green_points]
    end

    @series begin
        seriestype := :scatter
        markercolor := :blue
        label := "acceptable"
        blue_points, stepsizes[blue_points]
    end

    @series begin
        seriestype := :scatter
        markercolor := :red
        label := "unsucessfull"
        red_points, stepsizes[red_points]
    end

    @series begin
        seriestype := :scatter
        markercolor := :orange
        label := "model improving"
        marker := :diamond
        red_diamonds, stepsizes[red_diamonds]
    end
end

@userplot PlotFunctionValues
@recipe function f( d::PlotFunctionValues )
    if !( isa( d.args[1], AlgoConfig ) )
        error("plotstepsizes needs an 'AlgoConfig' object as the first argument.")
    end

    iter_data = d.args[1].iter_data;
    iterate_indices = iter_data.iterate_indices;
    n_iters = length(iterate_indices);

    iter_colors = default_palette( n_iters );

    if length( d.args ) >= 2 && isa( d.args[2], Vector{Int64} )
        plot_iter_indices = d.args[2][:];
        plot_iter_indices = plot_iter_indices[ 1 .<= plot_iter_indices .<= n_iters ]
    else
        n = min( n_iters, 6 )

        plot_iter_indices = n_iters == 1 ? [1] : unique(ceil.(Int64, exp10.(range(0, log10(n_iters); length=n))))
        plot_iter_indices = plot_iter_indices[ 1 .<= plot_iter_indices .<= n_iters ]
        if length(plot_iter_indices) < 3
            if n_iters >= 2 plot_iter_indices = [1; n_iters]; end
            if n_iters >= 3
                push!(plot_iter_indices, ceil(Int64, n_iters./2) )
            end
        end
    end
    sort!(plot_iter_indices)

    unscaled_vals = iter_data.values_db[iterate_indices];
    n_out = length(unscaled_vals[1])

    extrema_tuples = extrema( hcat( unscaled_vals... ), dims = n_out == 1 ? 1 : 2 )
    MAX = [e[2] for e in extrema_tuples]
    MIN = [e[1] for e in extrema_tuples]
    vals = [(v .- MIN)./(MAX .- MIN) for v in unscaled_vals[plot_iter_indices] ]

    n_out = length( vals[1] )
    if n_out == 1
        vals = [ [v[1]; v[1]] for v in vals ]
    end

    title := "(Scaled) Objective Values."
    grid := true
    xguide := "Function index."

    @series begin
        seriestype := :path;
        #fillcolor := :match;
        fillrange := 0;
        #fillalpha := .25

        seriescolor := reverse(iter_colors)[ plot_iter_indices ]'
        linecolor := length(plot_iter_indices) > 1 ? get(grayC, range(1,0;length=n)) : :lightgray

        ticks := n_out == 1 ? [1] : collect( 1 : n_out )

        label := reshape( ["it. $i" for i in plot_iter_indices], (1, length(plot_iter_indices)) )

        vals
    end
end
