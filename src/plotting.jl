# plotting recipes
using RecipesBase

# Decision Space Plotting
@userplot PlotDecisionSpace
@recipe function f( d :: PlotDecisionSpace )

    if !( isa( d.args[1], AlgoConfig ) )
        error("Plot of Decision Space needs an 'AlgoConfig' object as the only argument.")
    end

    opt_obj = d.args[1];
    iter_data = opt_obj.iter_data;
    all_sites = iter_data.sites_db;
    iter_ind = iter_data.iterate_indices;

    decision_x = [ site[1] for site ∈ all_sites ];
    decision_y = [ site[2] for site ∈ all_sites ];
    d_iter_x = decision_x[ iter_ind ];
    d_iter_y = decision_y[ iter_ind ];

    framestyle := :axes
    grid := true
    legend := false

    title --> "Decision Space."
    xguide --> "x_1"
    yguide --> "x_2"

    # if Pareto Set is given for Comparison, Plot it first
    if length(d.args) >= 2
        pset_x, pset_y = d.args[2];
        @series begin
            seriestype := :scatter
            markercolor := :mediumseagreen
            markerstrokewidth := 0.0
            markersize --> 3
            pset_x, pset_y
        end
    end

    markershape := :circle

    @series begin
        markercolor := :red
        seriestype := :scatter
        markersize --> 5
        decision_x, decision_y
    end

    @series begin
        markercolor := :lightgoldenrod
        seriestype := :path
        linecolor := :cornflowerblue
        linewidth := 1.5
        markersize --> map( x -> 5 + 5 * x^4, range(0,1, length = length(iter_ind)) )
        d_iter_x, d_iter_y
    end
end

# Objective Space Plotting

@userplot PlotObjectiveSpace
@recipe function f( d :: PlotObjectiveSpace )

    if !( isa( d.args[1], AlgoConfig ) )
        error("Plot of Objectives Space needs an 'AlgoConfig' object as the only argument.")
    end

    opt_obj = d.args[1];
    iter_data = opt_obj.iter_data;
    all_values = iter_data.values_db;
    iter_ind = iter_data.iterate_indices;

    f_1 = [ val[1] for val ∈ all_values ];
    f_2 = [ val[2] for val ∈ all_values ];
    f_iter_1 = f_1[ iter_ind ];
    f_iter_2 = f_2[ iter_ind ];

    framestyle := :axes
    grid := true
    legend := false

    title --> "Objective Space."
    xguide --> "f_1"
    yguide --> "f_2"

    # if Pareto Frontier is given for Comparison, Plot it first
    if length(d.args) >= 2
        pfront_1, pfront_2 = d.args[2];
        @series begin
            seriestype := :scatter
            markercolor := :mediumseagreen
            markerstrokewidth := 0.0
            markersize --> 3
            pfront_1, pfront_2
        end
    end

    markershape := :circle

    @series begin
        markercolor := :red
        seriestype := :scatter
        markersize --> 5
        f_1, f_2
    end

    @series begin
        markercolor := :lightgoldenrod

        seriestype := :path
        linecolor := :cornflowerblue
        linewidth := 1.5
        markersize --> map( x -> 5 + 5 * x^4, range(0,1, length = length(iter_ind)) )
        f_iter_1, f_iter_2
    end
end
