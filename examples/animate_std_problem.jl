using Morbit
using LinearAlgebra: norm
using Plots
using LaTeXStrings

cd(ENV["HOME"])
if !isdir("test_anim")
    mkdir("test_anim")
end
cd("test_anim")

f1(x) = sum( (x .- 1.0).^2 )
f2(x) = sum( (x .+ 1.0).^2 )

x0 = [10.0; -6.0]
#x0 = rand(2)

mop = MixedMOP()
add_objective!(mop, f1, :expensive)
add_objective!(mop, f2, :expensive)
opt = AlgoConfig( max_iter = 20, Δ_max = 2.0, sampling_algorithm = :monte_carlo )
optimize!(opt, mop, x0)

##

SIZE = (450,500)
MARKERSIZE = 8

@userplot TrustRegionPlot
@recipe function f(trp::TrustRegionPlot)
    x, Δ = trp.args

    linestyle := :solid
    fillcolor := :orange
    fillalpha --> .15
    linecolor := :orange
    linealpha --> 1.2 * plotattributes[:fillalpha]
    aspec_ratio := :equal
    legend := false

    ll = x .- Δ
    ur = x .+ Δ
    @series begin
        label = []
        Shape( [ ll[1]; ur[1]; ur[1]; ll[1]; ll[1] ], [ ll[2]; ll[2]; ur[2]; ur[2]; ll[2] ] )
    end
end

@userplot DecisionIterPlot
@recipe function f(dip::DecisionIterPlot)
    x, x₊= dip.args

    legend --> false

    @series begin
        seriestype := :line
        linecolor := :red
        linewidth := 1.0

        d = x₊ .- x
        d /= norm(d)
        eps = .05
        [ x[1] + eps*d[1], x₊[1] - eps *d[1] ], [ x[2] + eps*d[2], x₊[2] - eps *d[2] ]
    end

    markersize --> MARKERSIZE
    seriestype := :scatter

    @series begin
        markershape --> :circle
        markercolor --> :green
        [x[1]], [x[2]]
    end

    @series begin
        markershape --> :diamond
        markercolor --> :blue
        [x₊[1]], [x₊[2]]
    end
end

@userplot ObjectiveIterPlot
@recipe function f(dip::ObjectiveIterPlot)
    fx, fx₊= dip.args

    legend --> false

    @series begin
        seriestype := :line
        linecolor := :red
        linewidth := 1.0

        d = fx₊ .- fx
        d /= norm(d)
        eps = .05
        [ fx[1] + eps*d[1], fx₊[1] - eps *d[1] ], [ fx[2] + eps*d[2], fx₊[2] - eps *d[2] ]
    end

    markersize --> MARKERSIZE
    seriestype := :scatter

    @series begin
        markershape --> :circle
        markercolor --> :green
        [fx[1]], [fx[2]]
    end

    @series begin
        markershape --> :diamond
        markercolor --> :blue
        [fx₊[1]], [fx₊[2]]
    end
end

##
pyplot()

N = length(opt.iter_data.iterate_indices) - 1

id = opt.iter_data

ν_accept = opt.ν_accept
ν_success = opt.ν_success

xlims,ylims = extrema( hcat(id.sites_db...), dims = 2 )
xlims = xlims .+ 1e-2 * (xlims[2]-xlims[1]) .* (-1.0,1.0)
ylims = ylims .+ 1e-2 .* (ylims[2]-ylims[1]) .* (-1.0,1.0)

xlims2,ylims2 = extrema( hcat(id.values_db...), dims = 2 )
xlims2 = xlims2 .+ 1e-2 * (xlims2[2]-xlims2[1]) .* (-1.0,1.0)
ylims2= ylims2 .+ 1e-2 .* (ylims2[2]-ylims2[1]) .* (-1.0,1.0)

PS = vcat( hcat( range(-1,1;length=50)... ), hcat( range(-1,1;length=50)... ) )
PF = hcat( f1.(eachcol(PS)), f2.(eachcol(PS)) )'

MARKERSIZE = 8

anim_decision_space = @animate for i = 1 : N, j = 1 : 3
    #global ν_accept, ν_success
    x_ind = id.iterate_indices[i]
    x₊_ind = id.trial_point_indices[i]

    Δ = id.Δ_array[i]
    ρ = id.ρ_array[i]

    x = id.sites_db[x_ind]
    x₊ = id.sites_db[x₊_ind]
    fx = id.values_db[x_ind]
    fx₊ = id.values_db[x₊_ind]

    model_point_indices = unique(vcat( x_ind,
        id.model_info_array[i].round1_indices...,
        id.model_info_array[i].round2_indices...,
        id.model_info_array[i].round3_indices...
    ))
    model_points = hcat( id.sites_db[model_point_indices]...);
    model_values = hcat( id.values_db[model_point_indices]...);

    other_points = hcat( id.sites_db[ 1:x₊_ind ]... );
    other_values = hcat( id.values_db[ 1:x₊_ind ]... );

    previous_iterates = hcat( id.sites_db[ id.iterate_indices[1:i] ]... )
    previous_values = hcat( id.values_db[ id.iterate_indices[1:i] ]... )

    p1 = scatter( range(-1,1;length=50), range(-1,1;length=50); markercolor=:lightblue, markerstrokecolor=nothing, markersize = MARKERSIZE/2,
        xlabel = L"x_1", ylabel = L"x_2", size = SIZE, thickness_scaling = 1.3, guidefontsize = 14, legend = false , grid = false )#, aspect_ratio = :equal)
    title!("Decision Space.")
    xlims!( xlims)
    ylims!( ylims)

    scatter!(other_points[1,:], other_points[2,:]; markersize = MARKERSIZE-2 , markercolor = :yellow, markeralpha = .6)
    scatter!(model_points[1,:], model_points[2,:]; markersize = MARKERSIZE+3, markershape = :diamond, markercolor = nothing, markerstrokecolor = :black, markerstrokealpha = 1.0)

    if i > 1
        plot!( previous_iterates[1,:], previous_iterates[2,:]; markersize = MARKERSIZE, markercolor = :orange, linecolor = :red, markershape = :circle)
        for k = 1 : i - 1
            alph_val = .1 * ( k/(i-1) )
            trustregionplot!( previous_iterates[:,k], id.Δ_array[k]; fillalpha = alph_val )
        end
    end

    if j == 1
        trustregionplot!(x, Δ)
        decisioniterplot!(x, x₊)
    elseif j == 2
        trustregionplot!(x, Δ)
        ms = ρ >= ν_accept ? [:none :circle :circle] : [:none :circle :star5]
        mc = ρ >= ν_accept ? [:red :green :green] : [:red :green :red]
        decisioniterplot!(x, x₊; markershape = ms, markercolor = mc)
    elseif j == 3
        trustregionplot!(x, Δ)
        ms = ρ >= ν_accept ? [:none :circle :circle] : [:none :circle :none]
        mc = ρ >= ν_accept ? [:red :orange :green] : [:red :green :red]
        decisioniterplot!(x, x₊; markershape = ms, markercolor = mc)
    end

    ###############

    p2 = scatter( PF[1,:], PF[2,:]; markercolor=:lightblue, markerstrokecolor=nothing, markersize = MARKERSIZE/2,
        xlabel = L"f_1(\mathbf{x})", ylabel = L"f_2(\mathbf{x})", size = SIZE, thickness_scaling = 1.3, guidefontsize = 14, legend = false )

    title!("Objective Space.")

    scatter!(other_values[1,:], other_values[2,:]; markersize = MARKERSIZE-2 , markercolor = :yellow, markeralpha = .6)
    scatter!(model_values[1,:], model_values[2,:]; markersize = MARKERSIZE+3, markershape = :diamond, markercolor = nothing, markerstrokecolor = :black, markerstrokealpha = 1.0)

    plot!(previous_values[1,:], previous_values[2,:]; markersize = MARKERSIZE, markercolor = :orange, linecolor = :red, markershape = :circle)

    if j == 1
        objectiveiterplot!(fx, fx₊)
    elseif j == 2
        ms = ρ >= ν_accept ? [:none :circle :circle] : [:none :circle :star5]
        mc = ρ >= ν_accept ? [:red :green :green] : [:red :green :red]
        objectiveiterplot!(fx, fx₊; markershape = ms, markercolor = mc)
    elseif j == 3
        ms = ρ >= ν_accept ? [:none :circle :circle] : [:none :circle :none]
        mc = ρ >= ν_accept ? [:red :orange :green] : [:red :green :red]
        objectiveiterplot!(fx, fx₊; markershape = ms, markercolor = mc)
    end

    xlims!(xlims2)
    ylims!(ylims2)

    plot( p1, p2; size = (1000, 500) )

end

#gif( anim_decision_space, "decision_space_iteration.gif", fps = 3)
#mp4( anim_decision_space, "decision_space_iteration.mp4", fps = 2)
mov( anim_decision_space, "decision_space_iteration.mov", fps = 2)
#cp(anim_decision_space.dir, joinpath(pwd(), "tmp"); force = true)

@userplot ObjectivePathPlot
@recipe function f( opp::ObjectivePathPlot )
    x0, x1, func = opp.args

    d = x1 .- x0
    t = range(0, 1; length=30)

    p = hcat( [ x0 .+ τ .* d for τ in t ]... )

    seriestype := :line
    linecolor --> :red
    return( p[1,:], p[2,:], func.(eachcol(p)))
end

MARKERSIZE = 8
X = range(xlims...,length=30)
Y = range(ylims..., length=30)

F1vals = [f1( [x;y] ) for y ∈ Y, x ∈ X ]
F2vals = [f2( [x;y] ) for y ∈ Y, x ∈ X ]

z1lims = ( 0, maximum( [v[1] for v in id.values_db[id.iterate_indices] ] ) )
z2lims = ( 0, maximum( [v[2] for v in id.values_db[id.iterate_indices] ] ) )

clevels = [0.1, 0.5, 2.0, 8.0, 32.0, 64.0, 132.0]

anim_objectives = @animate for i = 1 : N, j = 1 : 2
    #global ν_accept, ν_success
    x_ind = id.iterate_indices[i]
    x₊_ind = id.trial_point_indices[i]

    ρ = id.ρ_array[i]

    x = id.sites_db[x_ind]
    x₊ = id.sites_db[x₊_ind]
    fx = id.values_db[x_ind]
    fx₊ = id.values_db[x₊_ind]

    mcolors = j == 2 ? ( ρ >= ν_accept ? (:green, :green) : (:green, :red) ) : (:green, :blue)

    previous_iterates = hcat( id.sites_db[ id.iterate_indices[1:i] ]... )
    previous_values = hcat( id.values_db[ id.iterate_indices[1:i] ]... )

    p1 = plot(X, Y, F1vals; seriestype = :surface, seriescolor = cgrad(:dense, alpha = .65), legend = false, colorbar = :none, camera = (-35,35), size = SIZE )
    scatter!([[1.0]], [[1.0]], [[0.0]]; markershape = :star7, markercolor = :black, markersize = MARKERSIZE/2 )
    contour!(X,Y,F1vals; seriescolor = cgrad(:dense, alpha = .65), levels = clevels )
    title!(string( L"f_1" ,", min. at (1,1)."))
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(xlims)
    ylims!(ylims)
    zlims!(z1lims)

    for k = 1 : i - 1
        objectivepathplot!( previous_iterates[:,k], previous_iterates[:, k+1], f1 )
        scatter!([previous_iterates[1,k]], [previous_iterates[2,k]], [previous_values[1,k]]; markercolor = :orange, markersize = MARKERSIZE - 2)

        scatter!([previous_iterates[1,k]], [previous_iterates[2,k]]; markercolor = :orange, markeralpha = .35, markersize = MARKERSIZE/2)
        plot!([previous_iterates[1,k], previous_iterates[1, k+1]], [previous_iterates[2,k], previous_iterates[2, k+1]]; st = :line, linecolor = :orange, linealpha = .25 )
    end

    objectivepathplot!(x, x₊, f1)
    plot!( [x[1], x₊[1]], [x[2], x₊[2]];st = :line, linecolor = :orange, linealpha = .25)

    scatter!([x[1]],[x[2]],[fx[1]] ; markercolor = mcolors[1], markersize = MARKERSIZE)
    scatter!([x[1]],[x[2]],; markercolor = mcolors[1], markeralpha = .35, markersize = MARKERSIZE/2)
    scatter!([x₊[1]],[x₊[2]],[fx₊[1]] ; markercolor = mcolors[2], markersize = MARKERSIZE)
    scatter!([x₊[1]],[x₊[2]],; markercolor = mcolors[2], markeralpha = .35, markersize = MARKERSIZE/2)

    #### 2ND OBJECTIVE

    p2 = plot(X, Y, F2vals; seriestype = :surface, seriescolor = cgrad(:dense, alpha = .65), legend = false, colorbar = :none, camera = (-40,45), size = SIZE )
    scatter!([[-1.0]], [[-1.0]], [[0.0]]; markershape = :xcross, markercolor = :black, markersize = MARKERSIZE/2 )
    plot!(X,Y,F2vals; st = :contour, seriescolor = cgrad(:dense, alpha = .65), levels = clevels )
    title!(string( L"f_2" ,", min. at (-1,-1)."))
    xlabel!(L"x_1")
    ylabel!(L"x_2")
    xlims!(xlims)
    ylims!(ylims)
    zlims!(z2lims)

    for k = 1 : i - 1
        objectivepathplot!( previous_iterates[:,k], previous_iterates[:, k+1], f2 )
        scatter!([previous_iterates[1,k]], [previous_iterates[2,k]], [previous_values[2,k]]; markercolor = :orange, markersize = MARKERSIZE - 2)

        scatter!([previous_iterates[1,k]], [previous_iterates[2,k]]; markercolor = :orange, markeralpha = .35, markersize = MARKERSIZE/2)
        plot!([previous_iterates[1,k], previous_iterates[1, k+1]], [previous_iterates[2,k], previous_iterates[2, k+1]]; st = :line, linecolor = :orange, linealpha = .25 )
    end

    objectivepathplot!(x, x₊, f2)
    plot!( [x[1], x₊[1]], [x[2], x₊[2]];st = :line, linecolor = :orange, linealpha = .25)

    scatter!([x[1]],[x[2]],[fx[2]] ; markercolor = mcolors[1], markersize = MARKERSIZE)
    scatter!([x[1]],[x[2]],; markercolor = mcolors[1], markeralpha = .35, markersize = MARKERSIZE/2)
    scatter!([x₊[1]],[x₊[2]],[fx₊[2]] ; markercolor = mcolors[2], markersize = MARKERSIZE)
    scatter!([x₊[1]],[x₊[2]],; markercolor = mcolors[2], markeralpha = .35, markersize = MARKERSIZE/2)

    plot( p1, p2; size = (1000, 500) )
end
#gif( anim_objectives; fps =2 )
mov( anim_objectives, "objectives3d.mov", fps=2)
