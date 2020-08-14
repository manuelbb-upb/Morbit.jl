using Morbit
using Plots
using Plots.PlotMeasures
using LaTeXStrings

pyplot()

sites = collect( range(-.4,.4;length=5) )
sites_arrays = (x->[x]).(sites)

f1(x) = sin(5/(x + 1))

X = range(-.6, .6; length = 100)

kernel = :multiquadric

rbf = Morbit.RBFModel(
    kernel = kernel,
    shape_parameter = 10.0,
    training_sites = sites_arrays,
    training_values = (x->[x]).(f1.(sites)),
    polynomial_degree = -1
)

Morbit.train!(rbf)

p1 = plot(X, f1; label = L"f", linewidth = 1.5, linecolor = :blue, background_color_legend = :white, size = (480, 320), right_margin = 60px)
plot!(X, x -> Morbit.output( rbf, 1, [x]); label = L"m", linewidth = 1.5, linecolor = :red )
scatter!(sites, f1; label ="training data", markercolor = :orange, markersize = 6)
title!("Multiquadric Approximation.")
ylabel!(L"f,m")

p = twinx()
ylabel!(p,"basis functions")
for i = 1 : length( sites )
    global p, sites, rbf, kernel
    local φ = x -> Morbit.kernel( Val(kernel), abs(x - sites[i]), 10.0)
    plot!(p, X, rbf.rbf_coefficients[i] .* φ.(X); linealpha = .6, linestyle = :dash, label = :none)
end

display(p1)
cd(ENV["HOME"])
savefig("multiquadric_example.png")
