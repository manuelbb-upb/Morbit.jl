using Morbit
using LinearAlgebra: norm
using Plots
const m=Morbit
n = 1

t_lb = 0
t_w = 1

f1(x) = sum( (x .- 1.0).^2 )
f2(x) = exp( sum(x) )
f(x) = [f1(x); f2(x)];

t_sites = [[ .1 .+ .1 * rand(n) for i = 1 : 10 ]; [[ .9 ]] ]
t_vals = [[v] for v in f1.(t_sites)]

dist( s1 :: Vector{Float64}, s2:: Vector{Float64}) = norm( s1 - s2, 2)
dist( s1 :: Vector{Float64}, S2 :: Vector{Vector{Float64}}) = [ dist(s1, s2) for s2 ∈ S2 ]

X = range(-.1, 1.1; length=200)
Xa = [[x] for x in X]
e_vals = f1.(Xa)


r1 = m.RBFModel(
    training_sites = t_sites,
    training_values = t_vals,
    kernel = :multiquadric,
    shape_parameter = 10.0,
    polynomial_degree = 1,
    )
train!(r1)

sp = rand(Float64, length(t_sites))
min_dist_average = 0.0;
for s1_ind ∈ eachindex(t_sites)
    global min_dist_average
    s1 = t_sites[s1_ind]
    min_dist = minimum( dist(s1, t_sites[ [1 : s1_ind - 1; s1_ind + 1 : end ]]) )
    min_dist_average += min_dist
    sp[s1_ind] = 1/ (20*min_dist);
end
min_dist_average /= length(t_sites)

sp = max.(sp,1/(20*min_dist_average))

r2 = m.RBFModel(
    training_sites = t_sites,
    training_values = t_vals,
    kernel = :multiquadric,
    shape_parameter = sp,
    polynomial_degree = 1,
)

train!(r2)

r1_vals = r1.(Xa)
r2_vals = r2.(Xa)

plot(X, vcat(e_vals...))
plot!(X, vcat(r1_vals...))
plot!(X, vcat(r2_vals...))
scatter!(vcat( t_sites...), vcat(t_vals...) )
