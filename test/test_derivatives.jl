using Morbit
using Plots
using ForwardDiff: gradient
f1(x) = x^2;
f2(x) = sin(x);
f(x) = [f1(x), f2(x)];

sites = range( -2, 2, length = 10) ;
vals = f.(sites);

m = RBFModel(
    training_sites = [[s] for s in sites],
    training_values = vals,
    kernel = :multiquadric,
    shape_parameter = 1e-1,
    polynomial_degree = 1,
);
train!(m);

x = range(-4, 4; length = 100);
y1 = f1.(x);
y2 = f2.(x);

m1(x) = output(m, 1, x);
m2(x) = output(m, 2, x);

z1 = m1.(x);
z2 = m2.(x);

# derivatives
g1_fwd = x -> gradient(m1, [x])[end]    # ForwardDiff derivative
g1_m = x -> grad(m, 1, x)[end]    # grad returns an array, n == 1 ⇒ extract single entry as derivative
g2_fwd = x -> gradient(m2, [x])[end]
g2_m = x -> grad(m, 2, x)[end]
gf1 = x -> gradient( X -> f1.(X)[end], [x])[end]
gf2 = x -> gradient( X -> f2.(X)[end], [x])[end]

z1_fwd′ = g1_fwd.(x)
z1′ = g1_m.(x)
z2_fwd′ = g2_fwd.(x)
z2′ = g2_m.(x)
y1′ = gf1.(x);
y2′ = gf2.(x);

Jm = Morbit.Jacobian(m, x[1])

@assert isapprox(z1_fwd′[1], z1′[1])
@assert isapprox(z1′[1], Jm[1, 1])

# Plotting
l = @layout grid(2, 2)
p = plot(layout = l)

plot!(x, y1, label = "f1")
plot!(x, z1, label = "m1")
scatter!(sites, [v[1] for v ∈ vals], markersize = 2, label = "")

plot!(x, z1_fwd′, label = "∇m1_fwd", subplot = 2, legend = :best)
plot!(x, z1′, label = "∇m1", subplot = 2)
plot!(x, y1′, label = "∇f1", subplot = 2)

plot!(x, y2, label = "f2", subplot = 3)
plot!(x, z2, label = "m2", subplot = 3)
scatter!(sites, [v[2] for v ∈ vals], markersize = 2, label = "", subplot = 3)

plot!(x, z2_fwd′, label = "∇m2_fwd", subplot = 4, legend = :best)
plot!(x, z2′, label = "∇m2", subplot = 4)
plot!(x, y2′, label = "∇f2", subplot = 4)
