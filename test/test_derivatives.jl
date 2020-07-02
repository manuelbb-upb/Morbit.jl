using Morbit
using Plots
using ForwardDiff: gradient
f1( x ) = x^2;
f2( x ) = exp(x);
f(x) = [f1(x), f2(x)];

sites = [rand() for i = 1:5];
vals = f.(sites);

m = RBFModel(training_sites = [[s] for s in sites], training_values = vals, kernel = "thin_plate_spline");
train!(m);

x = range( -4, 4; length = 100 );
y1 = f1.(x);
y2 = f2.(x);

m1 = m.output_handles[1];
m2 = m.output_handles[2];

z1 = m1.(x);
z2 = m1.(x);

g1_true(x) = gradient( m1, [x] )[1]
g2_true(x) = gradient( m2, [x] )[1]
g1_m = m.gradient_handles[1];
g2_m = m.gradient_handles[2];

l = @layout grid(2,2)
p = plot(x, y1, label = "data", layout = l)
plot!(x, z1, label = "approx")

plot!(x, g1_true.(x), label = "dm1_fwd", subplot = 2, legend = :bottomleft)
plot!(x, g1_m.(x), label = "dm1", subplot=2)
