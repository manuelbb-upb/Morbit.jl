using Morbit

f₁ = x -> sum( (x .- 1).^2 )
f₂ = x -> sum( (x .+ 1).^2 )
∇f₁ = x -> 2 .* ( x .- 1 )
∇f₂ = x -> 2 .* ( x .+ 1 )

mop = MixedMOP(2);  # problem with 2 variables
add_objective!(mop, f₁, ∇f₁ )
add_objective!(mop, f₂, ∇f₂ )

# starting point
x₀ = [ -π ;  2.71828 ]

x, fx, id = optimize( mop, x₀ ) 
x
db = Morbit.db(id);

# let's retrieve the iteration sites for plotting:
# wrapping in `Tuple` for easy plotting
it_sites = Tuple.( Morbit.get_iterate_sites(db) );

using AbstractPlotting, CairoMakie

# %% Pareto Set ≙ line from (-1,-1) to (1,1)
fig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,
    figure = (resolution = (600, 600),) )
ax.title[] = "Pareto Set and Iterates."
ax.xgridvisible[] = false # hide
ax.ygridvisible[] = false # hide
# Plot the iteration sites:
lines!(it_sites)
scatter!(it_sites; 
    color = LinRange(0, 1, length(it_sites)), 
    colormap = :winter
)

# Plot function contours 
Y = X = LinRange(-4, 4, 100)
Z₁ = [ f₁([x;y]) for x ∈ X, y ∈ X ]
Z₂ = [ f₂([x;y]) for x ∈ X, y ∈ X ]
levels = [ i.^2 for i = LinRange(.1, 6, 6) ]
contour!(X,Y,Z₁; colormap = :greens, levels = levels, linewidth = .5 )
contour!(X,Y,Z₂; colormap = :heat, levels = levels, linewidth = .5 )
fig

#%%
using Morbit
ac = AlgoConfig( max_iter = 10 )

mop_rbf = MixedMOP()

# define the RBF surrogates
rbf_cfg = RbfConfig( 
    kernel = :multiquadric, 
    shape_parameter = "20/Δ" 
)
# add objective functions to `mop_rbf`
add_objective!(mop_rbf, f₁, rbf_cfg )
add_objective!(mop_rbf, f₂, rbf_cfg )

# a array of well spread points in [-4,4]²
X =[
 [-4.0, -4.0],
 [3.727327839472812, 3.8615291196035457],
 [3.804712690019901, -3.9610212058521235],
 [-0.14512898384374573, -0.005775390168885508],
 [-3.775315499879552, 3.8150054323309064],
 [1.714228746087743, 1.8435786475209621],
 [-1.9603720505875337, -2.0123206708499275],
 [3.9953803225349187, -0.47734576293976794],
 [-3.9944468955728745, 0.49857343385493635],
 [-1.0455585089057458, 2.735699160002545]
]

fig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,
    figure = (resolution = (600, 600), ),
    axis = (title="Different Starting Points",), 
)

start_fin_points = Dict();
db₀ = nothing 
for x₀ ∈ X
    x_fin, fx_fin, id = optimize( mop_rbf, x₀; algo_config = ac, populated_db = db₀ )
    start_fin_points[x₀] = x_fin
    db₀ = Morbit.merge( db₀, Morbit.db(id) )
end


for (k,v) in start_fin_points
    lines!( [ Tuple(k), Tuple(v) ] )
end

scatter!( Tuple.(keys(start_fin_points)); 
    color = :green
)
scatter!( Tuple.(values(start_fin_points)); 
    color = :lightblue
)
