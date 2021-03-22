# Two Parabolas

The “two parabolas” problem in two dimensions reads as
```math
    \min_{𝐱 ∈ X } 
    \begin{bmatrix} f₁(\mathbf{x}) \\ f₂(\mathbf{x}) \end{bmatrix} = 
    \min_{\mathbf{x} ∈ X}
    \begin{bmatrix}
    (x₁ - 1)² + (x₂ - 1)² \\
    (x₁ + 1)² + (x₂ + 1)²
    \end{bmatrix}.
```
It is unconstrained if the feasible set is ``X = ℝ^2``.
The individual minima ``[1,1]`` and ``[-1,-1]`` are such that (in the unconstrained case)
the global Pareto Set is 
```math
\mathcal{P}_{S} = \{ \mathbf{x} ∈ ℝ^2 : x₁ = x₂, \, -1 \le x₁, x₂ \le 1  \}.
```

## Solve using Exact Functions

The gradients are easily calculated as 
```math
\nabla f_1 (\mathbf x) = 2 \begin{bmatrix}
x_1 -1 \\ x_2 - 1 \end{bmatrix}, \;
\nabla f_2 (\mathbf x) = 2 \begin{bmatrix}
x_1 +1 \\ x_2 + 1 \end{bmatrix}, \;
```

We can provide them to the solver to find a critical point:

```@example 1
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

ac = AlgoConfig(max_iter = 20)
x, fx, id = optimize( mop, x₀; algo_config = ac ) 
x
```

Hopefully, `x` is critical.

### Plotting Iteration Sites

We can retrieve iteration data from `id` and the database `Morbit.db(id)`.
```@example 1
db = Morbit.db(id);

# let's retrieve the iteration sites for plotting:
# (conversion to Tuples for easy plotting)
it_sites = Tuple.(Morbit.get_iterate_sites(db))
nothing # hide
```

Let's plot the Pareto Set and the iteration sites:
```@example 1
using AbstractPlotting, CairoMakie

# Pareto Set ≙ line from (-1,-1) to (1,1)
fig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,
    figure = (resolution=(600, 650),),
    axis = (aspect = 1, title = "Pareto Set and Iterates.") )
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
save("two_parabolas_cheap.png", fig) # hide
nothing # hide
```
![Two Parabolas (cheap): Pareto Set and Iterates](two_parabolas_cheap.png)

## Solving using RBF Surrogates

Suppose now that we do not have access to the objective gradients and that the objectives also take some time to evaluate.
In this situation, we could try to model them using surrogate models.
To use radial basis function models, pass an `RbfConfig` when specifying the objective:
```@example 1
mop_rbf = MixedMOP()

# define the RBF surrogates
rbf_cfg = RbfConfig( 
    kernel = :multiquadric, 
    shape_parameter = "20/Δ" 
)
# add objective functions to `mop_rbf`
add_objective!(mop_rbf, f₁, rbf_cfg )
add_objective!(mop_rbf, f₂, rbf_cfg )

# only perform 10 iterations
x, fx, id = optimize( mop, x₀; algo_config = ac ) 
x
```
```@setup 1
it_sites_rbf = Tuple.(Morbit.get_iterate_sites(Morbit.db(id)))
lines!(it_sites)
scatter!(it_sites; color = :orange)
save("two_parabolas_cheap_and_rbf.png", fig)
```
The iteration site are the orange circles:
![Two Parabolas (cheap and RBF): Pareto Set and Iterates](two_parabolas_cheap_and_rbf.png)

## Different Starting Points and Recycling Data 

The method could converge to different points depending on the starting point. 
We can pass the evaluation data from previous runs to facilitate the construction of surrogate models:
```@setup 2
f₁ = x -> sum( (x .- 1).^2 )
f₂ = x -> sum( (x .+ 1).^2 )

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

using AbstractPlotting, CairoMakie
fig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,
    figure = (resolution = (600, 600), ),
    axis = (title="Different Starting Points",), 
)
```

```@example 2
# Suppose, `X` is a list of different points in ℝ²

# dict to hold starting and end points
start_fin_points = Dict();

# perform several runs:
db₀ = nothing 
for x₀ ∈ X
    global db₀
    x_fin, fx_fin, id = optimize( mop_rbf, x₀; algo_config = ac, populated_db = db₀ )
    start_fin_points[x₀] = x_fin
    db₀ = Morbit.merge( db₀, Morbit.db(id) )
end

# Plot
for (k,v) in start_fin_points
    lines!( [ Tuple(k), Tuple(v) ] )
end

scatter!( Tuple.(keys(start_fin_points)); 
    color = :green
)
scatter!( Tuple.(values(start_fin_points)); 
    color = :lightblue
)
save("two_parabolas_different_starting_points.png", fig) # hide 
nothing # hide
```

In the plot, the green points show the starting points and the lightblue circles show the final iterates:
![Two Parabolas - Different Starting Points](two_parabolas_different_starting_points.png)