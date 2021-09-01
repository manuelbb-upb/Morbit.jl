# # Two Parabolas
# 
# The “two parabolas” problem in two dimensions reads as
# ```math
#     \min_{𝐱 ∈ X } 
#     \begin{bmatrix} f₁(\mathbf{x}) \\ f₂(\mathbf{x}) \end{bmatrix} = 
#     \min_{\mathbf{x} ∈ X}
#     \begin{bmatrix}
#     (x₁ - 1)² + (x₂ - 1)² \\
#     (x₁ + 1)² + (x₂ + 1)²
#     \end{bmatrix}.
# ```
# It is unconstrained if the feasible set is ``X = ℝ^2``.
# The individual minima ``[1,1]`` and ``[-1,-1]`` are such that (in the unconstrained case)
# the global Pareto Set is 
# ```math
# \mathcal{P}_{S} = \{ \mathbf{x} ∈ ℝ^2 : x₁ = x₂, \, -1 \le x₁, x₂ \le 1  \}.
# ```
# 
# ## Solve using Exact Functions
# 
# The gradients are easily calculated as 
# ```math
# \nabla f_1 (\mathbf x) = 2 \begin{bmatrix}
# x_1 -1 \\ x_2 - 1 \end{bmatrix}, \;
# \nabla f_2 (\mathbf x) = 2 \begin{bmatrix}
# x_1 +1 \\ x_2 + 1 \end{bmatrix}, \;
# ```
# 
# We can provide them to the solver to find a critical point:

using Pkg #src
Pkg.activate(@__DIR__) #src
using Test #src

using Morbit
Morbit.print_all_logs() #src

f₁ = x -> sum( (x .- 1).^2 )
f₂ = x -> sum( (x .+ 1).^2 )
∇f₁ = x -> 2 .* ( x .- 1 )
∇f₂ = x -> 2 .* ( x .+ 1 )

mop = MixedMOP(2);  # problem with 2 variables
add_objective!(mop, f₁, ∇f₁ )
add_objective!(mop, f₂, ∇f₂ )

#~ starting point
x₀ = [ -π ;  2.71828 ]

#~ set maximum number of iterations 
ac = AlgoConfig( max_iter = 20)
#~ `optimize` will return parameter and result vectors as well 
#~ as an return code and the evaluation database:
x, fx, ret_code, db = optimize( mop, x₀; algo_config = ac );
x

# Hopefully, `x` is critical, i.e., `x[1] ≈ x[2]`.
@test x[1] ≈ x[2] atol = .1 #src

# !!! note
#     To print more information on what the solver is doing, you can use the `Logging` module: 
#     ```julia 
#     import Logging: global_logger, ConsoleLogger
#     global_logger( ConsoleLogger( stderr, Morbit.loglevel4; 
#         meta_formatter = Morbit.morbit_formatter ) )
#     ```
#     `loglevel4` is the most detailed and `loglevel1` is least detailed. 
#     `Morbit.print_all_logs()` is a convenient shorthand.
 
#%% #src
# ### Plotting Iteration Sites 
# Let's retrieve the iteration sites.
# We convert to Tuples for easier plotting.
iteration_indices = [ iter_.x_index for iter_ in db.iter_info]
it_sites = Tuple.(Morbit.get_site.(db, iteration_indices))

# For Plotting we use CairoMakie
using Makie, CairoMakie

#~ Pareto Set ≙ line from (-1,-1) to (1,1)
fig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,
    figure = (resolution = (600, 600),) )

#~ Plot the iteration sites:
lines!(it_sites)
scatter!(it_sites; 
    color = LinRange(0, 1, length(it_sites)), 
    colormap = :winter
)

#~ Plot function contours 
Y = X = LinRange(-4, 4, 100)
Z₁ = [ f₁([x;y]) for x ∈ X, y ∈ X ]
Z₂ = [ f₂([x;y]) for x ∈ X, y ∈ X ]
levels = [ i.^2 for i = LinRange(.1, 6, 6) ]
contour!(X,Y,Z₁; colormap = :greens, levels = levels, linewidth = .5 )
contour!(X,Y,Z₂; colormap = :heat, levels = levels, linewidth = .5 )

#~ Show the plot:
ax.title[] = "Pareto Set and Iterates."
ax.xgridvisible[] = false 
ax.ygridvisible[] = false

fig

#%% #src

# ## Solving using RBF Surrogates
# 
# Suppose now that we do not have access to the objective gradients and that the objectives 
# also take some time to evaluate.
# In this situation, we could try to model them using surrogate models.
# To use radial basis function models, pass an `RbfConfig` when specifying the objective:
# 
mop_rbf = MixedMOP()

#~ Define the RBF surrogates
rbf_cfg = RbfConfig( 
    kernel = :inv_multiquadric 
)
#~ Add objective functions to `mop_rbf`
add_objective!(mop_rbf, f₁, rbf_cfg )
add_objective!(mop_rbf, f₂, rbf_cfg )

#~ only perform 10 iterations
ac = AlgoConfig( max_iter = 10 )
x, fx, _, db = optimize( mop_rbf, x₀; algo_config = ac ) 
x

#src Setup and save plot for docs.
iteration_indices_rbf = [ iter_.x_index for iter_ in db.iter_info]
it_sites_rbf = Tuple.(Morbit.get_site.(db, iteration_indices_rbf))
lines!(it_sites_rbf) #hide
scatter!(it_sites_rbf; color = :orange) #hide
nothing #hide 

# The iteration sites are the orange circles:
fig #hide

#%% #src

# ## Different Starting Points and Recycling Data 
# 
# The method could converge to different points depending on the starting point. 
# We can pass the evaluation data from previous runs to facilitate the construction of surrogate models:
 
#src Setup the problem anew, to ensure fresh start
ac = AlgoConfig( #hide
    max_iter = 10 #hide
    ); #hide
mop_rbf = MixedMOP(); #hide
#~ define the RBF surogates #hide
rbf_cfg = RbfConfig(  #hide
    kernel = :inv_multiquadric, #hide
); #hide
#~ add objective functions to `mop_rbf` #hide
add_objective!(mop_rbf, f₁, rbf_cfg ); #hide
add_objective!(mop_rbf, f₂, rbf_cfg ); #hide

#~ an array of well spread points in [-4,4]² #hide
X =[ #hide
 [-4.0, -4.0], #hide
 [3.727327839472812, 3.8615291196035457], #hide
 [3.804712690019901, -3.9610212058521235], #hide
 [-0.14512898384374573, -0.005775390168885508], #hide
 [-3.775315499879552, 3.8150054323309064], #hide
 [1.714228746087743, 1.8435786475209621], #hide
 [-1.9603720505875337, -2.0123206708499275], #hide
 [3.9953803225349187, -0.47734576293976794], #hide
 [-3.9944468955728745, 0.49857343385493635], #hide
 [-1.0455585089057458, 2.735699160002545] #hide
]; #hide

# Suppose, `X` is a list of different points in ℝ².

#src This is the code block visible in the docs:

#~ A dict to associate starting and end points:
start_fin_points = Dict();

#~ perform several runs:
db₀ = nothing # initial database can be `nothing`
for x₀ ∈ X
    global db₀, start_fin_points
    x_fin, fx_fin, _, db₀ = optimize( mop_rbf, x₀; algo_config = ac, populated_db = db₀ )
    #~ add points to dict
    start_fin_points[x₀] = x_fin
end

# Plotting: 

fig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,
    figure = (resolution = (600, 600), ),
    axis = (title="Different Starting Points",), 
)

for (k,v) in start_fin_points
    lines!( [ Tuple(k), Tuple(v) ]; color = :lightgray )
end

scatter!( Tuple.(keys(start_fin_points)); 
    color = :green
)
scatter!( Tuple.(values(start_fin_points)); 
    color = :lightblue
)

fig #hide

# In the plot, the green points show the starting points and the lightblue circles show the final iterates: