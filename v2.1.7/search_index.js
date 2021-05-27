var documenterSearchIndex = {"docs":
[{"location":"example_two_parabolas/#Two-Parabolas","page":"Two Parabolas","title":"Two Parabolas","text":"","category":"section"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"The “two parabolas” problem in two dimensions reads as","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"    min_𝐱  X  \n    beginbmatrix f₁(mathbfx)  f₂(mathbfx) endbmatrix = \n    min_mathbfx  X\n    beginbmatrix\n    (x₁ - 1)² + (x₂ - 1)² \n    (x₁ + 1)² + (x₂ + 1)²\n    endbmatrix","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"It is unconstrained if the feasible set is X = ℝ^2. The individual minima 11 and -1-1 are such that (in the unconstrained case) the global Pareto Set is ","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"mathcalP_S =  mathbfx  ℝ^2  x₁ = x₂  -1 le x₁ x₂ le 1  ","category":"page"},{"location":"example_two_parabolas/#Solve-using-Exact-Functions","page":"Two Parabolas","title":"Solve using Exact Functions","text":"","category":"section"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"The gradients are easily calculated as ","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"nabla f_1 (mathbf x) = 2 beginbmatrix\nx_1 -1  x_2 - 1 endbmatrix \nnabla f_2 (mathbf x) = 2 beginbmatrix\nx_1 +1  x_2 + 1 endbmatrix ","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"We can provide them to the solver to find a critical point:","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"using Morbit\n\nf₁ = x -> sum( (x .- 1).^2 )\nf₂ = x -> sum( (x .+ 1).^2 )\n∇f₁ = x -> 2 .* ( x .- 1 )\n∇f₂ = x -> 2 .* ( x .+ 1 )\n\nmop = MixedMOP(2);  # problem with 2 variables\nadd_objective!(mop, f₁, ∇f₁ )\nadd_objective!(mop, f₂, ∇f₂ )\n\n# starting point\nx₀ = [ -π ;  2.71828 ]\n\nac = AlgoConfig(max_iter = 20)\nx, fx, id = optimize( mop, x₀; algo_config = ac ) \nx","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"Hopefully, x is critical.","category":"page"},{"location":"example_two_parabolas/#Plotting-Iteration-Sites","page":"Two Parabolas","title":"Plotting Iteration Sites","text":"","category":"section"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"We can retrieve iteration data from id and the database Morbit.db(id).","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"db = Morbit.db(id);\n\n# let's retrieve the iteration sites for plotting:\n# (conversion to Tuples for easy plotting)\nit_sites = Tuple.(Morbit.get_iterate_sites(db))\nnothing # hide","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"Let's plot the Pareto Set and the iteration sites:","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"using AbstractPlotting, CairoMakie\n\n# Pareto Set ≙ line from (-1,-1) to (1,1)\nfig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,\n    figure = (resolution=(600, 650),),\n    axis = (aspect = 1, title = \"Pareto Set and Iterates.\") )\nax.xgridvisible[] = false # hide\nax.ygridvisible[] = false # hide\n# Plot the iteration sites:\nlines!(it_sites)\nscatter!(it_sites; \n    color = LinRange(0, 1, length(it_sites)), \n    colormap = :winter\n)\n\n# Plot function contours \nY = X = LinRange(-4, 4, 100)\nZ₁ = [ f₁([x;y]) for x ∈ X, y ∈ X ]\nZ₂ = [ f₂([x;y]) for x ∈ X, y ∈ X ]\nlevels = [ i.^2 for i = LinRange(.1, 6, 6) ]\ncontour!(X,Y,Z₁; colormap = :greens, levels = levels, linewidth = .5 )\ncontour!(X,Y,Z₂; colormap = :heat, levels = levels, linewidth = .5 )\nsave(\"two_parabolas_cheap.png\", fig) # hide\nnothing # hide","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"(Image: Two Parabolas (cheap): Pareto Set and Iterates)","category":"page"},{"location":"example_two_parabolas/#Solving-using-RBF-Surrogates","page":"Two Parabolas","title":"Solving using RBF Surrogates","text":"","category":"section"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"Suppose now that we do not have access to the objective gradients and that the objectives also take some time to evaluate. In this situation, we could try to model them using surrogate models. To use radial basis function models, pass an RbfConfig when specifying the objective:","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"mop_rbf = MixedMOP()\n\n# define the RBF surrogates\nrbf_cfg = RbfConfig( \n    kernel = :multiquadric, \n    shape_parameter = \"20/Δ\" \n)\n# add objective functions to `mop_rbf`\nadd_objective!(mop_rbf, f₁, rbf_cfg )\nadd_objective!(mop_rbf, f₂, rbf_cfg )\n\n# only perform 10 iterations\nx, fx, id = optimize( mop, x₀; algo_config = ac ) \nx","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"it_sites_rbf = Tuple.(Morbit.get_iterate_sites(Morbit.db(id)))\nlines!(it_sites)\nscatter!(it_sites; color = :orange)\nsave(\"two_parabolas_cheap_and_rbf.png\", fig)","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"The iteration site are the orange circles: (Image: Two Parabolas (cheap and RBF): Pareto Set and Iterates)","category":"page"},{"location":"example_two_parabolas/#Different-Starting-Points-and-Recycling-Data","page":"Two Parabolas","title":"Different Starting Points and Recycling Data","text":"","category":"section"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"The method could converge to different points depending on the starting point.  We can pass the evaluation data from previous runs to facilitate the construction of surrogate models:","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"f₁ = x -> sum( (x .- 1).^2 )\nf₂ = x -> sum( (x .+ 1).^2 )\n\nusing Morbit\nac = AlgoConfig( max_iter = 10 )\n\nmop_rbf = MixedMOP()\n\n# define the RBF surrogates\nrbf_cfg = RbfConfig( \n    kernel = :multiquadric, \n    shape_parameter = \"20/Δ\" \n)\n# add objective functions to `mop_rbf`\nadd_objective!(mop_rbf, f₁, rbf_cfg )\nadd_objective!(mop_rbf, f₂, rbf_cfg )\n\n# a array of well spread points in [-4,4]²\nX =[\n [-4.0, -4.0],\n [3.727327839472812, 3.8615291196035457],\n [3.804712690019901, -3.9610212058521235],\n [-0.14512898384374573, -0.005775390168885508],\n [-3.775315499879552, 3.8150054323309064],\n [1.714228746087743, 1.8435786475209621],\n [-1.9603720505875337, -2.0123206708499275],\n [3.9953803225349187, -0.47734576293976794],\n [-3.9944468955728745, 0.49857343385493635],\n [-1.0455585089057458, 2.735699160002545]\n]\n\nusing AbstractPlotting, CairoMakie\nfig, ax, _ = lines( [(-1,-1),(1,1)]; color = :blue, linewidth = 2,\n    figure = (resolution = (600, 600), ),\n    axis = (title=\"Different Starting Points\",), \n)","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"# Suppose, `X` is a list of different points in ℝ²\n\n# dict to hold starting and end points\nstart_fin_points = Dict();\n\n# perform several runs:\ndb₀ = nothing \nfor x₀ ∈ X\n    global db₀\n    x_fin, fx_fin, id = optimize( mop_rbf, x₀; algo_config = ac, populated_db = db₀ )\n    start_fin_points[x₀] = x_fin\n    db₀ = Morbit.merge( db₀, Morbit.db(id) )\nend\n\n# Plot\nfor (k,v) in start_fin_points\n    lines!( [ Tuple(k), Tuple(v) ] )\nend\n\nscatter!( Tuple.(keys(start_fin_points)); \n    color = :green\n)\nscatter!( Tuple.(values(start_fin_points)); \n    color = :lightblue\n)\nsave(\"two_parabolas_different_starting_points.png\", fig) # hide \nnothing # hide","category":"page"},{"location":"example_two_parabolas/","page":"Two Parabolas","title":"Two Parabolas","text":"In the plot, the green points show the starting points and the lightblue circles show the final iterates: (Image: Two Parabolas - Different Starting Points)","category":"page"},{"location":"example_zdt/#ZDT3-Problem","page":"ZDT3","title":"ZDT3 Problem","text":"","category":"section"},{"location":"example_zdt/#Setup","page":"ZDT3","title":"Setup","text":"","category":"section"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"Install the test problem suite:","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"using Pkg \nPkg.activate(tempname())\nPkg.develop(url=\"https://github.com/manuelbb-upb/MultiObjectiveProblems.jl\")\nusing MultiObjectiveProblems","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"using Pkg \nPkg.activate(tempname())\nPkg.develop(url=\"https://github.com/manuelbb-upb/MultiObjectiveProblems.jl\")\nusing MultiObjectiveProblems","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"Import other dependencies:","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"using AbstractPlotting, CairoMakie\nusing Morbit","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"Retrieve test problem and define a MixedMOP","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"test_problem = ZDT3(2);\nbox = constraints(test_problem);\n\nobjectives = get_objectives(test_problem)\nx₀ = get_random_point(test_problem)\n\nmop = MixedMOP( box.lb, box.ub );\nobjf_cfg = ExactConfig()\nfor objf ∈ objectives\n    add_objective!(mop, objf, objf_cfg)\nend","category":"page"},{"location":"example_zdt/#Run","page":"ZDT3","title":"Run","text":"","category":"section"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"Run optimization and plot:","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"I = get_ideal_point(test_problem)\nac = AlgoConfig(; descent_method = :ps, reference_point = I )\n\nx, fx, id = optimize( mop, x₀; algo_config = ac);\n\npset = get_pareto_set(test_problem)\nPSx,PSy = get_scatter_points(pset, 100)\n\n# scatter Pareto set points in grey\nfig, ax, _ = scatter( PSx, PSy;\n    figure = (resolution = (600, 650),), \n)\n\n# set axis limits to whole feasible set\nxlims!(ax, (box.lb[1] .- .2, box.ub[1] .+ .2) ) \nylims!(ax, (box.lb[2] .- .2, box.ub[2] .+ .2) ) \n\n# final iterate in red\nscatter!(Tuple(x); color = :red)\nsave(\"example_zdt_scatter.png\", fig) # hide\nnothing # hide","category":"page"},{"location":"example_zdt/","page":"ZDT3","title":"ZDT3","text":"(Image: )","category":"page"},{"location":"","page":"Home","title":"Home","text":"CurrentModule = Morbit","category":"page"},{"location":"#Morbit","page":"Home","title":"Morbit","text":"","category":"section"},{"location":"","page":"Home","title":"Home","text":"The package Morbit.jl provides a local derivative-free solver for multiobjective optimization problems with possibly expensive objectives. It is meant to find a single Pareto-critical point, not a good covering of the global Pareto Set.","category":"page"},{"location":"","page":"Home","title":"Home","text":"“Morbit” stands for Multiobjective Optimization by Radial Basis Function Interpolation in Trust-regions.  The name was chosen so as to pay honors to the single objective algorithm ORBIT by Wild et. al.   There is a preprint in the arXiv that explains what is going on inside. It has been submitted to the MCA journal.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This was my first project using Julia and there have been many messy rewrites. Nonetheless, the solver should now work sufficiently well to tackle most problems.","category":"page"},{"location":"","page":"Home","title":"Home","text":"This project was founded by the European Region Development Fund.","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img src=\"https://www.efre.nrw.de/fileadmin/Logos/EU-Fo__rderhinweis__EFRE_/EFRE_Foerderhinweis_englisch_farbig.jpg\" width=\"45%\"/>","category":"page"},{"location":"","page":"Home","title":"Home","text":"<img src=\"https://www.efre.nrw.de/fileadmin/Logos/Programm_EFRE.NRW/Ziel2NRW_RGB_1809_jpg.jpg\" width=\"45%\"/>","category":"page"},{"location":"","page":"Home","title":"Home","text":"","category":"page"},{"location":"","page":"Home","title":"Home","text":"Modules = [Morbit]","category":"page"},{"location":"#Morbit.RbfConfig","page":"Home","title":"Morbit.RbfConfig","text":"RbfConfig(; kwarg1 = val1, … )\n\nConfiguration type for local RBF surrogate models.\n\nTo choose a kernel, use the kwarg kernel and a value of either  :cubic (default), :multiquadric, :exp or :thin_plate_spline. The kwarg shape_parameter takes a constant number or a string  that defines a calculation on Δ, e.g, \"Δ/10\".\n\nTo see other configuration parameters use fieldnames(Morbit.RbfConfig). They have individual docstrings attached.\n\n\n\n\n\n","category":"type"},{"location":"#Morbit.TransformerFn-Tuple{Vector{R} where R<:Real}","page":"Home","title":"Morbit.TransformerFn","text":"Unscale the point x̂ from internal to original domain.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._add!-Tuple{Morbit.AbstractMOP, Morbit.AbstractObjective, Union{Nothing, Vector{Int64}}}","page":"Home","title":"Morbit._add!","text":"Add an objective function to MOP with specified output indices.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._backtrack-Tuple{Vector{R} where R<:Real, Vector{R} where R<:Real, Real, Real, Morbit.SurrogateContainer, Bool}","page":"Home","title":"Morbit._backtrack","text":"Perform a backtracking loop starting at x with an initial step of step_size .* dir and return trial point x₊, the surrogate value-vector m_x₊ and the final step s = x₊ .- x.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._del!-Tuple{Morbit.AbstractMOP, Morbit.AbstractObjective}","page":"Home","title":"Morbit._del!","text":"Remove an objective function from MOP.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._get_shape_param-Tuple{RbfConfig, Morbit.AbstractIterData}","page":"Home","title":"Morbit._get_shape_param","text":"Get real-valued shape parameter for RBF model from current iter data. cfg allows for a string expression which would be evaluated here.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._init_model-Tuple{ExactConfig, Morbit.AbstractObjective, Morbit.AbstractMOP, Morbit.AbstractIterData, Morbit.AbstractConfig}","page":"Home","title":"Morbit._init_model","text":"Return an ExactModel build from a VectorObjectiveFunction objf.  Model is the same inside and outside of criticality round.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._init_model-Tuple{TaylorConfig, Morbit.AbstractObjective, Morbit.AbstractMOP, Morbit.AbstractIterData, Morbit.AbstractConfig}","page":"Home","title":"Morbit._init_model","text":"Return a TaylorModel build from a VectorObjectiveFunction objf.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._intersect_bounds-NTuple{4, Vector{R} where R<:Real}","page":"Home","title":"Morbit._intersect_bounds","text":"Return smallest positive and biggest negative and σ₊ and σ₋ so that x .+ σ± .* d stays within bounds.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._local_bounds-Tuple{Vector{R} where R<:Real, Union{Real, Vector{R} where R<:Real}, Vector{R} where R<:Real, Vector{R} where R<:Real}","page":"Home","title":"Morbit._local_bounds","text":"Return lower and upper bound vectors combining global and trust region constraints.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._multifactorial-Tuple{Vector{Int64}}","page":"Home","title":"Morbit._multifactorial","text":"Factorial of a multinomial.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._objf_index-Tuple{Morbit.AbstractObjective, Morbit.AbstractMOP}","page":"Home","title":"Morbit._objf_index","text":"Position of objf in list_of_objectives(mop).\n\n\n\n\n\n","category":"method"},{"location":"#Morbit._wrap_func-Tuple{Type{var\"#s19\"} where var\"#s19\"<:Morbit.AbstractObjective, Function, Morbit.SurrogateConfig, Int64, Int64}","page":"Home","title":"Morbit._wrap_func","text":"A general constructor.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.add_objective!","page":"Home","title":"Morbit.add_objective!","text":"add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, type :: Symbol = :expensive, n_out :: Int64 = 1, can_batch :: Bool = false )\n\nAdd scalar-valued objective function func to mop structure. func must take an RVec as its (first) argument, i.e. represent a function f ℝ^n  ℝ. type must either be :expensive or :cheap to determine whether the function is replaced by a surrogate model or not.\n\nIf type is :cheap and func takes 1 argument only then its gradient is calculated by ForwardDiff. A cheap function func with custom gradient function grad (representing f  ℝ^n  ℝ^n) is added by\n\nadd_objective!(mop, func, grad)\n\nThe optional argument n_out allows for the specification of vector-valued objective functions. This is mainly meant to be used for expensive functions that are in some sense inter-dependent.\n\nThe flag can_batch defaults to false so that the objective function is simply looped over a bunch of arguments if required. If can_batch == true then the objective function must be able to return an array of results when provided an array of input vectors (whilst still returning a single result, not a singleton array containing the result, for a single input vector).\n\nExamples\n\n# Define 2 scalar objective functions and a MOP ℝ^2 → ℝ^2\n\nf1(x) =  x[1]^2 + x[2]\n\nf2(x) = exp(sum(x))\n∇f2(x) = exp(sum(x)) .* ones(2);\n\nmop = MixedMOP()\nadd_objective!(mop, f1, :cheap)     # gradient will be calculated using ForwardDiff\nadd_objective!(mop, f2, ∇f2 )       # gradient is provided\n\n\n\n\n\n","category":"function"},{"location":"#Morbit.add_objective!-Tuple{MixedMOP, Function, Function}","page":"Home","title":"Morbit.add_objective!","text":"add_objective!( mop :: MixedMOP, func :: T where{T <: Function}, grad :: T where{T <: Function})\n\nAdd scalar-valued objective function func and its vector-valued gradient grad to mop struture.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.add_objective!-Tuple{MixedMOP, Function, Morbit.SurrogateConfig}","page":"Home","title":"Morbit.add_objective!","text":"Add a scalar objective to mop::MixedMOP modelled according to model_config.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.add_vector_objective!-Tuple{MixedMOP, Function, Morbit.SurrogateConfig}","page":"Home","title":"Morbit.add_vector_objective!","text":"Add a vector objective to mop::MixedMOP modelled according to model_config.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.combine-Tuple{Morbit.AbstractObjective, Morbit.AbstractObjective}","page":"Home","title":"Morbit.combine","text":"Combine two objectives. Only needed if combinable can return true.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.combine-Union{Tuple{T}, Tuple{F}, Tuple{F, T}} where {F<:Function, T<:Function}","page":"Home","title":"Morbit.combine","text":"Get a new function function handle stacking the output of func1 and func2.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.delete_surrogate!-Tuple{Morbit.SurrogateContainer, Int64}","page":"Home","title":"Morbit.delete_surrogate!","text":"Delete surrogate wrapper at position si from list sc.surrogates.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_all_objectives-Tuple{Morbit.AbstractMOP, Vector{R} where R<:Real}","page":"Home","title":"Morbit.eval_all_objectives","text":"(Internally) Evaluate all objectives at site x̂::RVec. Objective order might differ from order in which they were added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_and_sort_objectives-Tuple{Morbit.AbstractMOP, Vector{R} where R<:Real}","page":"Home","title":"Morbit.eval_and_sort_objectives","text":"Evaluate all objectives at site x̂::RVec and sort the result according to the order in which objectives were added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_handle-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.eval_handle","text":"Return a function that evaluates an objective at a unscaled site.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_models-Tuple{Morbit.ExactModel, Vector{R} where R<:Real, Int64}","page":"Home","title":"Morbit.eval_models","text":"Evaluate output ℓ of the ExactModel em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_models-Tuple{Morbit.ExactModel, Vector{R} where R<:Real}","page":"Home","title":"Morbit.eval_models","text":"Evaluate the ExactModel em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_models-Tuple{Morbit.SurrogateContainer, Morbit.AbstractMOP, Vector{R} where R<:Real, Int64}","page":"Home","title":"Morbit.eval_models","text":"Return model value for output l of sc at x̂. Index l is assumed to be an internal index in the range of 1,…,nobjfs, where nobjfs is the total number of (scalarized) objectives stored in sc.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.eval_objf-Tuple{Morbit.AbstractObjective, Vector{R} where R<:Real}","page":"Home","title":"Morbit.eval_objf","text":"Evaluate the objective at unscaled site(s). and increase counter.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.find_box_independent_points1!-Tuple{Morbit.RbfModel, RbfConfig, Morbit.AbstractIterData, Morbit.AbstractMOP, Morbit.AbstractConfig}","page":"Home","title":"Morbit.find_box_independent_points1!","text":"Find affinely independent results in database box of radius Δ around x Results are saved in rbf.tdata[tdata_index].  Both rbf.Y and rbf.Z are changed.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.find_points_in_box-Tuple{Morbit.AbstractIterData, Vector{R} where R<:Real, Vector{R} where R<:Real}","page":"Home","title":"Morbit.find_points_in_box","text":"Indices of sites in database that lie in box with bounds lb and ub.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.full_lower_bounds_internal-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.full_lower_bounds_internal","text":"Return lower variable bounds for scaled variables.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.full_upper_bounds_internal-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.full_upper_bounds_internal","text":"Return upper variable bounds for scaled variables.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_gradient-Tuple{Morbit.ExactModel, Vector{R} where R<:Real, Int64}","page":"Home","title":"Morbit.get_gradient","text":"Gradient vector of output ℓ of em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_gradient-Tuple{Morbit.SurrogateContainer, Morbit.AbstractMOP, Vector{R} where R<:Real, Int64}","page":"Home","title":"Morbit.get_gradient","text":"Return a gradient for output l of sc at x̂. Index 4 is assumed to be an internal index in the range of 1,…,nobjfs, where nobjfs is the total number of (scalarized) objectives stored in sc.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_jacobian-Tuple{Morbit.ExactModel, Vector{R} where R<:Real}","page":"Home","title":"Morbit.get_jacobian","text":"Jacobian Matrix of ExactModel em at scaled site x̂.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_optim_handle-Tuple{Morbit.SurrogateContainer, Morbit.AbstractMOP, Int64}","page":"Home","title":"Morbit.get_optim_handle","text":"Return a function handle to be used with NLopt for output l of sc. Index l is assumed to be an internal index in the range of 1,…,nobjfs, where nobjfs is the total number of (scalarized) objectives stored in sc.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.get_optim_handle-Tuple{Morbit.SurrogateModel, Int64}","page":"Home","title":"Morbit.get_optim_handle","text":"Return a function handle to be used with NLopt for output ℓ of model. That is, if model is a surrogate for two scalar objectives, then ℓ must  be either 1 or 2.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.inc_evals!","page":"Home","title":"Morbit.inc_evals!","text":"Increase evaluation count by N\n\n\n\n\n\n","category":"function"},{"location":"#Morbit.init_surrogates-Tuple{Morbit.AbstractMOP, Morbit.AbstractIterData, Morbit.AbstractConfig}","page":"Home","title":"Morbit.init_surrogates","text":"Return a SurrogateContainer initialized from the information provided in mop.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.initialize_data-Tuple{Morbit.AbstractMOP, Vector{R} where R<:Real, Vector{R} where R<:Real}","page":"Home","title":"Morbit.initialize_data","text":"Perform initialization of the data passed to optimize function.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.load_config-Tuple{AbstractString}","page":"Home","title":"Morbit.load_config","text":"load_config(fn :: AbstractString)\n\nLoad and return the AbstractConfig that was saved previously  with save_config. Return nothing on error.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.load_database-Tuple{AbstractString}","page":"Home","title":"Morbit.load_database","text":"load_database(fn :: AbstractString)\n\nLoad and return the <:AbstractDB that was saved previously  with save_database. Return nothing on error.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.load_iter_data-Tuple{AbstractString}","page":"Home","title":"Morbit.load_iter_data","text":"load_iter_data(fn :: AbstractString)\n\nLoad and return the <:AbstractIterData that was saved previously  with save_iter_data. Return nothing on error.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.local_bounds-Tuple{Morbit.AbstractMOP, Vector{R} where R<:Real, Union{Real, Vector{R} where R<:Real}}","page":"Home","title":"Morbit.local_bounds","text":"Local bounds vectors lb_eff and ub_eff using scaled variable constraints from mop.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.max_evals!-Tuple{Morbit.AbstractObjective, Int64}","page":"Home","title":"Morbit.max_evals!","text":"Set upper bound of № evaluations to N\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.max_evals-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.max_evals","text":"(Soft) upper bound on the number of function calls. \n\n\n\n\n\n","category":"method"},{"location":"#Morbit.model_cfg-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.model_cfg","text":"Return surrogate configuration used to model the objective internally.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.non_negative_solutions-Tuple{Int64, Int64}","page":"Home","title":"Morbit.non_negative_solutions","text":"Return array of solution vectors [x1, …, xlen] to the equation x_1 +  + x_len = rhs where the variables must be non-negative integers.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.num_evals!-Tuple{Morbit.AbstractObjective, Int64}","page":"Home","title":"Morbit.num_evals!","text":"Set evaluation counter to N.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.num_evals-Tuple{Morbit.AbstractObjective}","page":"Home","title":"Morbit.num_evals","text":"Number of calls to the original objective function.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.num_objectives-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.num_objectives","text":"Number of scalar-valued objectives of the problem.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.pop_objf!-Tuple{Morbit.AbstractMOP, Morbit.AbstractObjective}","page":"Home","title":"Morbit.pop_objf!","text":"Remove objf from list_of_objectives(mop) and return its output indices.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.ps_polish_algo-Tuple{Morbit.AbstractConfig}","page":"Home","title":"Morbit.ps_polish_algo","text":"Specify local algorithm to polish Pascoletti-Serafini solution. Uses 1/4 of maximum allowed evals.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.reset_evals!-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.reset_evals!","text":"Set evaluation counter to 0 for each VectorObjectiveFunction in m.vector_of_objectives.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.reverse_internal_sorting-Tuple{Vector{R} where R<:Real, Morbit.AbstractMOP}","page":"Home","title":"Morbit.reverse_internal_sorting","text":"Sort an interal objective vector so that the objectives are in the order in which they were added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.reverse_internal_sorting_indices-Tuple{Morbit.AbstractMOP}","page":"Home","title":"Morbit.reverse_internal_sorting_indices","text":"Return index vector so that an internal objective vector is sorted according to the order the objectives where added.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.save_config-Tuple{AbstractString, Morbit.AbstractConfig}","page":"Home","title":"Morbit.save_config","text":"save_config(filename, ac :: AbstractConfig )\n\nSave the configuration object ac at path filename. Ensures, that the file extension is .jld2. The fieldname to retrieve the database object is database.\n\nReturns the save path if successful and nothing else.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.save_database-Tuple{AbstractString, Morbit.AbstractDB}","page":"Home","title":"Morbit.save_database","text":"save_database(filename, DB :: AbstractDB )\n\nSave the database DB at path filename. Ensures, that the file extension is .jld2. The fieldname to retrieve the database object is database.\n\nReturns the save path if successful and nothing else.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.save_database-Tuple{AbstractString, Morbit.AbstractIterData}","page":"Home","title":"Morbit.save_database","text":"save_database(filename, id :: AbstractIterData )\n\nSave the database that is referenced by db(id) at path filename. Ensures, that the file extension is .jld2. The fieldname to retrieve the database object is database.\n\nReturns the save path if successful and nothing else.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.save_iter_data-Tuple{AbstractString, Morbit.AbstractIterData}","page":"Home","title":"Morbit.save_iter_data","text":"save_iter_data(filename, id :: AbstractIterData )\n\nSave the whole object id at path filename. Ensures, that the file extension is .jld2. The fieldname to retrieve the database object is database.\n\nReturns the save path if successful and nothing else.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.scale-Tuple{Vector{R} where R<:Real, Morbit.AbstractMOP}","page":"Home","title":"Morbit.scale","text":"Scale variables fully constrained to a closed interval to [0,1] internally.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.set_gradients!-Tuple{Morbit.ExactModel, Morbit.AbstractObjective, Morbit.AbstractMOP}","page":"Home","title":"Morbit.set_gradients!","text":"Modify/initialize thec exact model mod so that we can differentiate it later.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.set_gradients!-Tuple{Morbit.TaylorModel, Morbit.AbstractObjective, Morbit.AbstractMOP}","page":"Home","title":"Morbit.set_gradients!","text":"Modify/initialize thec exact model mod so that we can differentiate it later.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.strict_backtracking-Tuple{Morbit.AbstractConfig}","page":"Home","title":"Morbit.strict_backtracking","text":"Require a descent in all model objective components.  Applies only to backtracking descent steps, i.e., :steepest_descent.\n\n\n\n\n\n","category":"method"},{"location":"#Morbit.unscale-Tuple{Vector{R} where R<:Real, Morbit.AbstractMOP}","page":"Home","title":"Morbit.unscale","text":"Reverse scaling for fully constrained variables from [0,1] to their former domain.\n\n\n\n\n\n","category":"method"}]
}
