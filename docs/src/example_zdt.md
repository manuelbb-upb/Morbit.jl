# ZDT3 Problem

## Setup
Install the test problem suite:
```
using Pkg 
Pkg.activate(tempname())
Pkg.develop(url="https://github.com/manuelbb-upb/MultiObjectiveProblems.jl")
using MultiObjectiveProblems
```
```@setup zdt
using Pkg 
Pkg.activate(tempname())
Pkg.develop(url="https://github.com/manuelbb-upb/MultiObjectiveProblems.jl")
using MultiObjectiveProblems
```

Import other dependencies:
```@example zdt
using Morbit
using MultiObjectiveProblems
using AbstractPlotting, CairoMakie
```

Retrieve test problem and define a `MixedMOP`
```@example zdt
test_problem = ZDT3(2);
box = constraints(test_problem);

I = get_ideal_point(test_problem);
objectives = get_objectives(test_problem)
x₀ = get_random_point(test_problem)

ac = AlgoConfig(; descent_method = :ps, reference_point = I )
mop = MixedMOP( box.lb, box.ub );
objf_cfg = ExactConfig()
for objf ∈ objectives
    add_objective!(mop, objf, objf_cfg)
end
```

## Run
Run optimization and plot:
```@example zdt
x, fx, id = optimize( mop, x₀; algo_config = ac);

pset = get_pareto_set(test_problem)
PSx,PSy = get_scatter_points(pset, 100)

# scatter Pareto set points in grey
fig, ax, _ = scatter( PSx, PSy;
    figure = (resolution = (600, 650),), 
)

# set axis limits to whole feasible set
xlims!(ax, (box.lb[1] .- .2, box.ub[1] .+ .2) ) 
ylims!(ax, (box.lb[2] .- .2, box.ub[2] .+ .2) ) 

# final iterate in red
scatter!(Tuple(x); color = :red)
save("example_zdt_scatter.png", fig) # hide
nothing # hide
```

![](example_zdt_scatter.png)