using Morbit
using MultiObjectiveProblems
using AbstractPlotting, CairoMakie

test_problem = ZDT2(2);
box = constraints(test_problem);
I = get_ideal_point(test_problem);

objectives = get_objectives(test_problem)
x₀ = get_random_point(test_problem)

ac = AlgoConfig( descent_method = :ds, )
#reference_point = I )
mop = MixedMOP( box.lb, box.ub );
objf_cfg = ExactConfig()
for objf ∈ objectives
    add_objective!(mop, objf, objf_cfg)
end
#%%

x, fx, id = optimize( mop, x₀; algo_config = ac);
x
#%%
pset = get_pareto_set(test_problem)
PSx,PSy = get_scatter_points(pset, 100)

fig, ax, _ = scatter(PSx, PSy)
scatter!(Tuple(x); color = :red)
fig