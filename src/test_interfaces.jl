using Morbit
M = Morbit

#%%
using CairoMakie
Makie.convert_arguments(x::Circle) = (decompose(Point2f, x),)

#M.print_all_logs()
n_vars = 2

#%%
mop = M.MOP(n_vars)

vars = M.var_indices(mop)

f1 = x -> [ (x[1] - 2)^2 + sum( x[2:end] .- 1 ).^2 ] #x -> sum((x.-1).^2)
f2 = x -> [ (x[1] - 2)^2 + sum( x[2:end] .+ 1 ).^2 ]#x -> sum((x.+1).^2)
F = x -> [f1(x);f2(x)]

oind = M.add_objective!(mop, f1; model_cfg = M.RbfConfig() )
#oind = M.add_objective!(mop, f1; model_cfg = M.ExactConfig(), diff_method = M.AutoDiffWrapper )
#oind2 = M.add_objective!(mop, f2; model_cfg = M.RbfConfig() )
#oind2 = M.add_objective!(mop, f2; model_cfg = M.TaylorConfig() )
#oind2 = M.add_objective!(mop, f2; model_cfg = M.TaylorCallbackConfig(), diff_method = M.AutoDiffWrapper )
oind2 = M.add_objective!(mop, f2; model_cfg = M.ExactConfig() )#, diff_method = M.AutoDiffWrapper )

#= ineqconst = M.add_ineq_constraint!(mop, 
	[1 zeros(n_vars-1)'])
=#
#ineqconst = M.add_nl_ineq_constraint!(mop, x -> 1 - sum( x.^2 ); model_cfg = M.ExactConfig())
ineqconst = M.add_nl_ineq_constraint!(mop, x -> 1 - sum( x.^2 ); model_cfg = M.RbfConfig())

#algo_config = M.AlgoConfig( var_scaler = M.NoVarScaling)

#=
M.add_lower_bound!( mop, vars[1], -10 )
M.add_lower_bound!( mop, vars[2], -3 )
M.add_upper_bound!( mop, vars[1], 10 )
M.add_upper_bound!( mop, vars[2], 10)
M.add_lower_bound!( mop, vars[3], -10)
M.add_upper_bound!( mop, vars[3], 10)
=#

#%%
mop = M.MOP(2)
M.add_objective!( mop, x -> (x[1] + 1)^2 + x[2]^2; model_cfg = M.ExactConfig() )
M.add_objective!( mop, x -> (x[1] - 1)^2 + x[2]^2; model_cfg = M.ExactConfig() )
#M.add_objective!( mop, x -> x[1]^2 + x[2]^2; model_cfg = M.ExactConfig() )

centers = collect(Iterators.flatten( [
	[[ -1.0, 2*i ] for i = 1:4 ],
	[[ 1.0, 2*i ] for i = 1:4 ],
	[[ 0.0, 2*i + 1 ] for i = 1:4 ],
]))

radii = fill( 1/sqrt(2), length(centers) )

circles = Circle[]
for (c,r) = zip( centers, radii )
	push!(circles, Circle(Point(c...), r))
	M.add_nl_ineq_constraint!(mop, x -> r^2 - sum( (x .- c ).^2 ); model_cfg = M.ExactConfig() )
end

x0 = [0.2, 10.0]
x, fx, ret, sdb, id = M.optimize( mop, x0; verbosity = 1);

#%%
fig = Figure(resolution = (250, 800))
ax = Axis(fig[1,1])
ax.aspect = DataAspect()

lines!(ax, [-1.0, 1.0], [0.0, 0.0], color = :blue)
for circ in circles 
	lines!(ax, circ; color = :blue)
end

lines!(ax, Tuple.( t.x for t = sdb.iter_data) )
scatter!(ax, Tuple.( t.x for t = sdb.iter_data); color = collect(cgrad(:blues, length(sdb.iter_data))) )

fig
#%% manual iteration:
#=
@enter begin
	smop, id, sdb, sc, ac, filter, scal = M.initialize_data( mop, [-2,1]);
end
M.iterate!(id, sdb, smop, sc, ac, filter, scal)
=#

#%%
# single-call 
# TODO fix bug when input is int
algo_config = M.AlgorithmConfig{Float32}(;
	max_iter = 30,
	x_tol_rel = -Inf,
	f_tol_rel = -Inf,
	filter_type = M.StrictFilter,
	#descent_method = :ps
)
#%%
x0 = [-0.5f0, 0]

x0 = [-2.0, 0.0]
x, fx, ret, sdb, id = M.optimize( mop, x0; algo_config, verbosity = 4);
x

#%%#%%
circ = Circle( Point2(0.0,0.0), 1.0 )
fig, ax, _ = lines(circ)
ax.aspect = DataAspect(1.0)
lines!(ax, Tuple.( t.x for t = sdb.iter_data) )
scatter!(ax, Tuple.( t.x for t = sdb.iter_data) )
fig