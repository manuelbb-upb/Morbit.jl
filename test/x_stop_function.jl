using Morbit;
using Test;

using Pkg;
old_env = Pkg.project().path;
Pkg.activate( joinpath( @__DIR__, ".." ) );
import JuMP, OSQP;

#%%
# Define 2-Parabola Problem
f1(x) = sum( (x .- 1.0).^2 );
f2(x) = sum( (x .+ 1.0).^2 );

∇f1(x) = 2 .* ( x .- 1.0 );
∇f2(x) = 2 .* ( x .+ 1.0 );

UB = 2 .* ones(2);
LB = -UB;
x_0 = LB .+ (UB .- LB) .* rand(2);

mop = MixedMOP( lb = LB, ub = UB );
add_objective!( mop, f1, ∇f1 );
add_objective!( mop, f2, ∇f2 );
#%%

#%%
# Now build a stop function that returns `true`
# if the algorithm should stop
# Here: return ω(x) < STOP_TOL
x_stop_function = function(x)
    STOP_TOL = 5e-2;
    n_vars = length(x);
    ∇F = transpose( [ ∇f1(x) ∇f2(x) ] );

    prob = JuMP.Model( OSQP.Optimizer );
    JuMP.set_silent(prob);

    JuMP.set_optimizer_attribute(prob,"eps_rel",1e-5);
    JuMP.set_optimizer_attribute(prob,"polish",true);

    JuMP.@variable(prob, α );     # negative of marginal problem value
    JuMP.@variable(prob, d[1:n_vars] );   # direction vector
    
    JuMP.@objective(prob, Min, α);

    JuMP.@constraint(prob, descent_contraints, ∇F*d .<= α);
    JuMP.@constraint(prob, norm_constraints, -1.0 .<= d .<= 1.0);
    
    # the MOP is constrained
    JuMP.@constraint(prob, box_constraints, LB .<= x .+ d .<= UB );

    JuMP.optimize!(prob)
    ω = - JuMP.value( α )
    return ω < STOP_TOL;
end

# Setup optimization
opt = AlgoConfig(
    Δ_max = .4,
    ε_crit = 0.0,
    radius_update = :steplength,
    γ_grow = 1.0,
    all_objectives_descent = true,
    x_stop_function = x_stop_function,
);
#%%

X, FX = Morbit.optimize!(opt, mop, x_0);

@test x_stop_function( X )
Pkg.activate( old_env );

