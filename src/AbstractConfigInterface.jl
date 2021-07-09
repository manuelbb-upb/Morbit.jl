# In this file, methods for the abstract type `AbstractConfig` are defined.
# An implementation of a subtype can overwrite these. 
# Else, they work as a default. 
# `AbstractConfig` has a type parameter `F<:AbstractFloat` indicating
# the data precision. It usually does not matter for thresholds, but some 
# methods return parameters that influence either `x` or `Δ`.
# If not clear, possible return types are indicated for the implementation of subtypes.

Broadcast.broadcastable( ac :: AbstractConfig ) = Ref( ac );

# Criticality Test parameters 
# # ω threshold
ε_crit( :: AbstractConfig{F} ) where F = F(1e-3);
# # shrinking in critical loop
γ_crit( ::AbstractConfig{F} ) where F = F(.501);   
# # maximum number of loops before exiting
max_critical_loops( :: AbstractConfig )::Int = 5;

# is a database used? if yes, what is the type?
use_db( :: AbstractConfig ) :: Bool = true 

# should iterations, where the models are not fully linear, be counted for stopping?
count_nonlinear_iterations( :: AbstractConfig )::Bool = true;

# initial box radius (for fully constrained problems this is relative to ``[0,1]^n```)
( Δ⁰(::AbstractConfig{F}) ::Union{F, AbstractVector{F}} ) where F = F(0.1);

# radius upper bound(s)
( Δᵘ(::AbstractConfig{F}) ::Union{F, AbstractVector{F}} ) where F  = F(0.5);

# STOPPING 
# restrict number of evaluations and iterations
max_evals( :: AbstractConfig ) :: Int = typemax(Int);
max_iter( :: AbstractConfig ) :: Int = 50;

# relative stopping 
# stop if ||Δf|| ≤ ε ||f|| (or |Δf_ℓ| .≤ ε |f_ℓ| )
( f_tol_rel( :: AbstractConfig{F} ) :: Union{F, AbstractVector{F}} ) where F = F(1e-8);
# stop if ||Δx|| ≤ ε ||x||
( x_tol_rel( :: AbstractConfig{F} ) ::Union{F, AbstractVector{F}} ) where F = F(1e-8);

# absolute stopping
( f_tol_abs( :: AbstractConfig{F} ) ::Union{F, AbstractVector{F}} ) where F = F(-1);
( x_tol_abs( :: AbstractConfig{F} ) ::Union{F, AbstractVector{F}} ) where F = F(-1);

# stop if ω ≤ ω_tol_rel && Δ .≤ Δ_tol_rel
( ω_tol_rel( :: AbstractConfig{F} ) :: F ) where F = F(1e-3);
( Δ_tol_rel( :: AbstractConfig{F} ) ::Union{F, AbstractVector{F}} ) where F = F(1e-2);

# stop if ω <= ω_tol_abs 
( ω_tol_abs(ac :: AbstractConfig{F} ) :: F ) where F = F(-1);

# stop if Δ .<= Δ_tol_abs 
( Δ_tol_abs(ac :: AbstractConfig{F} ) ::Union{F, AbstractVector{F}} ) where F = F(1e-6);

# what method to use for the subproblems
descent_method( :: AbstractConfig )::Symbol = :steepest_descent # or :ps

"Require a descent in all model objective components. 
Applies only to backtracking descent steps, i.e., :steepest_descent."
strict_backtracking( :: AbstractConfig )::Bool = true;

# settings for pascoletti_serafini descent (and directed search)
( reference_point(::AbstractConfig{F}) :: AbstractVector{F} ) where F = F[];
( reference_direction(::AbstractConfig{F}) :: AbstractVector{F}) where F = F[];
max_ps_problem_evals(::AbstractConfig)::Int = -1;
max_ps_polish_evals(::AbstractConfig)::Int = -1;
max_ideal_point_problem_evals(::AbstractConfig) :: Int = -1;
ps_algo(::AbstractConfig)::Symbol= :GN_ISRES; #valid NLopt algorithm, e.g. GN_ISRES or GN_AGS (last only works for n<=10)
ideal_point_algo(::AbstractConfig)::Symbol=:GN_ISRES;
"Specify local algorithm to polish Pascoletti-Serafini solution. Uses 1/4 of maximum allowed evals."
ps_polish_algo(::AbstractConfig)::Union{Nothing,Symbol}=:LD_MMA

# acceptance test parameters
strict_acceptance_test( :: AbstractConfig )::Bool = true;
(ν_success( :: AbstractConfig{F} )::F) where F = F(0.1);
(ν_accept(::AbstractConfig{F})::F) where F = F(0);

(μ(::AbstractConfig{F})::F) where F = F(2e3);
(β(::AbstractConfig{F})::F) where F = F(1e3);

# Parameters for the radius update
radius_update_method(::AbstractConfig)::Symbol = :standard;
( γ_grow(::AbstractConfig{F})::F ) where F = F(2);
( γ_shrink(::AbstractConfig{F})::F ) where F = F(.75);
( γ_shrink_much(::AbstractConfig{F})::F )where F= F(.501);