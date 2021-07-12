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
_eps_crit( :: AbstractConfig{F} ) where F = F(1e-3);
# # shrinking in critical loop
_gamma_crit( ::AbstractConfig{F} ) where F = F(.501);   
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
descent_method( :: AbstractConfig ) :: Union{AbstractDescentConfig,Symbol} = :steepest

# acceptance test parameters
strict_acceptance_test( :: AbstractConfig )::Bool = true;
(_nu_success( :: AbstractConfig{F} )::F) where F = F(0.1);
(_nu_accept(::AbstractConfig{F})::F) where F = F(0);

(_mu(::AbstractConfig{F})::F) where F = F(2e3);
(_beta(::AbstractConfig{F})::F) where F = F(1e3);

# Parameters for the radius update
radius_update_method(::AbstractConfig)::Symbol = :standard;
( _gamma_grow(::AbstractConfig{F})::F ) where F = F(2);
( _gamma_shrink(::AbstractConfig{F})::F ) where F = F(.75);
( _gamma_shrink_much(::AbstractConfig{F})::F )where F= F(.501);

#=
# legacy ( TODO remove? )
ε_crit( ac :: AbstractConfig ) = _eps_crit( ac )
_gamma_crit( ac :: AbstractConfig ) = _gamma_crit( ac )
ν_success( ac :: AbstractConfig ) = _nu_success( ac )  
ν_accept( ac :: AbstractConfig ) = _nu_accept( ac )
μ( ac :: AbstractConfig ) = _mu( ac )
β( ac :: AbstractConfig ) = _beta( ac )
γ_grow( ac :: AbstractConfig ) = _gamma_grow( ac )
γ_shrink( ac :: AbstractConfig ) = _gamma_shrink( ac )
γ_shrink_much( ac :: AbstractConfig ) = _gamma_shrink_much( ac )
=#