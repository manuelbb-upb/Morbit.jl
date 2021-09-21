# In this file, methods for the abstract type `AbstractConfig` are defined.
# An implementation of a subtype can overwrite these. 
# Else, they work as a default. 
# `AbstractConfig` has a type parameter `F<:AbstractFloat` indicating
# the data precision. It usually does not matter for thresholds, but some 
# methods return parameters that influence either `x` or `Δ`.
# If not clear, possible return types are indicated for the implementation of subtypes.

Broadcast.broadcastable( ac :: AbstractConfig ) = Ref( ac )

# Criticality Test parameters 
# # ω threshold
_eps_crit( :: AbstractConfig ) :: Float64 = 1e-3
# # shrinking in critical loop
_gamma_crit( ::AbstractConfig ) :: Float64 = .501;   

# # maximum number of loops before exiting
max_critical_loops( :: AbstractConfig ) :: Int = 5

# is a database used? if yes, what is the type?
use_db( :: AbstractConfig ) :: Bool = true 

# should iterations, where the models are not fully linear, be counted for stopping?
count_nonlinear_iterations( :: AbstractConfig ) :: Bool = true

# initial box radius (for fully constrained problems this is relative to ``[0,1]^n```)
get_delta_0(::AbstractConfig) :: NumOrVec64 = 0.1

# radius upper bound(s)
get_delta_max(::AbstractConfig) :: NumOrVec64 = 0.5

# STOPPING 
# restrict number of evaluations and iterations
max_evals( :: AbstractConfig ) :: Int = typemax(Int)
max_iter( :: AbstractConfig ) :: Int = 50

# relative stopping 
# stop if ||Δf|| ≤ ε ||f|| (or |Δf_ℓ| .≤ ε |f_ℓ| )
f_tol_rel( :: AbstractConfig ) :: NumOrVec64 = 1e-8
# stop if ||Δx|| ≤ ε ||x||
x_tol_rel( :: AbstractConfig ) :: NumOrVec64 = 1e-8

# absolute stopping
f_tol_abs( :: AbstractConfig ) :: NumOrVec64 = -1.0
x_tol_abs( :: AbstractConfig ) :: NumOrVec64 = -1.0

# stop if ω ≤ omega_tol_rel && Δ .≤ Δ_tol_rel
omega_tol_rel( :: AbstractConfig ) :: Float64 = 1e-3
delta_tol_rel( :: AbstractConfig ) :: NumOrVec64 = 1e-2

# stop if ω <= omega_tol_abs 
omega_tol_abs(ac :: AbstractConfig ) :: Float64 = -1.0

# stop if Δ .<= Δ_tol_abs 
delta_tol_abs(ac :: AbstractConfig ) :: NumOrVec64 = 1e-6

# what method to use for the subproblems
descent_method( :: AbstractConfig ) :: Union{AbstractDescentConfig,Symbol} = :steepest_descent

# acceptance test parameters
strict_acceptance_test( :: AbstractConfig ) :: Bool = true
_nu_success( :: AbstractConfig ) :: Float64 = 0.2
_nu_accept(::AbstractConfig) :: Float64 = 1e-3

_mu(::AbstractConfig) :: Float64 = 2e3
_beta(::AbstractConfig) :: Float64 = 1e3

# Parameters for the radius update
radius_update_method(::AbstractConfig)::Symbol = :standard
_gamma_grow(::AbstractConfig) :: Float64 = 2.0
_gamma_shrink(::AbstractConfig) :: Float64 = .75
_gamma_shrink_much(::AbstractConfig) :: Float64 = .501

_combine_models_by_type(::AbstractConfig) :: Bool = true