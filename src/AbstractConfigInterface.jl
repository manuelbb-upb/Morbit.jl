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
_eps_crit( :: AbstractConfig ) = MIN_PRECISION(0.001f0)
# # shrinking in critical loop
_gamma_crit( ::AbstractConfig ) = MIN_PRECISION(0.51f0)

# # maximum number of loops before exiting
max_critical_loops( :: AbstractConfig ) :: Int = 5

# is a database used? if yes, what is the type?
use_db( :: AbstractConfig ) :: Bool = true 

# should iterations, where the models are not fully linear, be counted for stopping?
#count_nonlinear_iterations( :: AbstractConfig ) :: Bool = true

# initial box radius (for fully constrained problems this is relative to ``[0,1]^n```)
get_delta_0(::AbstractConfig) = MIN_PRECISION(0.1f0)

# radius upper bound(s)
get_delta_max(::AbstractConfig) = MIN_PRECISION(0.5f0)

# STOPPING 
# restrict number of evaluations and iterations
max_evals( :: AbstractConfig ) :: Int = typemax(Int)
max_iter( :: AbstractConfig ) :: Int = 50

# relative stopping 
# stop if ||Δf|| ≤ ε ||f|| (or |Δf_ℓ| .≤ ε |f_ℓ| )
f_tol_rel( :: AbstractConfig ) = sqrt(eps(MIN_PRECISION))
# stop if ||Δx|| ≤ ε ||x||
x_tol_rel( ac :: AbstractConfig ) = f_tol_rel(ac)

# absolute stopping
f_tol_abs( :: AbstractConfig ) = MIN_PRECISION(-1)
x_tol_abs( :: AbstractConfig ) = MIN_PRECISION(-1)

# stop if ω ≤ omega_tol_rel && Δ .≤ Δ_tol_rel
omega_tol_rel( ac :: AbstractConfig ) = 10 * f_tol_rel( ac )[end]
delta_tol_rel( ac :: AbstractConfig )= x_tol_rel( ac )[end]

# stop if ω <= omega_tol_abs 
omega_tol_abs(:: AbstractConfig ) = MIN_PRECISION(-Inf) #sqrt(eps(MIN_PRECISION))

# stop if Δ .<= Δ_tol_abs 
delta_tol_abs(ac :: AbstractConfig ) = f_tol_rel( ac )

# what method to use for the subproblems
descent_method( :: AbstractConfig ) :: Union{AbstractDescentConfig,Symbol} = :steepest_descent

# acceptance test parameters
strict_acceptance_test( :: AbstractConfig ) :: Bool = true
_nu_success( :: AbstractConfig ) = MIN_PRECISION(0.4f0)
_nu_accept(::AbstractConfig) = MIN_PRECISION(0)

_mu(::AbstractConfig) = MIN_PRECISION(2e3)
_beta(::AbstractConfig) = MIN_PRECISION(1e3)

# Parameters for the radius update
radius_update_method(::AbstractConfig)::Symbol = :standard
_gamma_grow(::AbstractConfig) = MIN_PRECISION(2.0f0)
_gamma_shrink(::AbstractConfig) = MIN_PRECISION(.75f0)
_gamma_shrink_much(::AbstractConfig) = MIN_PRECISION(.51f0)

_combine_models_by_type(::AbstractConfig) :: Bool = true

filter_type( :: AbstractConfig ) = MaxFilter
filter_shift( :: AbstractConfig ) = MIN_PRECISION(1e-4)

filter_kappa_psi( :: AbstractConfig ) = MIN_PRECISION(1e-4)
filter_psi( :: AbstractConfig ) = MIN_PRECISION(1)

filter_kappa_delta(:: AbstractConfig) = MIN_PRECISION(0.7f0)
filter_kappa_mu( :: AbstractConfig ) = MIN_PRECISION(100)
filter_mu( :: AbstractConfig ) = MIN_PRECISION(0.01f0)

var_scaler( :: AbstractConfig ) :: Union{AbstractVarScaler,Symbol} = :default # :none, :auto, :default
untransform_final_database( :: AbstractConfig ) = false
var_scaler_update( :: AbstractConfig ) = :none

iter_saveable_type( :: AbstractConfig ) = IterSaveable