
############################################
# Default implementation `DefaultConfig` accepts nearly 
# every `AbstractConfig` default, except that a 
# database is used.

struct DefaultConfig <: AbstractConfig end
global const default_config = DefaultConfig()
use_db( ::DefaultConfig ) = ArrayDB

############################################
# `AlgorithmConfig` is a struct with fields defining the method outputs.
@with_kw struct AlgorithmConfig{ D <: Union{Symbol,AbstractDescentConfig} } <: AbstractConfig @deftype Float64

    eps_crit = _eps_crit( default_config )
    gamma_crit = _gamma_crit(default_config)
    max_critical_loops :: Int = max_critical_loops(default_config)
   
    db :: Union{Nothing,Type{<:AbstractDB}} = ArrayDB

    count_nonlinear_iterations :: Bool = count_nonlinear_iterations( default_config )
    
    delta_0 :: NumOrVec64 = get_delta_0(default_config)
    delta_max :: NumOrVec64 = get_delta_max(default_config)

    max_evals :: Int = max_evals( default_config )
    max_iter :: Int = max_iter( default_config )

    # relative stopping 
    # stop if ||Δf|| ≤ ε ||f||
    f_tol_rel :: NumOrVec64 = f_tol_rel( default_config )
    # stop if ||Δx|| ≤ ε ||x||
    x_tol_rel:: NumOrVec64 = x_tol_rel(default_config)

    # absolute stopping
    f_tol_abs:: NumOrVec64  = f_tol_abs(default_config)
    x_tol_abs :: NumOrVec64  = x_tol_abs(default_config)

    # stop if ω ≤ omega_tol_rel && Δ .≤ Δ_tol_rel
    omega_tol_rel = omega_tol_rel(default_config)
    delta_tol_rel :: NumOrVec64 = delta_tol_rel(default_config)

    # stop if ω <= omega_tol_abs 
    omega_tol_abs = omega_tol_abs(default_config)

    # stop if Δ .<= Δ_tol_abs 
    delta_tol_abs :: NumOrVec64 = delta_tol_abs(default_config)
 
    descent_method :: D = descent_method( default_config )

    strict_acceptance_test :: Bool = strict_acceptance_test(default_config)
    nu_success = _nu_success( default_config )
    nu_accept = _nu_accept( default_config )

    mu = _mu( default_config )
    beta = _beta( default_config )

    radius_update_method :: Symbol = radius_update_method(default_config)
    gamma_grow = _gamma_grow(default_config)
    gamma_shrink = _gamma_shrink(default_config)
    gamma_shrink_much = _gamma_shrink_much(default_config)

    combine_models ::Bool = _combine_models_by_type( default_config )
    
    @assert descent_method isa AbstractDescentConfig ||
        ( descent_method isa Symbol && descent_method ∈ 
            [:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search] 
        ) "`descent_method` must be one of `:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search`."
end

for fn in fieldnames(AlgorithmConfig)
    if fn ∉ [ :use_db, ]
        @eval $fn( ac :: AlgorithmConfig ) = getfield( ac, Symbol($fn) )
    end
end

use_db( ac :: AlgorithmConfig ) = ac.db

#####################################################
# outer constructors for the lazy user and backwards compatibility
EmptyConfig() = default_config
AlgoConfig(args...; kwargs...) = AlgorithmConfig(args...; kwargs...)
