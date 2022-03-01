
############################################
# Default implementation `DefaultConfig` accepts nearly 
# every `AbstractConfig` default, except that a 
# database is used.

struct DefaultConfig <: AbstractConfig end
global const default_config = DefaultConfig()
use_db( ::DefaultConfig ) = ArrayDB

############################################
# `AlgorithmConfig` is a struct with fields defining the method outputs.
@with_kw struct AlgorithmConfig{
        R,
        D <: Union{Symbol,AbstractDescentConfig},
        VS <: Union{Symbol, AbstractVarScaler},
        FilterType, 
    } <: AbstractConfig

    eps_crit :: R = _eps_crit( default_config )
    gamma_crit :: R = _gamma_crit(default_config)
    max_critical_loops :: Int = max_critical_loops(default_config)
   
    db :: Union{Nothing,Type{<:AbstractDB}} = ArrayDB

    #count_nonlinear_iterations :: Bool = count_nonlinear_iterations( default_config )
    
    delta_0 :: R = delta_0(default_config)
    delta_max :: R = delta_max(default_config)

    max_evals :: Int = max_evals( default_config )
    max_iter :: Int = max_iter( default_config )

    # relative stopping 
    # stop if ||Δf|| ≤ ε ||f||
    f_tol_rel :: Union{R, Vector{R}} = f_tol_rel( default_config )
    # stop if ||Δx|| ≤ ε ||x||
    x_tol_rel :: Union{R, Vector{R}} = x_tol_rel(default_config)

    # absolute stopping
    f_tol_abs :: Union{R, Vector{R}} = f_tol_abs(default_config)
    x_tol_abs :: Union{R, Vector{R}} = x_tol_abs(default_config)

    # stop if ω ≤ omega_tol_rel && Δ .≤ Δ_tol_rel
    omega_tol_rel :: R = omega_tol_rel(default_config)
    delta_tol_rel :: Union{R, Vector{R}} = delta_tol_rel(default_config)

    # stop if ω <= omega_tol_abs 
    omega_tol_abs :: R = omega_tol_abs(default_config)

    # stop if Δ .<= Δ_tol_abs 
    delta_tol_abs :: Union{R, Vector{R}} = delta_tol_abs(default_config)
 
    descent_method :: D = descent_method( default_config )

    strict_acceptance_test :: Bool = strict_acceptance_test(default_config)
    nu_success :: R = _nu_success( default_config )
    nu_accept :: R = _nu_accept( default_config )

    mu :: R = _mu( default_config )
    beta :: R = _beta( default_config )

    radius_update_method :: Symbol = radius_update_method(default_config)
    gamma_grow :: R = _gamma_grow(default_config)
    gamma_shrink :: R = _gamma_shrink(default_config)
    gamma_shrink_much :: R = _gamma_shrink_much(default_config)

    combine_models :: Bool = _combine_models_by_type( default_config )
    
    @assert descent_method isa AbstractDescentConfig ||
        ( descent_method isa Symbol && descent_method ∈ 
            [:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search] 
        ) "`descent_method` must be one of `:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search`."

    var_scaler :: VS = var_scaler( default_config )
    untransform_final_database :: Bool = untransform_final_database( default_config )
    var_scaler_update :: Symbol = var_scaler_update( default_config )

    filter_type :: FilterType = filter_type(default_config)

    @assert var_scaler isa AbstractVarScaler || var_scaler in [:default, :none, :auto] "Invalid VarScaler. Try one of `:default, :none, :auto`."
end

_config_precision( :: AlgorithmConfig{R,<:Any,<:Any}) where R = R
function Base.convert(::Type{<:Union{R, Vector{R}}}, x :: Number ) where R<:Number
    return convert(R, x)
end

function AlgorithmConfig{R}(; 
    descent_method :: D = descent_method( default_config ),
    var_scaler_update :: VS = var_scaler_update( default_config ),
    filter_type :: FilterType = filter_type( default_config ),
    kwargs...) where {R,D,VS, FilterType}
    return AlgorithmConfig{R,D,VS,FilterType}(; descent_method, var_scaler_update, filter_type, kwargs...)
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
