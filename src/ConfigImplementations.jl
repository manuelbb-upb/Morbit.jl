
############################################
# Default implementation `DefaultConfig` accepts nearly 
# every `AbstractConfig` default, except that a 
# database is used.

struct DefaultConfig{F} <: AbstractConfig{F} end

global const default_config16 = DefaultConfig{Float16}();
global const default_config32= DefaultConfig{Float32}();
global const default_config64 = DefaultConfig{Float64}();
global const default_configBig = DefaultConfig{BigFloat}();

default_config( F :: Type{<:AbstractFloat}) = DefaultConfig{F}();
default_config( :: Type{Float16}) = default_config16;
default_config( :: Type{Float32}) = default_config32;
default_config( :: Type{Float64}) = default_config64;
default_config( :: Type{BigFloat}) = default_configBig;

use_db( ::DefaultConfig ) = ArrayDB;

############################################
# `AlgorithmConfig` is a struct with fields defining the method outputs.
@with_kw struct AlgorithmConfig{ F, D } <: AbstractConfig{F}
    
    # small hidden helper fields
    _x_ :: F = 1.0
    _F_ :: Type{F} = typeof(_x_)

    eps_crit :: F = _eps_crit( default_config( _F_ ) );
    gamma_crit :: F = _gamma_crit(default_config( _F_ ));
    max_critical_loops :: Int = max_critical_loops(default_config( _F_ ));
    
    max_evals :: Int = max_evals( default_config( _F_ ) )
    max_iter :: Int = max_iter( default_config( _F_ ) );

    db :: Union{Nothing,Type{<:AbstractDB}} = ArrayDB{_F_}

    count_nonlinear_iterations :: Bool = count_nonlinear_iterations( default_config( _F_ ) );
    
    delta_0 :: Union{F, Vector{F}} = Δ⁰(default_config( _F_ ));
    delta_max :: Union{F, Vector{F}} = Δᵘ(default_config( _F_ ));
        
    # relative stopping 
    # stop if ||Δf|| ≤ ε ||f||
    f_tol_rel::Union{F, Vector{F}} = f_tol_rel( default_config( _F_ ) );
    # stop if ||Δx|| ≤ ε ||x||
    x_tol_rel ::Union{F, Vector{F}} = x_tol_rel(default_config( _F_ ));

    # absolute stopping
    f_tol_abs ::Union{F, Vector{F}}  = f_tol_abs(default_config( _F_ ))
    x_tol_abs ::Union{F, Vector{F}}  = x_tol_abs(default_config( _F_ ));

    # stop if ω ≤ omega_tol_rel && Δ .≤ Δ_tol_rel
    omega_tol_rel :: F = omega_tol_rel(default_config( _F_ ));
    Δ_tol_rel ::Union{F, Vector{F}} = Δ_tol_rel(default_config( _F_ ));

    # stop if ω <= omega_tol_abs 
    omega_tol_abs :: F = omega_tol_abs(default_config( _F_ ));

    # stop if Δ .<= Δ_tol_abs 
    Δ_tol_abs ::Union{F, Vector{F}} = Δ_tol_abs(default_config( _F_ ));
 
    descent_method :: D = descent_method( default_config( _F_ ) )

    strict_acceptance_test :: Bool = strict_acceptance_test(default_config( _F_ ));
    nu_success :: F = _nu_success( default_config( _F_ ) );
    nu_accept :: F = _nu_accept( default_config( _F_ ) );

    mu :: F = _mu( default_config( _F_ ) );
    beta :: F = _beta( default_config( _F_ ) );

    radius_update_method :: Symbol = radius_update_method(default_config( _F_ ))
    gamma_grow :: F = _gamma_grow(default_config( _F_ ));
    gamma_shrink :: F = _gamma_shrink(default_config( _F_ ));
    gamma_shrink_much :: F = _gamma_shrink_much(default_config( _F_ ));
    
    @assert descent_method ∈ [:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search] "`descent_method` must be one of `:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search`."
end

for fn in fieldnames(AlgorithmConfig)
    if fn ∉ [ :use_db, ]
        @eval $fn( ac :: AlgorithmConfig ) = getfield( ac, Symbol($fn) )
    end
end

use_db( ac :: AlgorithmConfig ) = ac.db

#####################################################
# outer constructors for the lazy user and backwards compatibility
EmptyConfig() = default_config64;
AlgoConfig(args...; kwargs...) = AlgorithmConfig(args...; kwargs...);

export EmptyConfig, AlgoConfig;