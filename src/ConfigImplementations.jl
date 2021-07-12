
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
    eps_crit :: Real = _eps_crit(default_config(F));
    gamma_crit :: F = _gamma_crit(default_config(F));
    max_critical_loops :: Int = max_critical_loops(default_config(F));
    
    max_evals :: Int = max_evals( default_config(F) )
    max_iter :: Int = max_iter( default_config(F) );

    db :: Union{Nothing,Type{<:AbstractDB}} = ArrayDB;

    count_nonlinear_iterations :: Bool = count_nonlinear_iterations( default_config(F) );
    
    Δ_0 :: Union{F, Vector{F}} = Δ⁰(default_config(F));
    Δ_max :: Union{F, Vector{F}} = Δᵘ(default_config(F));
        
    # relative stopping 
    # stop if ||Δf|| ≤ ε ||f||
    f_tol_rel::Union{F, Vector{F}} = f_tol_rel( default_config(F) );
    # stop if ||Δx|| ≤ ε ||x||
    x_tol_rel ::Union{F, Vector{F}} = x_tol_rel(default_config(F));

    # absolute stopping
    f_tol_abs ::Union{F, Vector{F}}  = f_tol_abs(default_config(F))
    x_tol_abs ::Union{F, Vector{F}}  = x_tol_abs(default_config(F));

    # stop if ω ≤ ω_tol_rel && Δ .≤ Δ_tol_rel
    ω_tol_rel :: F = ω_tol_rel(default_config(F));
    Δ_tol_rel ::Union{F, Vector{F}} = Δ_tol_rel(default_config(F));

    # stop if ω <= ω_tol_abs 
    ω_tol_abs :: F = ω_tol_abs(default_config(F));

    # stop if Δ .<= Δ_tol_abs 
    Δ_tol_abs ::Union{F, Vector{F}} = Δ_tol_abs(default_config(F));
    
    descent_method :: D = descent_method( default_config(F) )
    
    strict_acceptance_test :: Bool = strict_acceptance_test(default_config(F));
    nu_success :: F = _nu_success( default_config(F) );
    nu_accept :: F = _nu_accept( default_config(F) );

    mu :: F = _mu( default_config(F) );
    beta :: F = _beta( default_config(F) );

    radius_update_method :: Symbol = radius_update_method(default_config(F))
    gamma_grow :: F = _gamma_grow(default_config(F));
    gamma_shrink :: F = _gamma_shrink(default_config(F));
    gamma_shrink_much :: F = _gamma_shrink_much(default_config(F));
    
    @assert descent_method ∈ [:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search] "`descent_method` must be one of `:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search`."
end

for fn in fieldnames(AlgorithmConfig)
    if fn ∉ [ :Δ_0, :Δ_max, :use_db ]
        @eval $fn( ac :: AlgorithmConfig ) = getfield( ac, Symbol($fn) )
    end
end

use_db( ac :: AlgorithmConfig ) = ac.db;
Δ⁰( ac :: AlgorithmConfig ) = ac.Δ_0;
Δᵘ( ac :: AlgorithmConfig ) = ac.Δ_max;

#####################################################
# outer constructors for the lazy user and backwards compatibility
EmptyConfig() = default_config64;
AlgoConfig(args...; kwargs...) = AlgorithmConfig{Float64}(args...; kwargs...);

export EmptyConfig, AlgoConfig;