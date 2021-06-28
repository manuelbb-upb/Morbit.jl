
############################################
# Default implementation `EmptyConfig` accepts nearly 
# every `AbstractConfig` default, except that a 
# database is used.

struct EmptyConfig{F} <: AbstractConfig{F} end;

global const empty_config16 = EmptyConfig{Float16}();
global const empty_config32= EmptyConfig{Float32}();
global const empty_config64 = EmptyConfig{Float64}();
global const empty_configBig = EmptyConfig{BigFloat}();

empty_config( F :: Type{<:AbstractFloat}) = EmptyConfig{F}();
empty_config( :: Type{Float16}) = empty_config16;
empty_config( :: Type{Float32}) = empty_config32;
empty_config( :: Type{Float64}) = empty_config64;
empty_config( :: Type{BigFloat}) = empty_configBig;

use_db( ::EmptyConfig ) = ArrayDB;

############################################
# `AlgoConfig` is a struct with fields defining the method outputs.
@with_kw struct AlgoConfig{F} <: AbstractConfig{F}
    ε_crit :: Real = ε_crit(empty_config(F));
    γ_crit :: F = γ_crit(empty_config(F));
    max_critical_loops :: Int = max_critical_loops(empty_config(F));
    
    max_evals :: Int = max_evals( empty_config(F) )
    max_iter :: Int = max_iter( empty_config(F) );

    db :: Union{Nothing,Type{<:AbstractDB}} = ArrayDB;

    count_nonlinear_iterations :: Bool = count_nonlinear_iterations( empty_config(F) );
    
    Δ_0 :: Union{F, Vector{F}} = Δ⁰(empty_config(F));
    Δ_max :: Union{F, Vector{F}} = Δᵘ(empty_config(F));
        
    # relative stopping 
    # stop if ||Δf|| ≤ ε ||f||
    f_tol_rel::Union{F, Vector{F}} = f_tol_rel( empty_config(F) );
    # stop if ||Δx|| ≤ ε ||x||
    x_tol_rel ::Union{F, Vector{F}} = x_tol_rel(empty_config(F));

    # absolute stopping
    f_tol_abs ::Union{F, Vector{F}}  = f_tol_abs(empty_config(F))
    x_tol_abs ::Union{F, Vector{F}}  = x_tol_abs(empty_config(F));

    # stop if ω ≤ ω_tol_rel && Δ .≤ Δ_tol_rel
    ω_tol_rel :: F = ω_tol_rel(empty_config(F));
    Δ_tol_rel ::Union{F, Vector{F}} = Δ_tol_rel(empty_config(F));

    # stop if ω <= ω_tol_abs 
    ω_tol_abs :: F = ω_tol_abs(empty_config(F));

    # stop if Δ .<= Δ_tol_abs 
    Δ_tol_abs ::Union{F, Vector{F}} = Δ_tol_abs(empty_config(F));
    
    descent_method :: Symbol = descent_method(empty_config(F));
    
    # steepest_descent settings
    strict_backtracking :: Bool = strict_backtracking(empty_config(F));
    
    # pascoletti_serafini settings
    reference_point :: RVec = reference_point(empty_config(F));
    reference_direction :: RVec = reference_direction(empty_config(F))
    max_ps_problem_evals :: Int = max_ps_problem_evals(empty_config(F));
    max_ps_polish_evals :: Int = max_ps_polish_evals(empty_config(F));
    max_ideal_point_problem_evals :: Int = max_ideal_point_problem_evals(empty_config(F));
    ps_algo :: Symbol = ps_algo(empty_config(F));
    ideal_point_algo :: Symbol = ideal_point_algo(empty_config(F));
    ps_polish_algo :: Union{Symbol,Nothing} = ps_polish_algo(empty_config(F));

    strict_acceptance_test :: Bool = strict_acceptance_test(empty_config(F));
    ν_success :: F = ν_success( empty_config(F) );
    ν_accept :: F = ν_accept( empty_config(F) );

    μ :: F = μ( empty_config(F) );
    β :: F = β( empty_config(F) );

    radius_update_method :: Symbol = radius_update_method(empty_config(F))
    γ_grow :: F = γ_grow(empty_config(F));
    γ_shrink :: F = γ_shrink(empty_config(F));
    γ_shrink_much :: F = γ_shrink_much(empty_config(F));
    
    @assert descent_method ∈ [:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search] "`descent_method` must be one of `:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search`."
end

for fn in fieldnames(AlgoConfig)
    if fn ∉ [ :Δ_0, :Δ_max, :use_db ]
        @eval $fn( ac :: AlgoConfig ) = getfield( ac, Symbol($fn) )
    end
end

use_db( ac :: AlgoConfig ) = ac.db;
Δ⁰( ac :: AlgoConfig ) = ac.Δ_0;
Δᵘ( ac :: AlgoConfig ) = ac.Δ_max;

#####################################################
# outer constructors for the lazy user and backwards compatibility
EmptyConfig64() = empty_config64;
AlgoConfig64(args...; kwargs...) = AlgoConfig{Float64}(args...; kwargs...);

export EmptyConfig64, AlgoConfig64;