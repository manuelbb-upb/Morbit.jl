############################################
# AbstractConfig
Broadcast.broadcastable( ac :: AbstractConfig ) = Ref( ac );

max_evals( :: AbstractConfig ) :: Int = typemax(Int);
max_iter( :: AbstractConfig ) :: Int = 50;

ε_crit( ::AbstractConfig )::Real=1e-3;
γ_crit( ::AbstractConfig)::Real=.501;
max_critical_loops( :: AbstractConfig )::Int = 10;

use_db( :: AbstractConfig )::Union{ Type{<:AbstractDB}, Nothing } = nothing;

count_nonlinear_iterations( :: AbstractConfig )::Bool=true;
# initial radius
Δ⁰(::AbstractConfig)::Union{RVec, Real} = 0.1;

# radius upper bound(s)
Δᵘ(::AbstractConfig)::Union{RVec, Real} = 0.5;

Δ_crit(::AbstractConfig)::Union{RVec,Real} = 1e-3;
Δₗ(ac::AbstractConfig)::Union{RVec, Real} = Δ_crit(ac) .* 1e-3;
stepsize_crit(ac::AbstractConfig)::Union{RVec,Real}= Δ_crit(ac) .* 1e-3;
stepsize_min(::AbstractConfig)::Union{RVec,Real} = eps(Float64) * 10;

descent_method( :: AbstractConfig )::Symbol = :steepest_descent # or :ps

"Require a descent in all model objective components. 
Applies only to backtracking descent steps, i.e., :steepest_descent."
strict_backtracking( :: AbstractConfig )::Bool = true;

# settings for pascoletti_serafini descent 
reference_point(::AbstractConfig) :: RVec = Real[];
reference_direction(::AbstractConfig) :: RVec = Real[];
max_ps_problem_evals(::AbstractConfig)::Int = -1;
max_ps_polish_evals(::AbstractConfig)::Int = -1;
max_ideal_point_problem_evals(::AbstractConfig) :: Int = -1;
ps_algo(::AbstractConfig)::Symbol= :GN_ISRES; #valid NLopt algorithm, e.g. GN_ISRES or GN_AGS (last only works for n<=10)
ideal_point_algo(::AbstractConfig)::Symbol=:GN_AGS;
"Specify local algorithm to polish Pascoletti-Serafini solution. Uses 1/4 of maximum allowed evals."
ps_polish_algo(::AbstractConfig)::Union{Nothing,Symbol}=:LD_MMA

strict_acceptance_test( :: AbstractConfig )::Bool = true;
ν_success( :: AbstractConfig )::Real = 0.1;
ν_accept(::AbstractConfig)::Real = 0.0;

μ(::AbstractConfig) = 2e3;
β(::AbstractConfig) = 1e3;

radius_update_method(::AbstractConfig)::Symbol = :standard;
γ_grow(::AbstractConfig)::Real = 2;
γ_shrink(::AbstractConfig)::Real = Float16(.75);
γ_shrink_much(::AbstractConfig)::Real=Float16(.501);

############################################

struct EmptyConfig <: AbstractConfig end;
global const empty_config = EmptyConfig();
use_db( ::EmptyConfig ) = ArrayDB;

############################################

@with_kw struct AlgoConfig <: AbstractConfig
    max_evals :: Int = max_evals( empty_config )
    max_iter :: Int = max_iter( empty_config );

    γ_crit::Real = γ_crit(empty_config);
    max_critical_loops :: Int = max_critical_loops(empty_config);
    ε_crit :: Real = ε_crit(empty_config);
    count_nonlinear_iterations :: Bool = count_nonlinear_iterations( empty_config );
    Δ_0 :: Union{Real, RVec} = Δ⁰(empty_config);
    Δ_max :: Union{Real, RVec } = Δᵘ(empty_config);
    
    Δ_min :: Union{Real, RVec} = Δₗ( empty_config );
    Δ_critical :: Union{Real, RVec } = Δ_crit( empty_config );
    stepsize_critical :: Union{Real,RVec} = stepsize_crit( empty_config );
    stepsize_min :: Union{Real,RVec} = stepsize_min(empty_config);

    descent_method :: Symbol = descent_method(empty_config);
    
    # steepest_descent settings
    strict_backtracking :: Bool = strict_backtracking(empty_config);
    
    # pascoletti_serafini settings
    reference_direction :: RVec = reference_direction(empty_config)
    reference_point :: RVec = reference_point(empty_config);
    max_ideal_point_problem_evals :: Int = max_ideal_point_problem_evals(empty_config);
    max_ps_problem_evals :: Int = max_ps_problem_evals(empty_config);
    max_ps_polish_evals :: Int = max_ps_polish_evals(empty_config);
    ps_algo :: Symbol = ps_algo(empty_config);
    ideal_point_algo :: Symbol = ideal_point_algo(empty_config);
    ps_polish_algo :: Union{Symbol,Nothing} = ps_polish_algo(empty_config);

    strict_acceptance_test :: Bool = strict_acceptance_test(empty_config);
    ν_success :: Real = ν_success( empty_config );
    ν_accept :: Real = ν_accept( empty_config );
    db :: Union{Nothing,Type{<:AbstractDB}} = ArrayDB;

    μ :: Real = μ( empty_config );
    β :: Real = β( empty_config );

    radius_update_method :: Symbol = radius_update_method(empty_config)
    γ_grow :: Real = γ_grow(empty_config);
    γ_shrink :: Real = γ_shrink(empty_config);
    γ_shrink_much::Real = γ_shrink_much(empty_config);
    
    @assert descent_method ∈ [:steepest_descent, :ps, :pascoletti_serafini, :ds, :directed_search]
end

max_evals( ac :: AlgoConfig ) = ac.max_evals;
max_iter( ac :: AlgoConfig ) = ac.max_iter;
max_critical_loops(ac::AlgoConfig)::Int=ac.max_critical_loops;
count_nonlinear_iterations(ac :: AlgoConfig) = ac.count_nonlinear_iterations;
Δ⁰( ac :: AlgoConfig ) = ac.Δ_0;
Δᵘ( ac :: AlgoConfig ) = ac.Δ_max;

Δₗ( ac :: AlgoConfig ) = ac.Δ_min;
Δ_crit( ac :: AlgoConfig ) = ac.Δ_critical;
stepsize_crit( ac :: AlgoConfig ) = ac.stepsize_critical;
stepsize_min(ac::AlgoConfig) = ac.stepsize_min;
use_db( ac :: AlgoConfig ) = ac.db;

descent_method( ac :: AlgoConfig ) = ac.descent_method;

strict_backtracking( ac :: AlgoConfig ) = ac.strict_backtracking;

reference_direction(ac :: AlgoConfig ) = ac.reference_direction;
reference_point(ac :: AlgoConfig ) = ac.reference_point;
max_ideal_point_problem_evals(ac :: AlgoConfig) :: Int = ac.max_ideal_point_problem_evals;
max_ps_problem_evals(ac::AlgoConfig)::Int = ac.max_ps_problem_evals;
max_ps_polish_evals(ac::AlgoConfig)::Int = ac.max_ps_polish_evals;
ps_algo(ac::AlgoConfig)::Symbol=ac.ps_algo
ideal_point_algo(ac::AlgoConfig)::Symbol=ac.ideal_point_algo
ps_polish_algo(ac::AlgoConfig)::Union{Nothing,Symbol}=ac.ps_polish_algo;

strict_acceptance_test( ac :: AlgoConfig ) = ac.strict_acceptance_test;
ν_success( ac :: AlgoConfig ) = ac.ν_success;
ν_accept( ac :: AlgoConfig ) = ac.ν_accept;

μ( ac :: AlgoConfig ) = ac.μ;
β( ac :: AlgoConfig ) = ac.β;

radius_update_method( ac :: AlgoConfig )::Symbol = ac.radius_update_method;
γ_grow(ac :: AlgoConfig)::Real = ac.γ_grow;
γ_shrink(ac :: AlgoConfig)::Real = ac.γ_shrink;
γ_shrink_much(ac :: AlgoConfig)::Real = ac.γ_shrink_much;

γ_crit(ac::AlgoConfig) = ac.γ_crit;
ε_crit(ac::AlgoConfig) = ac.ε_crit;