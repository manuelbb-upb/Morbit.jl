module Morbit

# steepest descent
using LinearAlgebra: norm
import JuMP;
import OSQP;

using Parameters: @with_kw, @unpack, @pack!
using MathOptInterface;
const MOI = MathOptInterface;
using Memoize: @memoize, memoize_cache;
import UUIDs;

import FiniteDiff#erences
const FD = FiniteDiff#erences

import ForwardDiff
const AD = ForwardDiff

include("shorthands.jl");

include("Interfaces.jl");

include("diff_wrappers.jl");

# implementations
include("VectorObjectiveFunction.jl");
include("MixedMOP.jl");

include("Surrogates.jl");

include("objectives.jl");

############################################

Broadcast.broadcastable( db :: AbstractDB ) = Ref( db );
Broadcast.broadcastable( id :: AbstractIterData ) = Ref( id );
Broadcast.broadcastable( ac :: AbstractConfig ) = Ref( ac );

############################################
# AbstractDB
Base.length( db :: AbstractDB ) = length(sites(db)) :: Int;

init_db( :: Type{<:AbstractDB} ) = nothing :: AbstractDB;
init_db( :: Nothing ) = nothing :: Nothing;

# array of all evaluation sites
sites( :: AbstractDB ) = RVec[] :: RVecArr;
values( :: AbstractDB ) = RVec[] :: RVecArr;

# index of current iterate 
function xᵗ_index( ::AbstractDB ) :: Int
   -1
end

xᵗ_index!(::AbstractDB, ::Int) :: Nothing = nothing;

add_site!( :: AbstractDB, :: RVec ) :: Int = -1;
add_value!( :: AbstractDB, :: RVec ) :: Int = -1;

Base.eachindex( db :: AbstractDB ) = eachindex( sites(db) );

change_site!( :: AbstractDB, :: Int, :: RVec) = nothing :: Nothing;
change_value!( :: AbstractDB, :: Int, :: RVec) = nothing :: Nothing;

stamp_x_index!( :: AbstractDB )::Nothing = nothing;

# DERIVED AND DEFAULTS
function stamp!(db::AbstractDB) ::Nothing 
    stamp_x_index!(db)
end

function get_site( db :: AbstractDB, pos :: Int) :: RVec
    let sites = sites(db);
        isempty(sites) ? Real[] : sites[pos]
    end
end
function get_value( db :: AbstractDB, pos :: Int) :: RVec
    let values = values(db);
        isempty(values) ? Real[] : values[pos]
    end
end

function get_result( db :: AbstractDB, pos :: Int ) :: Tuple{RVec, RVec}
    return ( get_site(db,pos), get_value(db,pos) )
end

function add_result!( db :: AbstractDB, x̂ :: RVec, ŷ :: RVec ) :: Int 
    site_id = add_site!(db, x̂);
    val_id = add_value!(db, ŷ);
    @assert site_id == val_id;
    return site_id;
end

"Add the next iterate `x̂` and its values `ŷ` to the database."
function add_iter_result!( db :: AbstractDB, x̂ :: RVec, ŷ :: RVec ) :: Int 
    pos = add_result!(db, x̂, ŷ);
    xᵗ_index!(db, pos);
    return pos; 
end

function change_result!( db :: AbstractDB, pos :: Int, 
    new_site :: RVec, new_value :: RVec) :: Nothing 
    change_site!(db, pos, new_site);
    change_value!(db, pos, new_value);
end

function find_result( db :: AbstractDB, x :: RVec, y :: RVec ) :: Int
    for pos ∈ eachindex(db)
        if get_site(db, pos) ≈ x 
            if get_value(db, pos) ≈ y
                change_result!(db, pos, x, y)
                return pos 
            end
        end
    end
    return -1
end

function contains_result( db :: AbstractDB, x :: RVec, y :: RVec ) :: Bool 
    return find_result(db, x, y) >= 1
end

function ensure_contains_result!( db :: AbstractDB, x :: RVec, y :: RVec ) :: Nothing 
    x_pos = find_result(db, x, y);
    if x_pos < 1
        x_pos = add_result!(db,x,y)
    end
    xᵗ_index!(db, x_pos)
    nothing 
end

function scale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for pos ∈ eachindex(db)
        x = get_site( db, pos);
        change_site!( db, pos, scale(x, mop) )
    end
    nothing
end

function unscale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for pos ∈ eachindex(db)
        x̂ = get_site( db, pos);
        change_site!( db, pos, unscale(x̂, mop) )
    end
    nothing
end

function apply_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for pos ∈ eachindex(db)
        y = get_value( db, pos );
        change_value!( db, pos, apply_internal_sorting( y, mop ) );
    end
    nothing 
end

function reverse_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing 
    for pos ∈ eachindex(db)
        @show ŷ = get_value(db, pos)
        change_value!( db, pos, reverse_internal_sorting( ŷ, mop) )
    end
    nothing  
end

############################################
# AbstractIterData

# current iterate and values
xᵗ( :: AbstractIterData ) = nothing :: RVec;
fxᵗ( :: AbstractIterData ) = nothing :: RVec;
# trust region radius
Δᵗ( :: AbstractIterData ) = nothing :: Union{Real, RVec};

db( :: AbstractIterData ) = nothing :: Union{AbstractDB,Nothing};

# setters
function xᵗ!( id :: AbstractIterData, x̂ :: RVec ) :: Nothing 
    nothing
end

function fxᵗ!( id :: AbstractIterData, ŷ :: RVec ) :: Nothing 
    nothing
end

function Δᵗ!( id :: AbstractIterData, Δ :: Union{Real, RVec} ) :: Nothing
    nothing
end

# generic initializer
init_iter_data( ::Type{<:AbstractIterData}, x :: RVec, fx :: RVec, Δ :: Union{Real, RVec}, 
    db :: Union{AbstractDB,Nothing}) = nothing :: AbstractIterData;

function set_next_iterate!( id :: AbstractIterData, x̂ :: RVec, 
    ŷ :: RVec, Δ :: Union{Real, RVec} ) :: Nothing
    xᵗ!(id, x̂);
    fxᵗ!(id, ŷ);
    Δᵗ!(id, Δ);

    add_iter_result!(db(id), x̂, ŷ)
    nothing 
end

function keep_current_iterate!( id :: AbstractIterData, x̂ :: RVec, ŷ :: RVec,
      Δ :: Union{Real, RVec}) :: Nothing
    Δᵗ!(id, Δ);
    add_result!(db(id), x̂, ŷ );
    nothing
end

############################################
# AbstractConfig
max_evals( :: AbstractConfig ) :: Int = 100;
max_iter( :: AbstractConfig ) :: Int = 10;

use_db( :: AbstractConfig )::Union{ Type{<:AbstractDB}, Nothing } = nothing;

# initial radius
Δ⁰(::AbstractConfig)::Union{RVec, Real} = 0.1;

# radius upper bound(s)
Δᵘ(::AbstractConfig)::Union{RVec, Real} = 0.5;

descent_method( :: AbstractConfig )::Symbol = :steepest_descent

"Require a descent in all model objective components. 
Applies only to backtracking descent steps, i.e., :steepest_descent."
strict_backtracking( :: AbstractConfig )::Bool = true;

strict_acceptance_test( :: AbstractConfig )::Bool = true;
ν_success( :: AbstractConfig )::Real = 0.4;
ν_accept(::AbstractConfig)::Real = 0.0;

μ(::AbstractConfig) = 2e3;
β(::AbstractConfig) = 1e3;

radius_update_method(::AbstractConfig)::Symbol = :standard;
γ_grow(::AbstractConfig)::Real = 2;
γ_shrink(::AbstractConfig)::Real = Float16(.75);
γ_shrink_much(::AbstractConfig)::Real=Float16(.501);

############################################
# Implementations

####### MockDB
struct NoDB <: AbstractDB end 
init_db( :: Type{NoDB} ) = NoDB();

####### ArrayDB
@with_kw mutable struct ArrayDB <: AbstractDB
    sites :: RVecArr = RVec[];   # could also use a Dict here …
    values :: RVecArr = RVec[]; # … but what about equal sites; also `push!` is easy
    x_index :: Int = 0;

    iter_indices :: Int = [];
end

init_db( :: Type{ArrayDB} ) = ArrayDB();
sites( db :: ArrayDB ) = db.sites :: RVecArr;
values( db :: ArrayDB ) = db.values :: RVecArr;
xᵗ_index( db :: ArrayDB ) = db.x_index :: Int;

function stamp_x_index!(db :: ArrayDB)::Nothing
    push!(db.iter_indices,x_index)
    nothing
end

function xᵗ_index!( db :: ArrayDB, N :: Int ) :: Nothing
    db.x_index = N
    nothing
end

function add_site!( db :: ArrayDB, x̂ :: RVec ) :: Int
    push!( db.sites, x̂ )
    return length( sites(db) )
end

function add_value!( db :: ArrayDB, ŷ :: RVec ) :: Int
    push!( db.values, ŷ )
    return length( sites(db) )
end

function change_site!( db :: ArrayDB, pos :: Int, new_site :: RVec) :: Nothing
    db.sites[pos][:] = new_site;
    nothing 
end

function change_value!( db :: ArrayDB, pos :: Int, new_val :: RVec) :: Nothing
    db.values[pos][:] = new_val;
    nothing 
end

####### IterData
@with_kw mutable struct IterData <: AbstractIterData
    sites :: RVecArr = RVec[];
    values :: RVecArr = RVec[];
    x :: Union{Nothing,RVec} = nothing;
    fx :: Union{Nothing, RVec} = nothing;
    Δ :: Union{Nothing,Real} = nothing;
    db :: Union{Nothing,AbstractDB} = nothing;    
end

xᵗ( id :: IterData ) = id.x :: Union{RVec, Nothing};
fxᵗ( id :: IterData ) = id.fx :: Union{RVec, Nothing};
Δᵗ( id :: IterData ) = id.Δ :: Union{Real, Nothing};
db( id :: IterData ) = id.db :: Union{AbstractDB, Nothing};

# setters
function xᵗ!( id :: IterData, x̂ :: RVec ) :: Nothing 
    id.x = x̂;
    nothing
end

function fxᵗ!( id :: IterData, ŷ :: RVec ) :: Nothing 
    id.fx = ŷ;
    nothing
end

function Δᵗ!( id :: IterData, Δ :: Union{Real, RVec} ) :: Nothing
    id.Δ = Δ;
    nothing
end

function init_iter_data( ::Type{IterData}, x :: RVec, fx :: RVec, Δ :: Union{Real, RVec}, 
    db :: Union{AbstractDB,Nothing}) :: IterData
    IterData(; x = x, fx = fx, Δ = Δ, db = db);
end

struct EmptyConfig <: AbstractConfig end;
global const empty_config = EmptyConfig();
use_db( ::EmptyConfig ) = ArrayDB;

@with_kw struct AlgoConfig <: AbstractConfig
    max_evals :: Int = max_evals( empty_config )
    max_iter :: Int = max_iter( empty_config );
    
    Δ_0 :: Union{Real, RVec} = Δ⁰(empty_config);
    Δ_max :: Union{Real, RVec } = Δᵘ(empty_config);
    
    descent_method :: Symbol = descent_method(empty_config);
    strict_backtracking :: Bool = strict_backtracking(empty_config);

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
end

max_evals( ac :: AlgoConfig ) = ac.max_evals;
max_iter( ac :: AlgoConfig ) = ac.max_iter;
Δ⁰( ac :: AlgoConfig ) = ac.Δ_0;
Δᵘ( ac :: AlgoConfig ) = ac.Δ_max;
use_db( ac :: AlgoConfig ) = ac.db;
descent_method( ac :: AlgoConfig ) = ac.descent_method;
strict_backtracking( ac :: AlgoConfig ) = ac.strict_backtracking;
strict_acceptance_test( ac :: AlgoConfig ) = ac.strict_acceptance_test;

ν_success( ac :: AlgoConfig ) = ac.ν_success;
ν_accept( ac :: AlgoConfig ) = ac.ν_accept;

μ( ac :: AlgoConfig ) = ac.μ;
β( ac :: AlgoConfig ) = ac.β;

radius_update_method( ac :: AlgoConfig )::Symbol = ac.radius_update_method;
γ_grow(ac :: AlgoConfig)::Real = ac.γ_grow;
γ_shrink(ac :: AlgoConfig)::Real = ac.γ_shrink;
γ_shrink_much(ac :: AlgoConfig)::Real = ac.γ_shrink_much;

function max_evals( objf :: AbstractObjective, ac :: AbstractConfig )
    min( max_evals(objf), max_evals(ac) )
end

include("descent.jl")

function shrink_radius( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return Δ * γ_shrink(ac);
    elseif radius_update_method(ac) == :steplength
        return steplength * γ_shrink(ac);
    end
end
function shrink_radius_much( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return Δ * γ_shrink_much(ac);
    elseif radius_update_method(ac) == :steplength
        return steplength * γ_shrink_much(ac);
    end
end
function grow_radius( ac :: AbstractConfig, Δ :: Real, steplength :: Real) :: Real
    if radius_update_method(ac) == :standard
        return min( Δᵘ(ac), γ_grow(ac) * Δ )
    elseif radius_update_method(ac) == :steplength
        return min( Δᵘ(ac), ( γ_grow + steplength/Δ ) * Δ );
    end
end

############################################
function optimize( mop :: AbstractMOP, x⁰ :: RVec, 
    fx⁰ :: RVec = Real[]; algo_config :: AbstractConfig = EmptyConfig(), 
    populated_db :: Union{AbstractDB,Nothing} = nothing )

    # parse fix configuration parameters
    ν_succ = ν_success( algo_config );
    ν_acc = ν_accept( algo_config );
    mu = μ( algo_config );
    beta = β( algo_config );

    # TODO warn here 
    reset_evals!( mop );

    # initialize first iteration site
    @assert !isempty( x⁰ );
    x = scale( x⁰, mop );
    
    # initalize first objective vector 
    if isempty( fx⁰ )
        # if no starting function value was provided, eval objectives
        @show fx = eval_all_objectives( mop, x );
    else
        fx = apply_internal_sorting( y, mop );
    end

    # initiliza database
    if !isnothing(populated_db)
        # has a database been provided? if yes, prepare
        data_base = populated_db;
        scale!( data_base, mop );
        apply_internal_sorting( data_base, mop );
    else
        data_base = init_db(use_db(algo_config));
    end
    ensure_contains_result!(data_base, x, fx);
    stamp!(data_base);
    @assert data_base isa AbstractDB;

    iter_data = init_iter_data(IterData, x, fx, Δ⁰(algo_config), data_base);

    # initialize surrogate models
    sc = init_surrogates( mop );
    sc.surrogates

    IMPROVEMENT_STEP_FLAG = false;
    for i = 1 : max_iter( algo_config )
        @info "Iteration $(i)."
        @show Δᵗ(iter_data);
        # read iter data to handy variables
        x = xᵗ(iter_data);
        fx = fxᵗ(iter_data);

        if i > 1
            update_surrogates!( sc, mop, iter_data; ensure_fully_linear = false );
        end

        ω, x₊, mx₊, steplength = compute_descent_step( algo_config, mop, iter_data, sc )

        mx = eval_models(sc, x);
        fx₊ = eval_all_objectives(mop, x₊);

        if strict_acceptance_test( algo_config )
            ρ = minimum( (fx .- fx₊) ./ (mx .- mx₊) )
        else
            ρ = (maximum(fx) - maximum( fx₊ ))/(maximum(mx)-maximum(mx₊))
        end
        @show fx 
        @show mx 
        @show fx₊
        @show mx₊
        @show ρ

        ACCEPT_TRIAL_POINT = false
        ρ = isnan(ρ) ? -Inf : ρ;
        old_Δ = Δᵗ(iter_data);
        if ρ >= ν_succ
            if old_Δ < beta * ω
                Δ = grow_radius(algo_config, old_Δ, steplength);
            end
            ACCEPT_TRIAL_POINT = true;
        else
            if fully_linear(sc)
                if ρ < ν_acc
                    Δ = shrink_radius_much(algo_config, old_Δ, steplength);
                else
                    Δ = shrink_radius(algo_config, old_Δ, steplength);
                    ACCEPT_TRIAL_POINT = true;
                end
            else
                IMPROVEMENT_STEP_FLAG = true;
            end
        end

        if ACCEPT_TRIAL_POINT
            set_next_iterate!(iter_data, x₊, fx₊, Δ);
        else
            keep_current_iterate!(iter_data, x₊, fx₊, Δ);
        end

        stamp!(data_base)
        @assert all(isapprox.(x₊,xᵗ(iter_data)))
    end

    # FINISHED :)
    # unscale sites and re-sort values to return to user
    if !isnothing( db(iter_data) )
        unscale!( db(iter_data) , mop);
        reverse_internal_sorting!( db(iter_data), mop);
    end

    return xᵗ(iter_data), fxᵗ(iter_data), iter_data

end# function optimize

end#module