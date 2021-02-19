module Morbit

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
xᵗ_index!(::AbstractDB, ::Int) = nothing :: Nothing;

add_site!( :: AbstractDB, :: RVec ) = nothing :: Int;
add_value!( :: AbstractDB, :: RVec ) = nothing :: Int;

Base.eachindex( db :: AbstractDB ) = eachindex( sites(db) );

change_site!( :: AbstractDB, :: Int, :: RVec) = nothing :: Nothing;
change_value!( :: AbstractDB, :: Int, :: RVec) = nothing :: Nothing;

get_site( db :: AbstractDB, pos :: Int) = sites(db)[pos] :: RVec;
get_value( db :: AbstractDB, pos :: Int) = values(db)[pos] :: RVec;

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
    x_pos = add_result!(db, x̂, ŷ);
    xᵗ_index!(db, x_pos);
    return site_id; 
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

function update!( id :: AbstractIterData, x̂ :: RVec, 
    ŷ :: RVec, Δ :: Union{Real, RVec} ) :: Nothing
    xᵗ!(id, x̂);
    fxᵗ!(id, ŷ);
    Δᵗ!(id, Δ);

    add_iter_result!(db, x̂, ŷ)
    nothing 
end

############################################
# AbstractConfig
max_evals( :: AbstractConfig ) = typemax(Int) :: Int;
max_iter( :: AbstractConfig ) = 10 :: Int;

use_db( :: AbstractConfig ) = nothing :: Union{ Type{<:AbstractDB}, Nothing };

# initial radius
Δ⁰(::AbstractConfig) = 0.1 :: Union{RVec, Real};

# radius upper bound(s)
Δᵘ(::AbstractConfig) = 0.5 :: Union{RVec, Real};

############################################
# Implementations

####### ArrayDB
@with_kw mutable struct ArrayDB <: AbstractDB
    sites :: RVecArr = RVec[];   # could also use a Dict here …
    values :: RVecArr = RVec[]; # … but what about equal sites; also `push!` is easy
    x_index :: Int = 0;
end

init_db( :: Type{ArrayDB} ) = ArrayDB();
sites( db :: ArrayDB ) = db.sites :: RVecArr;
values( db :: ArrayDB ) = db.values :: RVecArr;
xᵗ_index( db :: ArrayDB ) = db.x_index :: Int;

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
    id.x = ŷ;
    nothing
end

function Δᵗ!( id :: IterData, Δ :: Union{Real, Nothing} ) :: Nothing
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
    
    db :: Union{Nothing,Type{<:AbstractConfig}} = ArrayDB;
end

max_evals( ac :: AlgoConfig ) = ac.max_evals;
max_iter( ac :: AlgoConfig ) = ac.max_iter;
Δ⁰( ac :: AlgoConfig ) = ac.Δ_0;
Δᵘ( ac :: AlgoConfig ) = ac.Δ_max;
use_db( ac :: AlgoConfig ) = ac.db;

function max_evals( objf :: AbstractObjective, ac :: AbstractConfig )
    min( max_evals(objf), max_evals(ac) )
end
    
############################################
function optimize( mop :: AbstractMOP, x⁰ :: RVec, 
    fx⁰ :: RVec = Real[], ac :: AbstractConfig = EmptyConfig(), 
    populated_db :: Union{AbstractDB,Nothing} = nothing )
    
    # TODO warn here 
    reset_evals!( mop );

    # initialize first iteration site
    @assert !isempty( x⁰ );
    x = scale( x⁰, mop );
    
    # initalize first objective vector 
    if isempty( fx⁰ )
        # if no starting function value was provided, eval objectives
        fx = eval_all_objectives( mop, x⁰ );
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
        data_base = init_db(use_db(ac));
    end
    ensure_contains_result!(data_base, x, fx);

    iter_data = init_iter_data(IterData, x, fx, Δ⁰(ac), data_base);

    sc = init_surrogates( mop );

    # unscale sites and re-sort values to return to user
    if !isnothing( db(iter_data) )
        unscale!( db(iter_data) , mop);
        reverse_internal_sorting!( db(iter_data), mop);
    end

    return xᵗ(iter_data), fxᵗ(iter_data), iter_data

end# function optimize

end#module