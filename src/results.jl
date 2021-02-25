get_site( :: Result ) :: RVec = Real[];
get_val( :: Result ) :: RVec = Real[];
get_id( :: Result ) :: NothInt = nothing;
change_site!(::Result, RVec) :: Nothing = nothing;
change_val!(::Result, RVec) :: Nothing = nothing;

@with_kw mutable struct Res <: Result
    x :: RVec = Real[];
    y :: RVec = Real[];
    db_id :: NothInt = nothing 
end

get_site( res :: Res ) = res.x;
get_val( res :: Res ) = res.y;
get_id( res :: Res ) = res.db_id;
function change_site!(res ::Result, s :: RVec) :: Nothing 
    res.x = s;
    nothing;
end 
function change_val!(res ::Result, s :: RVec) :: Nothing 
    res.y = s;
    nothing;
end 

############################################
# AbstractDB

Broadcast.broadcastable( db :: AbstractDB ) = Ref( db );

Base.length( db :: AbstractDB ) = length(sites(db)) :: Int;

init_db( :: Type{<:AbstractDB} ) = nothing :: AbstractDB;
init_db( :: Nothing ) = nothing :: Nothing;

# array of all evaluation sites  (for end user, internally use Results)
sites( :: AbstractDB ) = RVec[] :: RVecArr;
values( :: AbstractDB ) = RVec[] :: RVecArr;

Base.eachindex( db :: AbstractDB ) :: NothIntVec = Int[];

get_value( db :: AbstractDB, id :: Nothing ) :: RVec = Real[];
get_site( db :: AbstractDB, id :: Nothing ) :: RVec = Real[];

function get_site( db :: AbstractDB, id :: Int) :: RVec
end

function get_value( db :: AbstractDB, id :: Int) :: RVec
end

#=
function get_result( db :: AbstractDB, id :: NothInt ) :: Tuple{RVec, RVec}
    return ( get_site(db, id ), get_value(db, id) )
end
=#

function get_result( db :: AbstractDB, id :: NothInt ) :: Result
    nothing
end

function stamp!(db::AbstractDB) ::Nothing 
    nothing 
end

function add_result!( db :: AbstractDB, res :: Result ) :: NothInt 
    nothing
end

function change_site!( db :: AbstractDB, id :: NothInt, x :: RVec ) :: Nothing end;
function change_value!( db :: AbstractDB, id :: NothInt, y :: RVec ) :: Nothing end;

function change_result!( db :: AbstractDB, id :: NothInt, 
    new_site :: RVec, new_value :: RVec) :: Nothing 
    change_site!(db, id, new_site);
    change_value!(db, id, new_value);
    nothing
end

function find_result( db :: AbstractDB, x :: RVec, y :: RVec ) :: NothInt
    for id ∈ eachindex(db)
        res = get_result(db, id)
        if all(get_site(res) .== x) && all(get_val(res) .== y)
            return pos
        end
    end
    return nothing
end

# TODO also use Tuples here instead of x&y
function contains_result( db :: AbstractDB, x :: RVec, y :: RVec ) :: Bool 
    return !isnothing(find_result(db, x, y))
end

function ensure_contains_result!( db :: AbstractDB, x :: RVec, y :: RVec ) :: NothInt
    x_pos = find_result(db, x, y);
    if isnothing(x_pos)
        x_pos = add_result!(db, Res( x, y, nothing))
    end
    x_pos
end

function scale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for id ∈ eachindex(db)
        x = get_site( db, id);
        change_site!( db, id, scale(x, mop) )
    end
    nothing
end

function unscale!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for id ∈ eachindex(db)
        x̂ = get_site( db, id);
        change_site!( db, id, unscale(x̂, mop) )
    end
    nothing
end

function apply_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing
    for id ∈ eachindex(db)
        y = get_value( db, id );
        change_value!( db, id, apply_internal_sorting( y, mop ) );
    end
    nothing 
end

function reverse_internal_sorting!( db :: AbstractDB, mop :: AbstractMOP ) :: Nothing 
    for id ∈ eachindex(db)
        ŷ = get_value(db, id)
        change_value!( db, id, reverse_internal_sorting( ŷ, mop) )
    end
    nothing  
end

############################################
# Implementations

####### MockDB
struct NoDB <: AbstractDB end 
init_db( :: Type{NoDB} ) = NoDB();

####### ArrayDB
@with_kw mutable struct ArrayDB <: AbstractDB
    res :: Dict{Int, Result} = Dict{Int,Result}();
    max_id = 0;
    x_index :: Int = 0;

    iter_indices :: Vector{Int} = [];
end
Base.length(db :: ArrayDB) = length(db.res);
Base.eachindex(db :: ArrayDB ) = collect(keys(db.res));
init_db( :: Type{ArrayDB} ) = ArrayDB();
sites( db :: ArrayDB ) :: RVecArr = [ get_site(res) for res ∈ values(db.res) ];
values( db :: ArrayDB ) :: RVecArr = [ get_val(res) for res ∈ values(db.res) ];

#=
function stamp!(db :: ArrayDB )::Nothing
    push!(db.iter_indices,xᵗ_index(db));
    nothing
end
=#

function get_site( db :: ArrayDB, id :: Int )
    return get_site(db.res[id]);
end

function get_value( db :: ArrayDB, id :: Int)
    return get_val(db.res[id]);
end

get_result(db::ArrayDB, id :: Int) = db.res[id];

function add_result!( db :: ArrayDB, res :: Result )
    db.max_id += 1;
    db.res[db.max_id] = res
    return db.max_id
end

function change_site!( db :: ArrayDB, id :: Int, new_site :: RVec) :: Nothing
    change_site!(db.res[id], new_site);
    nothing 
end

function change_value!( db :: ArrayDB, id :: Int, new_val :: RVec) :: Nothing
    change_val!(db.res[id], new_val);
    nothing 
end

############################################

Broadcast.broadcastable( id :: AbstractIterData ) = Ref( id );

############################################
# AbstractIterData

# current iterate and values
xᵗ( :: AbstractIterData ) = nothing :: RVec;
fxᵗ( :: AbstractIterData ) = nothing :: RVec;
# trust region radius
Δᵗ( :: AbstractIterData ) = nothing :: Union{Real, RVec};

db( :: AbstractIterData ) = nothing :: Union{AbstractDB,Nothing};

function xᵗ_index( :: AbstractIterData ) :: NothInt XInt(nothing) end;
function xᵗ_index!( :: AbstractIterData, :: Int ) :: Nothing nothing end;

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

    x_index = add_result!(db(id), Res( x̂, ŷ, nothing ) );
    xᵗ_index!( id, x_index );
    nothing 
end

function keep_current_iterate!( id :: AbstractIterData, x̂ :: RVec, ŷ :: RVec,
      Δ :: Union{Real, RVec}) :: Nothing
    Δᵗ!(id, Δ);
    add_result!(db(id),Res( x̂, ŷ, nothing ) );
    nothing
end

####### IterData
@with_kw mutable struct IterData <: AbstractIterData
    x :: Union{Nothing,RVec} = nothing;
    fx :: Union{Nothing, RVec} = nothing;
    Δ :: Union{Nothing,Real} = nothing;
    db :: Union{Nothing,AbstractDB} = nothing;    
    x_index :: NothInt = XInt(nothing);
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

function xᵗ_index( id:: IterData ) :: NothInt
    id.x_index 
end;
function xᵗ_index!( id :: IterData, N :: XInt ) :: Nothing 
    id.x_index = N;
    nothing
end;
function xᵗ_index!( id :: IterData, N :: Nothing ) :: Nothing
    id.x_index = XInt(N);
    nothing
end;
function xᵗ_index!( id :: IterData, N :: Int ) :: Nothing
    id.x_index = XInt(N);
    nothing
end;

function init_iter_data( ::Type{IterData}, x :: RVec, fx :: RVec, Δ :: Union{Real, RVec}, 
    db :: Union{AbstractDB,Nothing}) :: IterData
    IterData(; x = x, fx = fx, Δ = Δ, db = db);
end

# special retrieval commands for XInt (if no db is kept)
get_value(id::AbstractIterData,pos::Union{Nothing,Int}) = get_value(db(id), pos)
get_value(id::AbstractIterData, pos::XInt) = fxᵗ(id);

get_site(id::AbstractIterData,pos::Union{Nothing,Int}) = get_site(db(id), pos)
get_site(id::AbstractIterData, pos::XInt) = xᵗ(id);

function get_sites(id::AbstractIterData, positions :: NothIntVec)
    [ s for s ∈ get_site.(id, positions) if !isempty(s) ]
end

function get_values(id :: AbstractIterData, positions :: NothIntVec )
    [ s for s ∈ get_value.(id, positions) if !isempty(s) ]
end