get_site( :: Result ) :: RVec = Real[];
get_value( :: Result ) :: RVec = Real[];
get_id( :: Result ) :: NothInt = nothing;
change_site!(::Result, :: RVec) :: Nothing = nothing;
change_value!(::Result, :: RVec) :: Nothing = nothing;
change_id!(::Result, :: NothInt) :: Nothing = nothing;
init_res(::Type{<:Result}, args... ) :: Result = nothing;

struct NoRes <:Result end;
init_res(NoRes, args... ) = NoRes();

@with_kw mutable struct Res <: Result
    x :: RVec = Real[];
    y :: RVec = Real[];
    db_id :: NothInt = nothing 
end

get_site( res :: Res ) = res.x;
get_value( res :: Res ) = res.y;
get_id( res :: Res ) = res.db_id;
function change_site!(res ::Res, s :: RVec) :: Nothing 
    res.x = s;
    nothing;
end 
function change_value!(res ::Res, s :: RVec) :: Nothing 
    res.y = s;
    nothing;
end 
function change_id!(res ::Res, s :: NothInt) :: Nothing 
    res.db_id = s;
    nothing;
end 

function init_res( ::Type{Res}, x :: RVec = Real[], y::RVec = Real[], id :: NothInt = nothing ) :: Res
    return Res(; x = x, y = y, db_id = id )
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

get_value( db :: AbstractDB, id :: NothInt ) :: RVec = Real[];
get_site( db :: AbstractDB, id :: NothInt ) :: RVec = Real[];

#=
function get_result( db :: AbstractDB, id :: NothInt ) :: Tuple{RVec, RVec}
    return ( get_site(db, id ), get_value(db, id) )
end
=#

function get_result( db :: AbstractDB, id :: NothInt ) :: Result
    NoRes()
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

function find_result( db :: AbstractDB, target :: Result  ) :: NothInt
    x = get_site(target);
    y = get_value(target);
    for id ∈ eachindex(db)
        res = get_result(db, id)
        if all(get_site(res) .== x) && all(get_value(res) .== y)
            return pos
        end
    end
    return nothing
end

function ensure_contains_result!( db :: AbstractDB, res :: Result) :: NothInt
    x_pos = find_result(db, res);
    if isnothing(x_pos)
        x_pos = add_result!(db, res);
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
values( db :: ArrayDB ) :: RVecArr = [ get_value(res) for res ∈ values(db.res) ];

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
    return get_value(db.res[id]);
end

get_result(db::ArrayDB, id :: Int) = db.res[id];

function add_result!( db :: ArrayDB, res :: Result )
    db.max_id += 1;
    db.res[db.max_id] = res
    change_id!(res, db.max_id)
    return db.max_id
end

function change_site!( db :: ArrayDB, id :: Int, new_site :: RVec) :: Nothing
    change_site!(db.res[id], new_site);
    nothing 
end

function change_value!( db :: ArrayDB, id :: Int, new_val :: RVec) :: Nothing
    change_value!(db.res[id], new_val);
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

    x_index = add_result!(db(id), init_res( Res, x̂, ŷ, nothing));
    xᵗ_index!( id, x_index );

    nothing 
end

function keep_current_iterate!( id :: AbstractIterData, x̂ :: RVec, ŷ :: RVec,
      Δ :: Union{Real, RVec}) :: Nothing
    Δᵗ!(id, Δ);
    add_result!(db(id),init_res( Res, x̂, ŷ, nothing) );
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
function get_result( id :: AbstractIterData, pos :: Union{Nothing,Int})
    get_result( db(id), pos )
end
function get_result( id :: AbstractIterData, pos :: XInt)
    x_res = get_result( db(id), pos.val )

    if isempty(get_site( x_res ) )
        return init_res(Res, xᵗ(id), fxᵗ(id), xᵗ_index(id))
    else
        return x_res
    end
end

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

# modifies first two arguments
function _eval_and_store_new_results!(id :: AbstractIterData, res_list :: Vector{<:Result},
    mop :: AbstractMOP) :: Nothing
    DB = db(id);
    # evaluate at new sites
    # we first collect, so that batch evaluation can be exploited
    unstored_results = [ res for res ∈ res_list if isnothing( get_id( res) ) ];
    @info "We have to evaluate at $(length(unstored_results)) new sites."
    new_vals = eval_all_objectives.(mop, [ get_site(res) for res ∈ unstored_results ]);

    # add to db and modify results to contain new data
    for (res,val) ∈ zip( unstored_results, new_vals )
        change_value!(res, val)
        new_id = add_result!(DB, res)
        #change_id!(res, new_id)
    end
    
end

function _point_in_box( x̂ :: RVec, lb :: RVec, ub :: RVec ) :: Bool 
    return all( lb .<= x̂ .<= ub )
end

"Indices of sites in database that lie in box with bounds `lb` and `ub`."
function find_points_in_box( id :: AbstractIterData, lb :: RVec, ub :: RVec;
    exclude_indices :: Vector{<:NothInt} = NothInt[] ) :: Vector{Int}
    return [i for i = eachindex(db(id)) if i ∉ exclude_indices && 
        _point_in_box(get_site(id, i), lb, ub) ]
end

