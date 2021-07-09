
@with_kw mutable struct Result{F <: AbstractFloat} <: AbstractResult{F}
    x :: Vector{F} = F[]
    y :: Vector{F} = F[]
    db_id :: Int = -1
end

Base.length(::Result) = 1
Base.iterate(r :: Result) = (r, nothing)
Base.iterate(::Result, ::Nothing) = nothing

function Base.:(==)(r1 :: Result, r2::Result)
    return (
        r1.x == r2.x &&
        r1.y == r2.y &&
        r1.db_id == r2.db_id
    )
end

get_site( res :: Result ) = res.x;
get_value( res :: Result ) = res.y;
get_id( res :: Result ) = res.db_id;

function init_res( ::Type{<:Result{F}} , x :: Vec, y :: Vec, id :: Int ) where F
    return Result{F}( x, y, id )
end

struct NoRes{F} <: AbstractResult{F} end