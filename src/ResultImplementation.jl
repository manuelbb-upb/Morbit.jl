
@with_kw mutable struct Result{F <: AbstractFloat} <: AbstractResult{F}
    x :: Vector{F} = F[]
    y :: Vector{F} = F[]
    db_id :: Int = -1
end

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

function init_res( ::T , args... ) :: T where T<:Result
    return T( args... )
end

struct NoRes{F} <: AbstractResult{F} end