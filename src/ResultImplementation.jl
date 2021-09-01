
@with_kw mutable struct Result{XT <: VecF, YT <: VecF} <: AbstractResult{XT,YT}
    x :: XT = Float64[]
    y :: YT = Float64[]
    db_id :: Int = -1
end

# TODO: do i use this somewhere?
function Base.:(==)(r1 :: Result, r2::Result)
    return (
        r1.x == r2.x &&
        r1.y == r2.y &&
        r1.db_id == r2.db_id
    )
end

# mandatory implementations
get_site( res :: Result ) = res.x
get_value( res :: Result ) = res.y
get_id( res :: Result ) = res.db_id

function set_site!( res :: Result, x )
    res.x[:] .= x[:]
    return nothing 
end

function set_value!( res :: Result, y )
    empty!(res.y)
    append!(res.y, y)
    return nothing 
end

function init_res( T::Type{<:Result{XT,YT}} , x :: Vec, y :: Vec, id :: Int ) where{XT,YT}
    return T( x, y, id )
end