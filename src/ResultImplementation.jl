
@with_kw struct Result{XT <: VecF, YT <: VecF} <: AbstractResult{XT,YT}
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
    # `if-else` for if `res.y` is an MVector
    if length(res.y) == length(y)
        res.y[:] .= y[:]
    else
        empty!(res.y)
        append!(res.y, y)
    end
    return nothing 
end

function init_res( T::Type{<:Result}, id :: Int, x :: Vec, y :: AbstractVector = [])
    _y = isempty(y) ? empty_value( T ) : y
    return T( x, _y, id )
end