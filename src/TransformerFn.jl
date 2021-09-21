# wrapper to unscale x̂ from internal domain
struct TransformerFn{F}
    lb :: Vector{F}
    ub :: Vector{F}
    w :: Vector{F}
    inf_indices :: Vector{Int}
    not_inf_indices :: Vector{Int}
end

"Return the `TransformerFn` defined by `mop` with a minimum precision of `T`."
function TransformerFn(mop :: AbstractMOP, T :: Type{<:AbstractFloat} = MIN_PRECISION)
    LB, UB = full_bounds( mop )
    W = UB - LB
    I = findall(isinf.(W))
    NI = setdiff( 1 : length(W), I )
    W[ I ] .= 1

    F = Base.promote_eltype( T, W )
    return TransformerFn{F}(LB,UB,W,I,NI)
end

Base.broadcastable( tfn :: TransformerFn ) = Ref(tfn)

using LinearAlgebra: diagm 
function _jacobian_unscaling( tfn :: TransformerFn, x̂ :: Vec)
    # for our simple bounds scaling the jacobian is diagonal.
    return diagm(tfn.w)
end

"Unscale the point `x̂` from internal to original domain."
function (tfn:: TransformerFn)( x̂ :: AbstractVector{<:Real} )
    χ = copy(x̂)
    I = tfn.not_inf_indices
    χ[I] .= tfn.lb[I] .+ tfn.w[I] .* χ[I] 
    return χ
end

unscale( tfn :: TransformerFn, x̂ :: Vec ) = tfn(x̂)

function scale( tfn :: TransformerFn, x :: Vec )
    χ = copy( x )
    I = tfn.not_inf_indices;
    χ[I] .= ( χ[I] .- tfn.lb[I] ) ./ tfn.w[I]
    return χ
end
