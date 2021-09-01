const Vec = AbstractVector{<:Real}
const VecVec = AbstractVector{<:AbstractVector}
const NumOrVec = Union{Real, Vec}
const VecOrNum = NumOrVec 
const Mat = AbstractMatrix{<:Real}

const Vec64 = Vector{Float64}
const NumOrVec64 = Union{Float64, Vec64}

const VecF = AbstractVector{<:AbstractFloat}
const NumOrVecF = Union{AbstractFloat, VecF}
const MIN_PRECISION = Float32

ensure_vec(x :: AbstractVector ) = x 
ensure_vec(x :: Number) = [x,]
