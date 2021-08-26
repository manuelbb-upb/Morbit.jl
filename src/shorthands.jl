const RVec = AbstractVector{R} where R<:Real;
const RVecArr = AbstractVector{<:RVec};
const RMat = AbstractMatrix{R} where R <: Real;

#=
const Vec = AbstractVector{<:AbstractFloat}
const VecVec = AbstractVector{<:Vec}
const NumOrVec = Union{AbstractFloat, Vec}
const VecOrNum = NumOrVec
const Mat = AbstractMatrix{<:AbstractFloat}
=#
const Vec = AbstractVector{<:Real}
const VecVec = AbstractVector{<:AbstractVector}
const NumOrVec = Union{Real, Vec}
const VecOrNum = NumOrVec 
const Mat = AbstractMatrix{<:Real}

const MIN_PRECISION = Float32

struct XInt <: Integer 
    val :: Union{Int,Nothing}
end 

NothInt = Union{Nothing,Integer};
NothIntVec = Vector{<:NothInt};

Base.vec(x::Real) = [x,];

Base.convert( ::Type{<:Int}, xint :: XInt ) = isnothing(xint.val) ? -1 : xint.val;
Base.:(==)( i :: Int, xint :: XInt ) = isnothing(xint.val) ? false : xint.val == i;
Base.:(==)(x :: XInt, i :: Int) = i == x;