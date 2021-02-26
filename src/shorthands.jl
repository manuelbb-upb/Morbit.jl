const RVec = Vector{R} where R<:Real;
const RVecArr = Vector{<:RVec};
const RMat = Array{R, 2} where R <: Real;

struct XInt <: Integer 
    val :: Union{Int,Nothing}
end 

NothInt = Union{Nothing,Integer};
NothIntVec = Vector{<:NothInt};

Base.vec(x::Real) = [x,];

Base.convert( ::Type{<:Int}, xint :: XInt ) = isnothing(xint.val) ? -1 : xint.val;
Base.:(==)( i :: Int, xint :: XInt ) = isnothing(xint.val) ? false : xint.val == i;
Base.:(==)(x :: XInt, i :: Int) = i == x;