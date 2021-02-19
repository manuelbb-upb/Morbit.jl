const RVec = Vector{R} where R<:Real;
const RVecArr = Vector{<:RVec};
const RMat = Array{R, 2} where R <: Real;

Base.vec(x::Real) = [x,];