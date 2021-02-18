const RVec = Vector{R} where R<:Real;
const FVec = Vector{F} where F<:AbstractFloat;
const RVecArr = Vector{Vector{R}} where R<:Real;
const FVecArr = Vector{Vector{F}} where F<:AbstractFloat;
const RMat = Array{R, 2} where R <: Real;
const FMat = Array{F, 2} where F <: AbstractFloat;

Base.vec(x::Real) = [x,];