using LinearAlgebra: I, norm, qr
using Parameters: @with_kw

function _orthogonal_complement_matrix( Y, p = Inf )
    Q, _ = qr(Y)
    Z = Q[:, size(Y,2) + 1 : end]
    if size(Z,2) > 0
        Z ./= norm.( eachcol(Z), p )'
    end
    return Z
end 

# stateful (Y,Z are stored so that I have access to them later)
@with_kw mutable struct AffinelyIndependentPointFilter{
	F <: AbstractFloat, 
	VF <: AbstractVector{F}, 
	SV <: AbstractVector{<:AbstractVector{F}},
	SSV <: AbstractVector{<:AbstractVector{F}}, 
	I<:Real
	}
	x_0 :: VF

	seeds :: SV = Vector{VF}
	
	shifted_seeds :: SSV = [ s .- x_0 for s in seeds]

	candidate_indices :: Vector{Int} = collect(eachindex(seeds))

	n :: Int = length(x_0)
	
	Y :: Matrix{F} = Matrix{eltype(x_0)}(undef, n, 0)
	Z :: Matrix{F} = Matrix{eltype(x_0)}(I(n))

	p :: I = Inf 	# which vector norm to use

	pivot_val :: F = 1e-3

	return_indices :: Bool = false

	@assert n > 0 "`x_0` must not be empty and `n` must be positive."
	@assert n <= length(x_0) "Maximum number must be lower than `length(x_0)`."
end

function reset!( filter :: AffinelyIndependentPointFilter{F,VF,SF} ) where{F,VF,SF}
	filter.candidate_indices = collect(eachindex(filter.seeds))
	filter.Y = Matrix{F}(undef, filter.n, 0)
	filter.Z = Matrix{F}(I(filter.n))
	return nothing
end

function Base.iterate( filter :: AffinelyIndependentPointFilter{F,VF,SV} ) :: Union{Nothing, Tuple{<:Union{Int,VF}, Int}} where{F,VF,SV}
	isempty(filter.shifted_seeds) && return nothing
	_, i = findmax( norm.(filter.shifted_seeds, filter.p) )

	if length(filter.candidate_indices) != length(filter.seeds)
		reset!(filter)
	end

	filter.Y = hcat(filter.Y, filter.shifted_seeds[i])
	filter.Z = _orthogonal_complement_matrix( filter.Y, filter.p )

	setdiff!(filter.candidate_indices, i)

	if filter.return_indices
		return i, 1
	else
		return filter.seeds[i], 1
	end
end

function Base.iterate( filter :: AffinelyIndependentPointFilter{F,VF,SV}, num_found :: Int ) :: Union{Nothing,Tuple{<:Union{VF,Int}, Int}} where{F,VF,SV}
	num_found == filter.n && return nothing			# found enough points already
	isempty(filter.candidate_indices) && return nothing	# no more points to search

	Z = filter.Z

	best_val = -F(Inf)
	best_index = -1

	for i ∈ filter.candidate_indices
		x = filter.shifted_seeds[i]
		proj_x_Z = norm( Z*(Z'*x), filter.p )

		if proj_x_Z > best_val 
			best_val = proj_x_Z
			best_index = i 
			# NOTE we could check `proj_x_Z > filter.pivot_val` and break early here instead of finding the maximizer
		end
	end

	if best_val > filter.pivot_val
		i = best_index
		filter.Y = hcat(filter.Y, filter.shifted_seeds[i])
		filter.Z = _orthogonal_complement_matrix( filter.Y, filter.p )
		setdiff!(filter.candidate_indices, i)

		if filter.return_indices
			return i, num_found + 1
		else
			return filter.seeds[i], num_found + 1
		end
	end

	@debug "No point was sufficiently linearly independent."
	return nothing
end

Base.IteratorSize( ::AffinelyIndependentPointFilter ) = Base.SizeUnknown()
Base.IteratorEltype( :: AffinelyIndependentPointFilter ) = Base.HasEltype()
function Base.eltype( filter :: AffinelyIndependentPointFilter{F,VF,SV} ) where {F,VF,SV}
	if filter.return_indices
		return Int
	else
		return VF 
	end
end
