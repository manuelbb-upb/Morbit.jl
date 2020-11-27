inbounds(x) = all( 0 .<= x .<= 1 )
intobounds(x, ::Val{true}) = min.( max.(0, x), 1)  # project x into unit hypercube
intobounds(x, ::Val{false}) = x
intobounds(x, lb, ub) = min.( max.(lb, x), ub)  # project x into unit hypercube

@doc "Effective boundaries taking global constraints and Δ into account."
function effective_bounds_vectors( x :: Vector{R}, Δ :: Float64 , lb :: Union{R, Vector{R}},
        ub :: Union{R, Vector{R}} ) where{R<:Real}
    lb_eff = max.( lb, x .- Δ )
    ub_eff = min.( ub, x .+ Δ )
    return lb_eff, ub_eff
end

@doc "Return arrays `lb_eff` and `ub_eff` that provide effective lower bounds
on the current trust region while assuming unit hypercube boundaries."
function effective_bounds_vectors(  x :: Vector{R}, Δ :: Float64, constrained::Val{true} ) where{R<:Real}
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ, 0.0, 1.0 )
    return lb_eff, ub_eff
end

@doc "Return arrays `lb_eff` and `ub_eff` that provide effective lower bounds
on the current trust region for unconstrained problems."
function effective_bounds_vectors(x :: Vector{R}, Δ :: Float64, constrained::Val{false} ) where{R<:Real}
    return x .- Δ, x  .+ Δ
end

@doc "Find the scalars σ₊,σ₋ that so that x + σ±*d intersects the variable boundaries."
function intersect_bounds(x :: T, d :: T, lb :: T, ub :: T ) where{T<:Vector{Float64}}
    non_zero = (d .!= 0.0); # TODO tolerance
    if !any(non_zero)
        return Inf, -Inf
    else
        ε = 1e-18;
        σ_lb = (lb[non_zero] .+ ε .- x[non_zero]) ./ d[non_zero];
        σ_ub = (ub[non_zero] .- ε .- x[non_zero]) ./ d[non_zero];

        smallest_largest = sort( [σ_lb σ_ub], dims = 2 );    # first column contains the smallest factor we are allowed to move along each coordinate, second the largest factor

        σ_pos = minimum( smallest_largest[:, 2] );
        σ_neg = maximum( smallest_largest[:, 1] );

        return σ_pos, σ_neg
    end
end

function intersect_bounds( x :: Array{Float64,1}, d :: Array{Float64,1},
        Δ :: Float64, constrained_flag :: Bool = true )
    lb, ub = effective_bounds_vectors( x, Δ , Val(constrained_flag));
    intersect_bounds(x, d, lb, ub)
end
