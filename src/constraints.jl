inbounds(x) = all( 0 .<= x .<= 1 )
intobounds(x, ::Val{true}) = min.( max.(0, x), 1)  # project x into unit hypercube
intobounds(x, ::Val{false}) = x
intobounds(x, lb, ub) = min.( max.(lb, x), ub)  # project x into unit hypercube

@doc "Effective boundaries taking global constraints and Δ into account."
function effective_bounds_vectors( x :: Vector{R}, Δ :: Real , lb :: Union{Real, Vector{L}},
        ub :: Union{Real, Vector{U}} ) where{R<:Real, L<:Real,U<:Real}
    lb_eff = max.( lb, x .- Δ )
    ub_eff = min.( ub, x .+ Δ )
    return lb_eff, ub_eff
end

@doc "Return arrays `lb_eff` and `ub_eff` that provide effective lower bounds
on the current trust region while assuming unit hypercube boundaries."
function effective_bounds_vectors(  x :: Vector{R}, Δ :: Real, constrained::Val{true} ) where{R<:Real}
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ, 0, 1 )
    return lb_eff, ub_eff
end

@doc "Return arrays `lb_eff` and `ub_eff` that provide effective lower bounds
on the current trust region for unconstrained problems."
function effective_bounds_vectors(x :: Vector{R}, Δ :: Real, constrained::Val{false} ) where{R<:Real}
    return x .- Δ, x  .+ Δ
end

@doc "Find the scalars σ₊,σ₋ that so that x + σ±*d intersects the variable boundaries."
function intersect_bounds(x ::Vector{R} where R<:Real, d::Vector{R} where R<:Real,
        lb::Vector{R} where R<:Real, ub::Vector{R} where R<:Real 
    )
    non_zero = (d .!= 0); # TODO tolerance
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

function intersect_bounds( x :: Vector{R}, d :: Vector{D},
        Δ :: Real, constrained_flag :: Bool = true ) where{R<:Real, D<:Real}
    lb, ub = effective_bounds_vectors( x, Δ , Val(constrained_flag));
    intersect_bounds(x, d, lb, ub)
end
