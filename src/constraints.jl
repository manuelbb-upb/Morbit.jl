

@doc "Effective boundaries taking global constraints and Δ into account."
function effective_bounds_vectors( x, Δ, lb, ub )
    lb_eff = max.( lb, x .- Δ )
    ub_eff = min.( ub, x .+ Δ )
    return lb_eff, ub_eff
end

function effective_bounds_vectors( x, Δ )
    # assume lb = zeros, ub = ones
    ε_bounds = 0.0;
    n_vars = length(x);
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ, zeros( n_vars ) .+ ε_bounds, ones( n_vars ) .- ε_bounds)
    return lb_eff, ub_eff
end

inbounds(x) = all( 0 .<= x .<= 1 )
intobounds(x) = min.( max.(0, x), 1)  # project x into unit hypercube
intobounds(x, lb, ub) = min.( max.(lb, x), ub)  # project x into unit hypercube

@doc "Find the scalars σ₊,σ₋ that so that x + σ±*d intersects the variable boundaries."
function intersect_bounds(x :: T,d :: T,lb::T,ub::T ) where {T}# = Array{Float64,1}}
    non_zero = (d .!= 0.0); # TODO tolerance
    if !any(non_zero)
        return Inf, -Inf
    else
    #    x = intobounds(x, lb, ub);  # sometimes during iteration, x is outside of global box constraints by ε and then the routine below fails

        σ_lb = (lb[non_zero] .- x[non_zero]) ./ d[non_zero];
        σ_ub = (ub[non_zero] .- x[non_zero]) ./ d[non_zero];

        smallest_largest = sort( [σ_lb σ_ub], dims = 2 );    # first column contains the smallest factor we are allowed to move along each coordinate, second the largest factor

        σ_pos = minimum( smallest_largest[:, 2] );
        σ_neg = maximum( smallest_largest[:, 1] );

        return σ_pos, σ_neg
    end
end

function intersect_bounds( x :: Array{Float64,1}, d :: Array{Float64,1}, Δ :: Float64 )
    lb, ub = effective_bounds_vectors( x, Δ );
    intersect_bounds(x , d, lb, ub)
end

@doc "Scale a value vector via min-max-scaling."
function scale( id :: IterData, y )
    return ( y .- id.min_value ) ./( id.max_value .- id.min_value )
end
