module PointSampler
using LinearAlgebra: norm

export monte_carlo_th, MonteCarloThDesign

@doc """
Return the minium projected distance between two point vectors `p1` and `p2`,
i.e. the minimum absolute value of the differences of the coordinates of `p1` and `p2`.
"""
function projected_distance( p1 :: Vector{Float64}, p2 :: Vector{Float64} )
    minimum( abs.( p1 .- p2) )
end

@doc "Return array of projected distances for point `p1` against every point in `P`."
function projected_distance( p1 :: Vector{Float64}, P :: Vector{Vector{Float64}} )
    [ projected_distance(p1, p2) for p2 ∈ P ]
end

@doc "Return array of projected distances for point `p1` against every point in `P`, but every value below `threshold` is set to `0.0`."
function projected_distance_thresholded( p1 :: Vector{Float64}, P :: Vector{Vector{Float64}}, threshold :: Float64 = 0.1 )
    pdist_array = projected_distance( p1, P );
    pdist_array[ pdist_array .< threshold ] .= 0.0
    return pdist_array
end

@doc "Return Euclidean distance from `p1` to every point in `P`."
function distance( p1 :: Vector{Float64}, P :: Vector{Vector{Float64}} )
    [ norm(p1 .- p2, 2) for p2 ∈ P ]
end

@doc "Objective function that is meant to be maximized by the sampling prodecure. Returns weighted sum of intersite and projected (thresholded) distance."
function combined_objective( p1 :: Vector{Float64}, P :: Vector{Vector{Float64}}; intersite_factor :: Float64 = 1.0, pdist_factor :: Float64 = 1.0,  pdist_threshold :: Float64 = .1 )
    idist = minimum( distance( p1, P ) );
    pdist = minimum( projected_distance_thresholded(p1, P, pdist_threshold) )

    intersite_factor * idist + pdist_factor * pdist
end

function combined_objective( candidates :: Vector{Vector{Float64}}, P :: Vector{Vector{Float64}}, pdist_threshold_tolerance :: Float64 = .5 )
    N = length(P);
    d = length(P[1]);

    intersite_factor = (( N + 1 )^( 1/d ) - 1)/2
    pdist_factor = (N+1)/2;

    pdist_threshold = 2 * pdist_threshold_tolerance / N
    [ combined_objective(p1, P; intersite_factor = intersite_factor, pdist_factor = pdist_factor, pdist_threshold = pdist_threshold ) for p1 ∈ candidates ]
end

function bad_indices( set :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub = Vector{Float64} )
    [ any( s .< lb ) || any( s .> ub ) for s ∈ set]
end

function good_indices( set :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub = Vector{Float64} )
    [ all( s .>= lb ) && any( s .<= ub ) for s ∈ set]
end

function discard_bad_seeds!( seeds :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub = Vector{Float64} )
    bad_seeds = bad_indices( seeds, lb, ub )
    deleteat!(seeds, bad_seeds)
end

function scale_to_unit_square( p :: Vector{Float64}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    ( p .- lb ) ./ ( ub .- lb )
end

function scale_to_unit_square!( p :: Vector{Float64}, lb :: Vector{Float64}, ub :: Vector{Float64} )
     p .-= lb
     p ./= ( ub .- lb )
     return p
end

function scale_to_unit_square( P :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    [ scale_to_unit_square( p, lb, ub ) for p ∈ P ]
end

function scale_to_unit_square!( P :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    [ scale_to_unit_square!( p, lb, ub ) for p ∈ P ]
end

function unscale_from_unit_square( p :: Vector{Float64}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    lb .+ (ub .- lb) .* p
end

function unscale_from_unit_square!( p :: Vector{Float64}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    p .*= (ub .- lb)
    p .+= lb
end

function unscale_from_unit_square( P :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    [ unscale_from_unit_square(p, lb, ub) for p ∈ P ]
end

function unscale_from_unit_square!( P :: Vector{Vector{Float64}}, lb :: Vector{Float64}, ub :: Vector{Float64} )
    [ unscale_from_unit_square!(p, lb, ub) for p ∈ P ]
end

@doc """
    monte_carlo_th( n_points :: Int64 = 10, n_dims :: Int64 = 2; seeds :: Vector{Vector{Float64}} = [], spawn_factor :: Int64 = 50, pdist_threshold_tolerance :: Float64 = 0.5 )

    Return an array of length `n_points` containing Float64-arrays representing points in space with `n_dims` dimensions.
    The points are iteratively chosen from random point sets to maximize a space-filling criterion as described in

    "Surrogate Modelling of Computer Experiments with Sequential Experimental Design.", Crombecq, 2011

    The returned point set is constructed starting with the points in `seeds`. If `seeds` is empty (default), then the singleton set containing the zero vector is used.
"""
function monte_carlo_th( n_points :: Int64, n_dims :: Int64 ; seeds :: Vector{Vector{Float64}} = Vector{Vector{Float64}}(), spawn_factor :: Int64 = 50, pdist_threshold_tolerance :: Float64 = 0.5 )

    n_seeds = length(seeds);
    if n_seeds > 0
        if n_seeds < n_points
            P = deepcopy(seeds);
            if length(P[1]) != n_dims
                @error "`n_dims` and length of your `seeds` do not match."
            end
        else
            return seeds
        end
    else
        P = [ zeros(Float64, n_dims) ];
        n_seeds = 1;
    end

    for i = n_seeds : n_points - 1
        candidate_points =  [ rand(n_dims) for j = 1 : max( 200, i * spawn_factor )];
        scores = combined_objective( candidate_points, P, pdist_threshold_tolerance );
        best_index = argmax(scores);
        push!( P, candidate_points[best_index] )
    end
    return P
end

@doc "Scale the design returned by the unconstrained version of this function to the box defined by `lb` and `ub`."
function monte_carlo_th( n_points :: Int64, lb = Vector{Float64}, ub = Vector{Float64}; seeds :: Vector{Vector{Float64}} = Vector{Vector{Float64}}(), spawn_factor :: Int64 = 50, pdist_threshold_tolerance :: Float64 = 0.5, clean_seeds ::Bool = true )
    if isempty( lb ) || isempty(ub)
        @error "Both lower and upper variable boundaries must be provided."
    else
        if length(lb) != length(ub)
            @error "`lb` and `ub` must have samle length."
        else
            n_dims = length(lb)
        end
    end
    copied_seeds = deepcopy(seeds);
    if clean_seeds
        discard_bad_seeds!(copied_seeds, lb, ub)
    end
    scale_to_unit_square!(copied_seeds, lb, ub);
    unit_design = monte_carlo_th( n_points, n_dims; seeds = copied_seeds, spawn_factor = spawn_factor, pdist_threshold_tolerance = pdist_threshold_tolerance ); # as of Julia v1.5 keyword arguments could be written in a more concise manner
    P = unscale_from_unit_square( unit_design, lb, ub )
end


function score_arrays( P :: Vector{Vector{Float64}} )
    np = length( P )
    idist_scores = zeros(Float64, np)
    pdist_scores = zeors(Float64, np)
    th = 1 / np
    for pi ∈ 1 : np
        idist_scores[pi] = minimum( distance( point_array[pi], point_array[ [ 1 : pi - 1; pi + 1 : end] ] ) )
        pdist_scores[pi] = minimum( projected_distance_thresholded( point_array[pi], point_array[ [ 1 : pi - 1; pi + 1 : end] ], th ) )
    end
    return idist_scores, pdist_scores
end

struct MonteCarloThDesign
    n_points :: Int64
    dims :: Int64
    lb :: Vector{Float64}
    ub :: Vector{Float64}
    seeds :: Vector{Vector{Float64}}
end

function is_valid( des :: MonteCarloThDesign )
    des.n_points > 0 && des.dims > 0 && ! ( isempty(des.lb) || isempty(des.ub) ) # && n_points == length(point_array) == length(idist_scores) == length( pdist_scores )
end

function MonteCarloThDesign( n_points :: Int64, dims :: Int64 )
    if dims > 0 && n_points >= 0
        MonteCarloThDesign( n_points, dims, zeros(dims), ones(dims), Vector{Vector{Float64}}() )
    end
end

function MonteCarloThDesign( n_points :: Int64, lb :: T, ub :: T, seeds :: Vector{T} = Vector{T}(); clean_seeds = true ) where{T<:Vector{Float64}}
    if n_points >= 0
        dims = length(lb)
        if dims == length(ub)
            seed_indices = clean_seeds ? good_indices( seeds, lb, ub ) : eachindex(seeds)
            scaled_seeds = scale_to_unit_square( seeds, lb, ub )
            return MonteCarloThDesign( n_points, dims, lb, ub, scaled_seeds[seed_indices] )
        else
            @error "Dimensions of `lb` and `ub` do not match!"
        end
    else
        @error "`n_points` must be non-negative."
    end
end

function Base.iterate( des :: MonteCarloThDesign )
    if is_valid( des )
        # design has no point associated, initalize with zero vector or first seed
        next_point = isempty( des.seeds ) ? zeros( des.dims ) : des.seeds[1]
        point_array = [ next_point ]
        return ( unscale_from_unit_square(next_point, des.lb, des.ub), point_array )
    else
        return nothing
    end
end

function Base.iterate( des :: MonteCarloThDesign, point_array :: Vector{Vector{Float64}}  ) :: Union{ Nothing, Tuple{ Vector{Float64}, Vector{Vector{Float64}} } }
    n_points_so_far = length(point_array)
    if is_valid( des ) && n_points_so_far < des.n_points
        if n_points_so_far >= length( des.seeds )
            d = des.dims
            N = length(point_array)
            spawn_factor = 50

            intersite_factor = (( N + 1 )^( 1/d ) - 1)/2
            pdist_factor = (N+1)/2;
            th = 1 / N;

            candidates = [ rand(d) for j = 1 : max( 200, N * spawn_factor )];

            score_fn = p -> intersite_factor * minimum( distance( p, point_array ) ) + pdist_factor * minimum( projected_distance_thresholded(p, point_array, th) )
            scores = score_fn.(candidates)
            best_index = argmax( scores )

            next_point = candidates[ best_index ]
        else
            next_point = des.seeds[ n_points_so_far + 1 ]
        end
        push!( point_array, next_point )

        return unscale_from_unit_square(next_point, des.lb, des.ub), point_array
    end
end

Base.length( des:: MonteCarloThDesign ) = des.n_points

end
