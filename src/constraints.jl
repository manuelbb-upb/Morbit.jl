
#=@doc "Variable boundaries if x is moved to the origin; for descent calculation."
function local_bounds_vectors( x, Δ, lb, ub)
    lb_local = max.( -Δ .- x, lb .- x );
    ub_local = min.( Δ .- x, ub .- x );

    return lb_local, ub_local
end
=#
@doc "Effective boundaries taking global constraints and Δ into account."
function effective_bounds_vectors( x, Δ, lb, ub )
    lb_eff = max.( lb, x .- Δ )
    ub_eff = min.( ub, x .+ Δ )
    return lb_eff, ub_eff
end

function effective_bounds_vectors( x, Δ )
    # assume lb = zeros, ub = ones
    global ε_bounds;
    n_vars = length(x);
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ, zeros( n_vars ) .+ ε_bounds, ones( n_vars ) .- ε_bounds)
    return lb_eff, ub_eff
end

function get_scaling_function( lb, ub )
    function scaling_function( x )
        X = ( x .- lb ) ./ ( ub .- lb )
    end
end

function get_unscaling_function(lb, ub )
    function unscaling_function( X )
        x = X .* (ub .- lb) .+ lb ;
    end
end

inbounds(x) = all( 0 .<= x .<= 1 )
intobounds(x) = min.( max.(0, x), 1)  # project x into unit hypercube
intobounds(x, lb, ub) = min.( max.(lb, x), ub)  # project x into unit hypercube

@doc "Find the scalars σ₊,σ₋ that so that x + σ±*d intersects the variable boundaries."
function intersect_bounds(x :: T,d :: T,lb::T,ub::T ) where {T}# = Array{Float64,1}}
    non_zero = .!(isapprox.(d,0.0, rtol=1e-12)); # TODO tolerance
    if all( .!non_zero )
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

#=function intersect_bounds(config :: AlgoConfig, x :: T, d :: T, Δ :: Float64) where {T}# :: Array{Float64,1}}
    # assume l = zeros, u = ones
    lb_glob = zeros( length(x) ) .+ ε_bounds;
    ub_glob = ones( length(x)) .- ε_bounds;
    lb,ub = effective_bounds_vectors( x, Δ, lb_glob, ub_glob);

    intersect_bounds( x, d, lb, ub);
end
=#
function intersect_bounds( x :: Array{Float64,1}, d :: Array{Float64,1}, Δ :: Float64 )
    global n_vars;    # not defined yet;
    lb, ub = effective_bounds_vectors( x, Δ );
    intersect_bounds(x , d, lb, ub)
end


function init_objectives( problem, config_struct )
    lb = problem.lb;
    ub = problem.ub;

    if (isempty( lb ) && isempty(ub)) || all( isinf.(lb) .& isinf.(ub) )
        f_expensive = problem.f_expensive;
        f_cheap = problem.f_cheap;

        scaling_func(x) = x;
        unscaling_func(x) = x;

        is_constrained = false;
    else
        # sites are scaled internally, the objectives should unscale them first
        scaling_func = get_scaling_function(lb,ub);
        unscaling_func = get_unscaling_function(lb,ub);

        f̂_expensive = problem.f_expensive;
        f̂_cheap = problem.f_cheap;
        f_expensive(x) = f̂_expensive( unscaling_func(x) );
        f_cheap(x) = f̂_cheap( unscaling_func(x) );

        problem = reconstruct( problem, f_cheap = f_cheap, f_expensive = f_expensive );
        is_constrained = true;
    end

    f(x) = vcat( f_expensive(x), f_cheap(x) );
    config_struct.f = f;

    return is_constrained, problem, f, f_expensive, f_cheap, scaling_func, unscaling_func
end
