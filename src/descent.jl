# methods to compute a local descent step

# Continue Backtracking if true, suppose all_objectives_descent = true (= Val)
function continue_while( :: Val{true}, m_x₊, f_x, step_size, ω )
    any( f_x .- m_x₊ .< step_size * 1e-6 * ω )
end
# Continue Backtracking if true, suppose all_objectives_descent = false (= Val)
function continue_while( :: Val{false}, m_x₊, f_x, step_size, ω )
    maximum(f_x) .- maximum(m_x₊) .<  step_size * 1e-6 * ω
end

# TODO: enable passing of gradients
@doc """
    compute_descent_direction( Val(:method), f, x, f_x = [], Δ = 1.0, constrained_flag = false )

Compute a (vector) steepest descent direction `d` for the vector-valued function `f`.
`x` is a n-vector and `d`  is a k-vector, where ``f\\colon \\mathbb ℝ^n \\to \\mathbb ℝ^k``
"""
function compute_descent_direction( type::Val{:steepest}, f:: F where {F<:Function}, x :: Vector{Float64}, f_x :: Vector{Float64} = [],
    Δ :: Float64 = 1.0, constrained_flag :: Bool = false, all_objectives_descent :: Bool = false, args... )
    n_vars = length(x);

    if isempty( f_x )
        f_x = f(x);
    end

    ∇f = jacobian( f, x )

    # construct quadratic optimization problem as by fliege and svaiter
    prob = JuMP.Model(OSQP.Optimizer);
    JuMP.set_silent(prob)
    @variable(prob, d[1:n_vars] )

    @variable(prob, α )     # negative of marginal problem value # TODO check consistency of notation with Paper

    @objective(prob, Min, α)
    @constraint(prob, ∇con, ∇f*d .<= α)
    @constraint(prob, unit_con, -1.0 .<= d .<= 1.0);
    if constrained_flag
        @constraint(prob, global_const, 0.0 .- x .<= d .<= 1 .- x )   # honor global constraints
    end

    JuMP.optimize!(prob)
    ω = - value(α)
    dir = value.(d)

    # step size has to suffice a sufficient decrease condition
    ## perform armijo like backtracking
    if !constrained_flag
        dir = dir ./ init_norm; # necessary?
        step_size = Δ;
    else
        step_size, _ = intersect_bounds( x, dir, Δ );
    end
    x₊ = intobounds(x + step_size * dir);           # 'intobounds' for tiny errors
    m_x₊ = f( x₊ );

    backtrack_factor = 0.8;
    min_step_size = 1e-17;   # to avoid infinite loop at Pareto point
    while continue_while( Val(all_objectives_descent), m_x₊, f_x, step_size, ω ) && step_size > min_step_size
        step_size *= backtrack_factor;
        x₊ = intobounds(x + step_size .* dir);
        m_x₊ = f( x₊ );
    end
    @show step_size
    dir = x₊ .- x;

    return ω, dir, step_size
end

function compute_descent_direction(type::Val{:steepest}, m::Union{RBFModel, NamedTuple}, f :: F where {F<:Function},
    x :: Vector{Float64}, f_x :: Vector{Float64} = [], Δ :: Float64 = 1.0, constrained_flag :: Bool = false,
    all_objectives_descent :: Bool = false, args... )
    surrogate_function = x -> vcat( m.function_handle(x), f(x));
    compute_descent_direction( Val(:steepest), surrogate_function, x, f_x, Δ, constrained_flag, all_objectives_descent )
end

function compute_descent_direction( type::Val{:direct_search}, f :: F where {F<:Function}, x :: Vector{Float64}, f_x :: Vector{Float64} = [], Δ :: Float64 = 1.0, constrained_flag :: Bool = false, all_objectives_descent :: Bool = false, ideal_point = [], image_direction = [] )
    n_vars = length(x);
    ε_pinv = 0.0#1e-10;  # distance from bounds up to which pseudo-inverse is used

    if isempty( f_x )
        f_x = f(x);
    end

    ∇f = jacobian( f, x )

    # local lower and upper boundaries
    if constrained_flag
        lb, ub = effective_bounds_vectors( x, Δ);
    else
        lb = x .- Δ;
        ub = x .+ Δ;
    end

    if isempty( image_direction )
        if isempty( ideal_point )
            @info("\tComputing approximated local ideal point.")

            ideal_point = -Inf .* ones( size(f_x) );

            # Minimize each objective individually in trust region to approximate local ideal point
            for l = 1 : length(f_x)
                opt = Opt(:LD_MMA, n_vars)
                opt.lower_bounds = lb;
                opt.upper_bounds = ub;
                opt.xtol_rel = 1e-6;
                opt.maxeval = 200;
                # TODO Optimize individual output minimization
                opt.min_objective = (x,g) -> begin
                    g[:] = gradient( X -> f(X)[l] , x)
                    return f(x)[l]
                end
                (minf,minx,ret) = NLopt.optimize(opt, intobounds(x));
                ideal_point[l] = minf;
            end
        end

        image_direction = ideal_point .- f_x;
    end

    @info("\tIdeal point is $ideal_point and im direction is $image_direction.")

    if any( image_direction .>= 0 )
        # deem x critical point
        ω = 0.0;
        dir = zeros(n_vars);
        step_size = 0;
    else
        pinv_flag = !constrained_flag || (all( x .- lb .>= ε_pinv ) && all( ub .- x .>= ε_pinv ));
        if pinv_flag
            # as long as we are not on decision space boundary, pseudo inverse suffices
            ∇f_pinv = pinv( ∇f );
            dir = ∇f_pinv * image_direction;
            @show norm(dir, Inf)
            dir ./= norm( dir, Inf );
            ω = - maximum( ∇f * dir );
            step_size, _ = constrained_flag ? intersect_bounds( x, dir, Δ ) : (Δ, 0.0);
            @show intersect_bounds( x, dir, Δ )
            @show step_size
        else
            @info "\tUsing QP solver to find direction."
            dir_prob = JuMP.Model( OSQP.Optimizer );

            set_optimizer_attribute(dir_prob,"eps_rel",1e-5)
            set_optimizer_attribute(dir_prob,"polish", true);

            #JuMP.set_silent(dir_prob);
            @variable(dir_prob, d[1:n_vars] )

            @objective(dir_prob, Min, .5 * sum( d.^2 ) +  sum( (∇f * d .- image_direction).^2 ) );
            @constraint(dir_prob, df_constraint, image_direction .<= ∇f*d .<= max( .5 * maximum(image_direction), -1e-12  ));
            @constraint(dir_prob, global_const, 0.0 .- x .<= d .<= 1.0 .- x )   # honor global constraints

            JuMP.optimize!(dir_prob)
            dir = value.(d)
            step_size = 1.0
            ω = - maximum( ∇f * dir / norm(dir, Inf) );

        end

        # step size has to suffice a sufficient decrease condition
        ## perform armijo like backtracking
        if !constrained_flag
            step_size = Δ;
        else

        end
        #@info step_size
        x₊ = intobounds(x + step_size * dir);
        m_x₊ = f( x₊ );

        backtrack_factor = 0.8;
        min_step_size = 1e-15; #backtrack_factor^15 * step_size;    # to avoid infinite loop at Pareto point

        while continue_while( Val(all_objectives_descent), m_x₊, f_x, step_size, ω ) && step_size > min_step_size
            step_size *= backtrack_factor;
            x₊ = intobounds(x + step_size .* dir);
            m_x₊ = f( x₊ );
        end

        dir = x₊ - x;
        step_size = norm( dir, 2);
        @show step_size
    end

    return ω, dir, step_size
end

function compute_descent_direction( type::Val{:direct_search}, m::Union{RBFModel, NamedTuple}, f:: F where {F<:Function},
    x :: Vector{Float64}, f_x :: Vector{Float64} = [], Δ :: Float64 = 1.0, constrained_flag :: Bool = false,
    all_objectives_descent :: Bool = false, ideal_point = [], image_direction = [] )
    surrogate_function = x -> vcat( m.function_handle(x), f(x));
    compute_descent_direction( Val(:direct_search), surrogate_function, x, f_x, Δ, constrained_flag , all_objectives_descent, ideal_point, image_direction)
end
