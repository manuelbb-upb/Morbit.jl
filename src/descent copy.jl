# methods to compute a local descent step

# Continue Backtracking if true, suppose all_objectives_descent = true (= Val)
function continue_while( :: Val{true}, m_x₊, f_x, step_size, ω )
    any( f_x .- m_x₊ .< step_size * 1e-6 * ω )
end
# Continue Backtracking if true, suppose all_objectives_descent = false (= Val)
function continue_while( :: Val{false}, m_x₊, f_x, step_size, ω )
    maximum(f_x) - maximum(m_x₊) <  step_size * 1e-6 * ω
end

# TODO make these available from AlgoConfig
const backtrack_factor = 0.8;
const max_backtrack = 40;   # to avoid infinite loop at Pareto point

@doc """
Perform a backtracking loop starting at `x` with an initial step of
`step_size .* dir` and return trial point `x₊`, the surrogate value-vector `m_x₊`
and the final step `s = x₊ .- x`.
"""
function backtrack( x :: Vector{R}, dir :: Vector{F}, 
        step_size :: Real, ω :: Real, sc :: SurrogateContainer, descent_all :: Bool 
    ) where{ R<:Real, F<:Real }
    
    global backtrack_factor, max_backtrack;

    x₊ = x .+ step_size .* dir
    m_x₊ = eval_models( sc, x₊ )

    m_x = eval_models(sc, x)

    n_backtrack = 0;
    while continue_while( Val(descent_all), m_x₊, m_x, step_size, ω ) && n_backtrack < max_backtrack
        step_size *= backtrack_factor;
        x₊[:] = x .+ step_size .* dir;
        m_x₊[:] = eval_models( sc, x₊ );
        n_backtrack += 1;
    end

    step = step_size .* dir;
    return x₊, m_x₊, step
end

@doc """
    get_initial_step( dir :: Vector{R}, Δ :: Real )

Return the ∞-normed direction and a stepsize σ, so that
* σ is the minimum of `Δ` and `norm(dir,Inf)` if any of those values is less than 1
* or σ is Δ elsewise.
"""
function get_initial_step( dir :: Vector{R} where R<:Real, Δ :: Real )
    d_norm = norm(dir,Inf);
    step_size = (d_norm + 1e-10 < 1 || Δ <= 1) ? min( Δ, d_norm) : Δ;
    scaled_dir = dir ./ d_norm;
    return scaled_dir, step_size
end

function steepest_direction( x :: Vector{R} where R<:Real,
        ∇m :: T where T <: AbstractArray, constrained :: Bool )
    # construct linear optimization problem as by (Fliege, Svaiter)
    n_vars = length(x);
    prob = JuMP.Model(OSQP.Optimizer);
    JuMP.set_silent(prob)

    set_optimizer_attribute(prob,"eps_rel",1e-5)
    set_optimizer_attribute(prob,"polish",true)

    @variable(prob, α )     # negative of marginal problem value
    @objective(prob, Min, α)

    @variable(prob, d[1:n_vars] )   # direction vector
    @constraint(prob, descent_contraints, ∇m*d .<= α)
    @constraint(prob, norm_constraints, -1 .<= d .<= 1);
    if constrained
        @constraint(prob, box_constraints, 0 .<= x .+ d .<= 1 )
    end
    try
        JuMP.optimize!(prob)
        #@show x .+ value.(d)
        return value.(d), -value(α)
    catch e
        return zeros( n_vars ), -Inf
    end
end

@doc """
    compute_descent_direction( Val(:steepest), cfg :: AlgoConfig, sc :: SurrogateContainer )

Compute a (vector) steepest descent direction `d` for the vector-valued
surrogates stored in `sc`.
*Return* the criticality value `ω`, direction `d`, trial point `x₊` and surrogate
value vector `m_x₊` at the trial point.

The direction is calculated by solving a linear program, cf. (Fliege, Svaiter).
"""
function compute_descent_direction( type::Val{:steepest},
        config_struct::AlgoConfig, sc :: SurrogateContainer )
    # objective & problem information:
    @unpack n_vars, iter_data, problem, all_objectives_descent = config_struct;
    @unpack x, f_x, Δ = iter_data;

    ∇m = get_jacobian( sc, x )
    d, ω = steepest_direction( x , ∇m, problem.is_constrained )

    ω = problem.is_constrained ? min( ω, 1 ) : ω
    dir, step_size = get_initial_step( d, Δ )
    x₊, m_x₊, dir = backtrack(x, dir, step_size, ω, sc, all_objectives_descent)

    # safeguard if something went wrong in descent step calculation
    x₊ = intobounds(x₊, Val(problem.is_constrained))
    return ω, dir, x₊, m_x₊
end

# Nonlinear Conjugate Gradient Methods, See Lucambio Pérez and Prudente
function compute_descent_direction( type::Val{:cg},
        config_struct::AlgoConfig, sc :: SurrogateContainer )
    # objective & problem information:
    @unpack n_vars, iter_data, problem, all_objectives_descent = config_struct;
    @unpack x, f_x, Δ = iter_data;

    max_iter = n_vars;

    # initialization
    d = dir = zeros(n_vars);
    x₊ = copy(x);
    m_x₊ = f_x;
    ω = 1;
    for k = 1 : max_iter
        ∇m = get_jacobian( sc, x₊)
        v, ω₊ = steepest_direction( x , ∇m, problem.is_constrained )

        β = ω₊ / ω
        d = v .+ β .* d

        dir_0, step_size = get_initial_step( d, Δ )
        x₊, m_x₊, dir  = backtrack(x₊, dir_0, step_size, ω₊, sc, all_objectives_descent)

        ω = ω₊;
        #@show ω
        if ω₊ <= 1e-4
            break;
        end
    end
    # safeguard if something went wrong in descent step calculation
    x₊ = intobounds(x₊, Val(problem.is_constrained))
    return ω, dir, x₊, m_x₊
end


@doc """
    compute_descent_direction( Val(:direct_search), cfg :: AlgoConfig, sc :: SurrogateContainer )

Compute a (vector) descent direction `d` for the vector-valued surrogates stored
in `sc`.
*Return* the criticality value `ω`, direction `d`, trial point `x₊` and surrogate
value vector `m_x₊` at the trial point.

The direction is calculated using directed search.
"""
function compute_descent_direction( type::Val{:ps},
        config_struct::AlgoConfig, sc :: SurrogateContainer )

    # objective & problem information:
    @unpack n_vars, iter_data, problem, all_objectives_descent = config_struct;
    @unpack x, f_x, Δ = iter_data;

    ideal_point, image_direction = compute_ideal_point( config_struct, sc :: SurrogateContainer )
    @info("\t(Local) ideal point is \n\t\t$ideal_point and direction is \n\t\t$image_direction.")

    r = - image_direction;
    m_x = eval_models( sc, x )
    lb, ub = effective_bounds_vectors( x, Δ, Val(problem.is_constrained));
    x_0 = [0; intobounds(x, lb, ub)]

    opt_objf = function( χ, g )
        if !isempty(g)
            g[1] = 1
            g[2:end] .= 0
        end
        return χ[1]
    end

    opt = Opt(:GN_ISRES, n_vars + 1)
    opt.lower_bounds = [-1; lb];
    opt.upper_bounds = [0; ub];
    opt.xtol_rel = 1e-3;
    opt.maxeval = 500*(n_vars+2);
    opt.min_objective = opt_objf;
    for ℓ = 1 : length(f_x)
        ℓth_constraint = function( χ, g )
            if !isempty(g)
                g[1] = image_direction[1];
                g[2:end] .= get_gradient(sc, χ[2:end], ℓ)
            end
            ℓ_val = eval_models(sc, χ[2:end], ℓ ) .- m_x[ℓ] .- χ[1] .* r[ℓ]
            return ℓ_val
        end
        inequality_constraint!(opt, ℓth_constraint)
    end

    (τ,χ_min,ret) = NLopt.optimize(opt, x_0 );

    ω = abs(τ);
    x₊ = χ_min[2:end];
    m_x₊ = eval_models(sc, x₊)
    dir = x₊ .- x;
    return ω, dir, x₊, m_x₊
end

@doc """
    compute_descent_direction( Val(:direct_search), cfg :: AlgoConfig, sc :: SurrogateContainer )

Compute a (vector) descent direction `d` for the vector-valued surrogates stored
in `sc`.
*Return* the criticality value `ω`, direction `d`, trial point `x₊` and surrogate
value vector `m_x₊` at the trial point.

The direction is calculated using directed search.
"""
function compute_descent_direction( type::Val{:direct_search},
        config_struct::AlgoConfig, sc :: SurrogateContainer )

    # objective & problem information:
    @unpack n_vars, iter_data, problem, all_objectives_descent = config_struct;
    @unpack x, f_x, Δ = iter_data;

    ∇m = get_jacobian( sc, x )

    ideal_point, image_direction = compute_ideal_point( config_struct, sc :: SurrogateContainer )

    @info("\t(Local) ideal point is \n\t\t$ideal_point and direction is \n\t\t$image_direction.")

    if any( image_direction .>= 0 )
        # deem x critical point
        ω = 0;
        dir = zeros(n_vars);
        x₊ = x; m_x₊ = f_x;
    else
        ∇m⁺ = pinv( ∇m );
        if !problem.is_constrained
            d = ∇m⁺ * image_direction;
        else
            prob = JuMP.Model(OSQP.Optimizer);
            JuMP.set_silent(prob)

            set_optimizer_attribute(prob,"eps_rel",1e-5)
            set_optimizer_attribute(prob, "polish", true)
            
            # slightly adapted version of directed search (allow only non-anscent directions)
            @variable(prob, D[1:n_vars] )
            @objective(prob, Min, sum(( ∇m * D .- image_direction ).^2) )
            @constraint(prob, norm_constraint, -1 .<= D .<= 1)
            @constraint(prob, descent, ∇m*D .<= 0)
            if problem.is_constrained
                @constraint(prob, global_constraint, 0 .<= x .+ D .<= 1 )
            end
            JuMP.optimize!(prob)
            d = value.(D)
            
            #=
            # This probably corresponds best to directed search
            @variable(prob, 0 <= λ <= 1)
            @objective(prob, Max, λ )
            if problem.is_constrained
                @constraint(prob, global_constraints, 0 .<= x .+ λ .* (∇m⁺ * image_direction) .<= 1 );
            end
            @constraint(prob, norm_constraints, -1 .<= λ .* (∇m⁺ * image_direction) .<= 1 );
            JuMP.optimize!(prob)

            λ_opt = value(λ)
            d = λ_opt .* (∇m⁺ * image_direction);
            @show λ_opt
            =#
             
        end
        dir, step_size = get_initial_step( d, Δ )
        # taking minimum of abs values because near critical points the signs get confused
        ω = let o = minimum( abs.(∇m * dir) ); problem.is_constrained ? min( o, 1 ) : o end

        x₊, m_x₊, dir = backtrack(x, dir, step_size, ω, sc, all_objectives_descent)
    end
    #x₊ = intobounds(x₊, Val(problem.is_constrained))

    return ω, dir, x₊, m_x₊
end

function compute_descent_direction( config_struct::AlgoConfig, sc :: SurrogateContainer )
    @unpack descent_method = config_struct;     # method parameters
    compute_descent_direction( Val(descent_method), config_struct, sc )
end

function compute_ideal_point( config_struct :: AlgoConfig, sc :: SurrogateContainer )
    # local lower and upper boundaries for ideal point calculation
    @unpack n_vars, problem, iter_data, θ_ideal_point,
        image_direction, ideal_point = config_struct;
    @unpack x, f_x, Δ = iter_data;

    lb, ub = effective_bounds_vectors( x, θ_ideal_point * Δ, Val(problem.is_constrained));

    if isempty( image_direction )
        if isempty( ideal_point )
            @info("\tComputing approximated local ideal point.")

            ideal_point = fill( -Inf, size(f_x) );

            # Minimize each objective individually in trust region to approximate
            # the local ideal point.
            # TODO calculate *global* optima
            x_0 = intobounds(x, lb, ub);
            for l = 1 : length(f_x)
                # setup an optimization problem
                opt = Opt(:GN_ISRES, n_vars)
                opt.lower_bounds = lb;
                opt.upper_bounds = ub;
                opt.xtol_rel = 1e-3;
                opt.maxeval = 500*(n_vars+1);
                opt.min_objective = get_optim_handle( sc, l )
                (minf,minx,ret) = NLopt.optimize(opt, x_0 );
                ideal_point[l] = minf;
            end
        end
        image_direction = ideal_point .- f_x;
    end
    return ideal_point, image_direction
end