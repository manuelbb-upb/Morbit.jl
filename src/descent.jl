# methods to compute a local descent step

# Continue Backtracking if true, suppose all_objectives_descent = true (= Val)
function continue_while( :: Val{true}, m_x₊, f_x, step_size, ω )
    any( f_x .- m_x₊ .< step_size * 1e-4 * ω )
end
# Continue Backtracking if true, suppose all_objectives_descent = false (= Val)
function continue_while( :: Val{false}, m_x₊, f_x, step_size, ω )
    maximum(f_x) .- maximum(m_x₊) .<  step_size * 1e-4 * ω
end

const backtrack_factor = 0.8;
const min_step_size = 1e-14;   # to avoid infinite loop at Pareto point

function backtrack( x, f_x, dir, step_size , ω, constrained_flag, all_objectives_descent, surrogate_handle )
    global backtrack_factor, min_step_size;

    x₊ = intobounds( x .+ step_size .* dir, Val(constrained_flag) )
    m_x₊ = surrogate_handle( x₊ )

    while continue_while( Val(all_objectives_descent), m_x₊, f_x, step_size, ω ) && step_size > min_step_size
        step_size *= backtrack_factor;
        x₊[:] = intobounds(x .+ step_size .* dir, Val(constrained_flag));
        m_x₊[:] = surrogate_handle( x₊);
    end
    step = step_size .* dir;
    return x₊, m_x₊, step, norm(step, Inf)
end

function scale( dir :: Vector{Float64}, Δ :: Float64 )
    d_norm = norm(dir,Inf);
    step_size = d_norm ≈ 1.0 ? Δ : min( Δ, d_norm);
    scaled_dir = dir ./ d_norm;
    return scaled_dir, step_size
end

# TODO: enable passing of gradients
@doc """
    compute_descent_direction( Val(:method), f, x, f_x = [], Δ = 1.0, constrained_flag = false )

Compute a (vector) steepest descent direction `d` for the vector-valued function `f`.
`x` is a n-vector and `d`  is a k-vector, where ``f\\colon \\mathbb ℝ^n \\to \\mathbb ℝ^k``
"""
function compute_descent_direction( type::Val{:steepest}, config_struct::AlgoConfig, m::Union{ RBFModel, NamedTuple } )
    @unpack problem = config_struct;        # objective & problem information
    constrained_flag = problem.is_constrained;

    @unpack iter_data, n_vars = config_struct;
    @unpack x, f_x, Δ = iter_data;                  # iteration information

    @unpack all_objectives_descent = config_struct;

    ∇f = eval_jacobian( problem, m, x )

    # construct quadratic optimization problem as by fliege and svaiter
    prob = JuMP.Model(OSQP.Optimizer);
    JuMP.set_silent(prob)
    @variable(prob, d[1:n_vars] )

    @variable(prob, α )     # negative of marginal problem value # TODO check consistency of notation with Paper

    @objective(prob, Min, α)
    @constraint(prob, ∇con, ∇f*d .<= α)
    @constraint(prob, unit_con, -1.0 .<= d .<= 1.0);
    if constrained_flag
        #@constraint(prob, global_const, 0.0 .<=  x .+ Δ .* d .<= 1 )
        @constraint(prob, global_const, 0.0 .<= x .+ d .<= 1.0 )   # honor global constraints
    end

    JuMP.optimize!(prob)
    ω = let o = - value(α); constrained_flag ? min( o, 1.0 ) : o end
    dir, step_size = scale( value.(d), Δ )

    x₊, m_x₊, dir, step_size = backtrack( x, f_x, dir, step_size, ω, constrained_flag, all_objectives_descent, X -> eval_surrogates(problem, m, X) )

    return ω, dir, step_size
end

function compute_descent_direction( type::Val{:direct_search}, config_struct::AlgoConfig, m::Union{ RBFModel, NamedTuple } )
    @unpack problem = config_struct;        # objective & problem information
    constrained_flag = problem.is_constrained;

    @unpack iter_data, n_vars = config_struct;
    @unpack x, f_x, Δ = iter_data;                  # iteration information

    @unpack all_objectives_descent = config_struct;
    @unpack ideal_point, image_direction, θ_enlarge_1 = config_struct;

    ∇f = eval_jacobian( problem, m, x )

    ε_pinv = 1e-3;  # distance from bounds up to which pseudo-inverse is used

    # local lower and upper boundaries for ideal point calculation
    if constrained_flag
        lb, ub = effective_bounds_vectors( x, θ_enlarge_1 * Δ, Val(constrained_flag));
    else
        lb = x .- θ_enlarge_1 * Δ;
        ub = x .+ θ_enlarge_1 * Δ;
    end

    if isempty( image_direction )
        if isempty( ideal_point )
            @info("\tComputing approximated local ideal point.")

            ideal_point = -Inf .* ones( size(f_x) );

            # Minimize each objective individually in trust region to approximate local ideal point
            X_0 = intobounds(x, Val(constrained_flag))
            for l = 1 : length(f_x)
                opt = Opt(:LD_MMA, n_vars)
                opt.lower_bounds = lb;
                opt.upper_bounds = ub;
                opt.xtol_rel = 1e-6;
                opt.maxeval = 200;
                # TODO calculate *global* optima
                opt.min_objective = get_optim_handle( problem, m, l )
                (minf,minx,ret) = NLopt.optimize(opt, X_0 );
                ideal_point[l] = minf;
            end
        end

        image_direction = ideal_point .- f_x;
    end

    @info("\t(Local) ideal point is $ideal_point and im direction is $image_direction.")

    if any( image_direction .>= 0 )
        # deem x critical point
        ω = 0.0;
        dir = zeros(n_vars);
        step_size = 0.0;
    else
        ∇f_pinv = pinv( ∇f );
        if !constrained_flag
            dir = ∇f_pinv * image_direction;
            norm_dir = norm(dir,Inf);
            step_size = Δ / norm_dir;
            ω = - maximum( ∇f * dir )/ norm_dir;
        else
            prob = JuMP.Model(OSQP.Optimizer);
            #JuMP.set_silent(prob)
            @variable(prob, 0.0 <= λ <= 1.0)
            @objective(prob, Max, λ )
            @constraint(prob, ∇con, 0.0 .<= x .+ λ .* (∇f_pinv * image_direction) .<= 1.0 );
            @constraint(prob, unit_const, -1.0 .<= λ .* (∇f_pinv * image_direction) .<= 1.0 );
            #@constraint(prob, global_const, 0.0 .<= x .+ d .<= 1.0 )   # honor global constraints
            JuMP.optimize!(prob)
            λ_opt = value(λ)
            @show λ_opt
            dir, step_size = scale( λ_opt .* (∇f_pinv * image_direction), Δ );
            ω = min(- maximum( ∇f * dir ) / norm(dir, Inf), 1.0)
        end

        #=
        pinv_flag = !constrained_flag || all( x .- lb .>= ε_pinv ) #&& all( ub .- x .>= ε_pinv )); # TODO check wheter boundaries have to be adapted (/θ_enlarge_1)
        if pinv_flag
            # as long as we are not on decision space boundary, pseudo inverse suffices
            ∇f_pinv = pinv( ∇f );
            dir = ∇f_pinv * image_direction;

            #@info "theoretical" minimum(-image_direction) / norm( dir, Inf );
            ω_backtrack_scaling = norm(dir, Inf);
            dir ./= ω_backtrack_scaling;
            ω = - maximum( ∇f * dir );

            step_size, _ = constrained_flag ? intersect_bounds( x, dir, Δ ) : (Δ, 0.0);

        else
            LB, UB = effective_bounds_vectors( x, Δ, Val(constrained_flag) );
            @info "\tUsing QP solver to find direction."
            dir_prob = JuMP.Model( OSQP.Optimizer );

            set_optimizer_attribute(dir_prob,"eps_rel",1e-5)
            set_optimizer_attribute(dir_prob,"polish", true);

            JuMP.set_silent(dir_prob);
            @variable(dir_prob, d[1:n_vars] )

            @objective(dir_prob, Min, .5 * sum( d.^2 ) +  sum( (∇f * d .- image_direction).^2 ) );
            @constraint(dir_prob, df_constraint, image_direction .<= ∇f*d .<= max( .5 * maximum(image_direction), -1e-7  ));
            @constraint(dir_prob, region_constraint, LB .- x .<= d .<= UB .- x )   # honor local & global constraints

            JuMP.optimize!(dir_prob)
            dir = value.(d)
            step_size = 1.0 # due to `region constraint`
            ω_backtrack_scaling = norm(dir, Inf);
            ω = - maximum( ∇f * dir / ω_backtrack_scaling );
        end
        =#
        x₊, m_x₊, dir, step_size = backtrack( x, f_x, dir, step_size, ω, constrained_flag, all_objectives_descent, X -> eval_surrogates(problem, m, X) )
    end

    return ω, dir, step_size
end

#=
function compute_descent_direction( type::Val{:direct_search}, m::Union{RBFModel, NamedTuple}, f:: F where {F<:Function},
    x :: Vector{Float64}, f_x :: Vector{Float64} = [], Δ :: Float64 = 1.0, constrained_flag :: Bool = false,
    all_objectives_descent :: Bool = false, ideal_point = [], image_direction = [], θ_enlarge_1 = 1.0 )
    surrogate_function = x -> vcat( m.function_handle(x), f(x));
    compute_descent_direction( Val(:direct_search), surrogate_function, x, f_x, Δ, constrained_flag , all_objectives_descent, ideal_point, image_direction, θ_enlarge_1)
end
=#
#=
function compute_descent_direction(type::Val{:steepest}, m::Union{RBFModel, NamedTuple}, f :: F where {F<:Function},
    x :: Vector{Float64}, f_x :: Vector{Float64} = [], Δ :: Float64 = 1.0, constrained_flag :: Bool = false,
    all_objectives_descent :: Bool = false, args... )
    surrogate_function = x -> vcat( m.function_handle(x), f(x));
    compute_descent_direction( Val(:steepest), surrogate_function, x, f_x, Δ, constrained_flag, all_objectives_descent )
end
=#

function compute_descent_direction( config_struct::AlgoConfig, m::Union{RBFModel, NamedTuple})
    @unpack descent_method = config_struct;     # method parameters
    compute_descent_direction( Val(descent_method), config_struct, m )
end
