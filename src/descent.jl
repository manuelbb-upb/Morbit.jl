# methods to compute a local descent step
function compute_descent_step( algo_config :: AbstractConfig, mop :: AbstractMOP,
    id :: AbstractIterData, sc :: SurrogateContainer )
    return compute_descent_step( Val(descent_method(algo_config)), algo_config, mop, id, sc )
end

# STEEPEST DESCENT AND BACKTRACKING

function _steepest_descent( x :: RVec, ∇F :: RMat, lb :: RVec, ub :: RVec )
        
    n = length(x);

    opt_problem = JuMP.Model( OSQP.Optimizer );
    JuMP.set_silent(opt_problem);

    JuMP.set_optimizer_attribute( opt_problem, "eps_rel", 1e-5 );
    JuMP.set_optimizer_attribute( opt_problem, "polish", true );

    JuMP.@variable(opt_problem, α)  # criticality measure with flipped sign
    JuMP.@objective(opt_problem, Min, α)

    JuMP.@variable(opt_problem, d[1:n]) # steepest descent direction 
    
    JuMP.@constraint(opt_problem, descent_constraints, ∇F*d .<= α)
    JuMP.@constraint(opt_problem, norm_constraints, -1 .<= d .<= 1 );
    JuMP.@constraint(opt_problem, global_scaled_var_bounds, lb .<= x .+ d .<= ub);

    try 
        JuMP.optimize!(opt_problem)
        return JuMP.value.(d), -JuMP.value(α)
    catch e
        println(e)
        catch_backtrace()
        @warn("Could not optimize steepest descent subproblem.\n")
        return zeros(n), -Inf
    end
end

function _armijo_condition( strict :: Val{true}, Fx, Fx₊, step_size, ω  )
    return minimum(Fx .- Fx₊) >= step_size * 1e-6 * ω
end

function _armijo_condition( strict :: Val{false}, Fx, Fx₊, step_size, ω  )
    return maximum(Fx) - maximum(Fx₊) >= step_size * 1e-6 * ω 
end

@doc """
Perform a backtracking loop starting at `x` with an initial step of
`step_size .* dir` and return trial point `x₊`, the surrogate value-vector `m_x₊`
and the final step `s = x₊ .- x`.
"""
function _backtrack( x :: RVec, dir :: RVec, step_size :: Real, ω :: Real, 
    sc :: SurrogateContainer, strict_descent :: Bool)

    MIN_STEPSIZE = eps(Float64);
    BACKTRACK_FACTOR = Float16(0.8);
    # values at iterate
    mx = eval_models(sc, x)

    # first trial point and its value vector
    x₊ = x .+ step_size .* dir
    mx₊ = eval_models( sc, x₊ )

    N = 0;
    while !_armijo_condition( Val(strict_descent), mx, mx₊, step_size, ω )
        if step_size <= MIN_STEPSIZE 
            @warn "Could not find a descent by backtracking."
            break;
        end 
        step_size *= BACKTRACK_FACTOR;
        x₊[:] = x .+ step_size .* dir;
        mx₊[:] = eval_models( sc, x₊);
        N += 1;
    end

    step = step_size .* dir;
    return x₊, mx₊, step
end

function compute_descent_step(::Val{:steepest_descent}, algo_config :: AbstractConfig,
    mop :: AbstractMOP, id :: AbstractIterData, sc :: SurrogateContainer )
    @logmsg loglevel3 "Calculating steepest descent."

    x = xᵗ(id);
    ∇m = get_jacobian(sc, x);
    lb, ub = full_bounds_internal( mop );    
    d, ω = _steepest_descent( x, ∇m, lb, ub );

    # scale direction for backtracking as in paper
    norm_d = norm(d,Inf);
    if norm_d > 0
        d_normed = d ./ norm_d;
        σ = intersect_bounds( mop, x, Δᵗ(id), d_normed; return_vals = :pos )
        # Note: For scalar Δ the above should equal 
        # `σ = norm(d,Inf) < 1 || Δ <= 1 ? min( d, norm(d,Inf) ) : Δ`
        @assert σ >= 0

        x₊, mx₊, step = _backtrack( x, d_normed, σ, ω, sc, strict_backtracking(algo_config) );
        return ω, x₊, mx₊, norm(step, Inf)
    else
        return 0, copy(x), eval_models( sc, x ), 0
    end
end

# PASCOLETTI-SERAFINI

function compute_descent_step(::Val{:pascoletti_serafini}, algo_config :: AbstractConfig,
    mop :: AbstractMOP, id :: AbstractIterData, sc :: SurrogateContainer )
    pascoletti_serafini(Val(ps), algo_config, mop, id, sc )
end

function compute_descent_step(::Val{:ps}, algo_config :: AbstractConfig,
    mop :: AbstractMOP, id :: AbstractIterData, sc :: SurrogateContainer )
    @logmsg loglevel3 "Calculating Pascoletti-Serafini descent."
    x = xᵗ(id);
    fx = fxᵗ(id);
    FX = reverse_internal_sorting(fx,mop)
    n_vars = num_vars(mop);
    n_out = num_objectives(mop);
    
    r = begin 
        if !isempty( reference_direction(algo_config) )
            dir = reference_direction(algo_config)
            if all( dir .<= 0 )
                dir *= -1
            end
        elseif !isempty( reference_point(algo_config))
            dir = FX .- reference_point(algo_config)
        else
            dir = FX .- _local_ideal_point(mop, algo_config, id, sc)
        end
        dir
    end

    @logmsg loglevel4 "Local image direction is $(_prettify(r))"

    if any( r .<= 0 )
        return 0, copy(x), eval_models(sc, x), 0
    end

    lb, ub = local_bounds(mop, x, 1.1 .* Δᵗ(id));
    mx = eval_models( sc, x );

    MAX_EVALS = max_ps_problem_evals(algo_config);
    if MAX_EVALS < 0 MAX_EVALS = 500 * (n_vars+1); end

    polish_algo = ps_polish_algo(algo_config)
    MAX_EVALS_global = isnothing( polish_algo ) ? MAX_EVALS : Int( floor( MAX_EVALS*3/4 ) );
    
    τ, χ_min, ret = _ps_optimization(sc,mop,ps_algo(algo_config),lb,ub,
        MAX_EVALS_global,[-0.5;x],mx,r,n_vars,n_out);

    if !isnothing(polish_algo)
        @logmsg loglevel4 "Local polishing enabled."
        MAX_EVALS_local = MAX_EVALS - MAX_EVALS_global;
        τₗ, χ_minₗ, retₗ = _ps_optimization(sc,mop,polish_algo,lb,ub,
            MAX_EVALS_local,χ_min,mx,r,n_vars,n_out);
        if !(retₗ == :FAILURE || isinf(τₗ) || isnan(τₗ))
            τ, χ_min, ret = τₗ, χ_minₗ, retₗ
        end
    end

    if ret == :FAILURE
        return 0, copy(x), eval_models(sc, x), 0
    else
        ω = abs( τ );
        x₊ = χ_min[2 : end];
        mx₊ = eval_models( sc, x₊ );
        sl = norm( x .- x₊, Inf );

        return ω, x₊, mx₊, sl
    end
end

function _ps_optimization( sc :: SurrogateContainer, mop :: AbstractMOP, 
    algo :: Symbol, lb :: RVec, ub :: RVec, MAX_EVALS :: Int, χ0 :: RVec, 
    mx :: RVec, r :: RVec, n_vars :: Int, n_out :: Int )
    
    opt = NLopt.Opt( algo, n_vars + 1 );
    opt.lower_bounds = [-1.0 ; lb ];
    opt.upper_bounds = [ 0.0 ; ub ];
    opt.xtol_rel = 1e-3;
    opt.maxeval = MAX_EVALS;
    opt.min_objective = _get_ps_objective_func();
    
    #NLopt.inequality_constraint!( opt, _get_ps_constraint_func(sc,mx,r), 1e-12 );
    for l = 1 : n_out 
        NLopt.inequality_constraint!( opt, _get_ps_constraint_func(sc,mop,mx,r,l), 1e-12 );
    end

    @logmsg loglevel4 "Starting PS optimization."
    τ, χ_min, ret = NLopt.optimize(opt, χ0 );
    @logmsg loglevel4 "Finished with $(ret) after $(opt.numevals) model evals."
    return τ, χ_min, ret 
end

function _get_ps_constraint_func( sc :: SurrogateContainer, mx :: RVec, dir :: RVec ) :: Function
    # return the l-th constraint functions for pascoletti_serafini
    # dir .>= 0 is the image direction
    # χ = [t;x] is the augmented variable vector
    # g is a matrix of gradients (columns are grads, hence g is a jacobian transposed)
    return function(χ, g)
        if !isempty(g)
            g[1, :] .= -dir;
            g[2:end, :] .= transpose(get_jacobian( sc, χ[2:end]));
        end
        return eval_models(sc, χ[2:end]) .- mx .- χ[1] .* dir
    end
end

function _get_ps_constraint_func( sc :: SurrogateContainer, mop :: AbstractMOP, mx :: RVec,
    dir :: RVec, l :: Int ) :: Function
    # return the l-th constraint functions for pascoletti_serafini
    # dir .>= 0 is the image direction
    # χ = [t;x] is the augmented variable vector
    
    return function(χ, g)
        if !isempty(g)
            g[1] = -dir[l]
            g[2:end] .= get_gradient(sc,mop,χ[2:end],l);
        end
        ret_val = eval_models(sc, mop, χ[2:end], l) - mx[l] - χ[1] * dir[l]
        return ret_val
    end
end

function _get_ps_objective_func() :: Function
    function( χ, g )
        if !isempty(g)
            g[1] = 1.0;
            g[2:end] .= 0.0;
        end
        return χ[1]
    end
end

function _local_ideal_point(mop :: AbstractMOP, algo_config :: AbstractConfig,
    id :: AbstractIterData, sc :: SurrogateContainer) :: RVec
    @logmsg loglevel4 "Computing local ideal point. This can take a bit…"
    x0 = xᵗ(id);
    # TODO enable adjustable enlargment factor here
    lb, ub = local_bounds(mop, x0, 1.1 .* Δᵗ(id));
    n_vars = num_vars(mop);
    
    MAX_EVALS = max_ideal_point_problem_evals(algo_config);
    if MAX_EVALS < 0 MAX_EVALS = 500 * (n_vars+1); end
    
    # preallocate local ideal point:
    ȳ = fill( typemin( eltype( fxᵗ(id) ) ), num_objectives(mop) );

    # minimize each individual scalar surrogate output 
    for l = 1 : num_objectives(mop)
        opt = NLopt.Opt( ideal_point_algo(algo_config), n_vars );
        opt.lower_bounds = lb;
        opt.upper_bounds = ub;
        opt.xtol_rel = 1e-3;
        opt.maxeval = MAX_EVALS;
        opt.min_objective = get_optim_handle( sc, mop, l )
        minf, _ = NLopt.optimize( opt, x0 );
        ȳ[l] = minf;
    end
    return ȳ;
end

