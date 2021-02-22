# methods to compute a local descent step
function compute_descent_step( algo_config :: AbstractConfig, mop :: AbstractMOP,
    id :: AbstractIterData, sc :: SurrogateContainer )
    return compute_descent_step( Val(descent_method(algo_config)), algo_config, mop, id, sc )
end

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
    return all(Fx - Fx₊ .>= step_size * 1e-6 * ω)
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

    MAX_BACKTRACK = 15;
    BACKTRACK_FACTOR = Float16(0.8);
    # values at iterate
    mx = eval_models(sc, x)

    # first trial point and its value vector
    x₊ = x .+ step_size .* dir
    mx₊ = eval_models( sc, x₊ )

    N = 0;
    while !_armijo_condition( Val(strict_descent), mx₊, mx, step_size, ω ) && N < MAX_BACKTRACK
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