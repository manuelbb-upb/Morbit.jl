# In this file, methods for the descent step calculation are defined.
# The method `get_criticality` takes all the same arguments as `iterate!`
# • `mop` to have access to constraint information 
# • `scal` to transform and untransform sites to and from scaled space
# • `iter_data` for iteration data (x, fx, …)
# • `data_base` (just in case, e.g. if some simplex gradients have to be done on the fly)
# • `sc` to evaluate the surrogate model
# • `algo_config` to have access to `desc_cfg`, an AbstractDescentConfiguration
#
# For the individual descent step methods a method with signature 
# `get_criticality( :: typeof{desc_cfg}, :: AbstractMOP, ::AbstractIterate,
#  ::SuperDB, sc :: SurrogateContainer )`
# should be defined, because this is then called internally
# 
# Return: a criticality measure the new trial point (same eltype as x),
# the new trial point model values and the steplength (inf-norm)

# method to only compute criticality
function get_criticality( mop :: AbstractMOP, scal :: AbstractVarScaler, x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
        data_base, sc :: SurrogateContainer, algo_config :: AbstractConfig;
        kwargs...
    )

    return get_criticality( descent_method( algo_config ), mop, scal, x_it, x_it_n, data_base, sc, algo_config; kwargs... ) 
end

# methods to compute a local descent step
function compute_descent_step( mop :: AbstractMOP, scal :: AbstractVarScaler, x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
       data_base, sc :: SurrogateContainer, algo_config :: AbstractConfig, args...; kwargs...
    )
    return compute_descent_step( descent_method( algo_config ), mop, scal, x_it, x_it_n, data_base, sc, algo_config, args... ; kwargs...)
end

# fallback, if `get_criticality` allready does the whole job: simply return `ω` and `data`

function compute_descent_step(cfg :: AbstractDescentConfig, mop :: AbstractMOP, scal :: AbstractVarScaler, 
        x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
        data_base, sc :: SurrogateContainer, algo_config :: AbstractConfig, ω, ω_data; kwargs...
    )
    return ω, ω_data...
end

function compute_descent_step( cfg :: AbstractDescentConfig, mop :: AbstractMOP, scal :: AbstractVarScaler, 
        x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
        data_base, sc :: SurrogateContainer, algo_config :: AbstractConfig; kwargs...
    )
    ω, data = get_criticality(cfg, mop, scal, x_it, x_it_n, data_base, sc, algo_config; kwargs...)
    return ω, data...
end

# ## STEEPEST DESCENT AND BACKTRACKING

@with_kw struct SteepestDescentConfig <: AbstractDescentConfig

    "Require a descent in all model objective components."
    strict_backtracking :: Bool = true 

    armijo_const_rhs :: Float64 = 1e-6
    armijo_const_shrink :: Float64 = .75 

    max_loops :: Int = 30
    min_stepsize :: Float64 = 1e-15

    normalize :: Bool = true
end

raw"""
    _steepest_descent_direction( x, Df, lb, ub, A_eq, b_eq, A_ineq, b_ineq;
        normalize = true)

Provided `x` and the (surrogate) jacobian `Df` at `x`, return the constrained 
steepest descent direction, i.e., the optimizer ``d`` of
```math
\begin{aligned}
&\min_{β, d\in ℝ^n} β \quad\text{s.t.}\\
& -1 ≤ d ≤ 1 ,
& Df ⋅ d ≤ β , \\
& lb ≤ x + d ≤ ub, \\ 
& A (x+d) - b ≦ 0.
\end{aligned}
```
If `normalize`, then the rows of `Df` are normalized, resulting in a more “central” direction.
"""
function _steepest_descent_direction(
    x :: AbstractVector{F}, ∇F :: Mat, lb :: Vec, ub :: Vec,
    A_eq = [], b_eq = [], A_ineq = [], b_ineq = [], normalize = true
    ) where F <: AbstractFloat
    
    n = length(x);
    try 
        opt_problem = JuMP.Model( LP_OPTIMIZER );
        JuMP.set_silent(opt_problem)

        JuMP.set_optimizer_attribute( opt_problem, "eps_rel", 1e-5 )
        JuMP.set_optimizer_attribute( opt_problem, "polish", true )

        JuMP.@variable(opt_problem, α)  # criticality measure with flipped sign
        JuMP.@objective(opt_problem, Min, α)

        JuMP.@variable(opt_problem, d[1:n]) # steepest descent direction 

        if normalize
            JuMP.@constraint(opt_problem, descent_constraints, ∇F*d .<= α .* norm.(eachrow(∇F)) )
        else
            JuMP.@constraint(opt_problem, descent_constraints, ∇F*d .<= α)
        end
        JuMP.@constraint(opt_problem, norm_constraints, -1 .<= d .<= 1 )

        # original problem constraints (have to be transformed beforehand)
        JuMP.@constraint(opt_problem, global_scaled_var_bounds, lb .<= x .+ d .<= ub);
        if !isempty(A_eq)
            @assert size(A_eq, 1) == length(b_eq) "Equality constraint dimension mismatch."
            JuMP.@constraint(opt_problem, linear_eq_constraints,  A_eq*d .- b_eq .== 0 )
        end
        
        if !isempty(A_ineq)
            @assert size(A_ineq, 1) == length(b_ineq) "Inequality constraint dimension mismatch."
            JuMP.@constraint(opt_problem, linear_ineq_constraints, A_ineq*d .- b_ineq .<= 0 )
        end

        JuMP.optimize!(opt_problem)
        return F.(JuMP.value.(d)), -F(JuMP.value(α))
    catch e
        println(e)
        @warn("Could not optimize steepest descent subproblem.\n")
        return zeros(F, n), -F(Inf)
    end
end

function _armijo_condition( strict :: Val{true}, Fx, Fx₊, step_size, ω, const_rhs  )
    return all( (Fx .- Fx₊) .>= step_size * const_rhs * ω )
end

function _armijo_condition( strict :: Val{false}, Fx, Fx₊, step_size, ω, const_rhs  )
    return maximum(Fx) - maximum(Fx₊) >= step_size * const_rhs * ω 
end

@doc """
Perform a backtracking loop starting at `x` with an initial step of
`step_size .* dir` and return trial point `x₊`, the surrogate value-vector `m_x₊`
and the final step `s = x₊ .- x`.
"""
function _backtrack( x :: AbstractVector{F}, dir, step_size, ω, sc, cfg, scal ) where F<:AbstractFloat

    MIN_STEPSIZE = cfg.min_stepsize >= 0 ? cfg.min_stepsize : eps(F)
    MAX_LOOPS = cfg.max_loops
    strict_backtracking = cfg.strict_backtracking 
    α = F(cfg.armijo_const_shrink)
    c = F(cfg.armijo_const_rhs)

    # values at iterate (required for _armijo_condition )
    mx = eval_container_objectives_at_scaled_site(sc, scal, x)
    # first trial point and its value vector
    x₊ = x .+ step_size .* dir
    mx₊ = eval_container_objectives_at_scaled_site(sc, scal, x₊ )

    for i = 1 : MAX_LOOPS 
        if _armijo_condition( Val(strict_backtracking), mx, mx₊, step_size, ω, c )
            break
        end
        if step_size <= MIN_STEPSIZE 
            @warn "Could not find a descent by backtracking."
            break
        end 
        step_size *= α
        x₊ = x .+ step_size .* dir
        mx₊ .= eval_container_objectives_at_scaled_site( sc, scal, x₊ )
    end

    step = step_size .* dir
    return x₊, mx₊, step
end

function get_criticality( desc_cfg :: SteepestDescentConfig, mop, scal, x_it, x_it_n, 
        data_base, sc, algo_config
    )

    @logmsg loglevel3 "Calculating steepest descent direction."

    x = get_x_scaled(x_it)
    x_n = get_x_scaled(x_it_n)

    ∇m = eval_container_objectives_jacobian_at_scaled_site(sc, scal, x_n)

    lb, ub = full_bounds_internal( scal )
    
    # For computing a step `t`, starting at `x + n`, the linear constraints are 
    # A⋅(x + n + t) - b ≦ 0 ⇔ At - b + A(x + n) ≦ 0
    # As `x` and `n` are constant we can use `b̃ = b - A(x + n)`.
    # As it happens, `x_it_n` holds `A(x+n) - b`, i.e., `-b̃`.
    _b_eq = -get_eq_const(x_it_n)
    _b_ineq = -get_ineq_const(x_it_n)
    _A_eq, _, _A_ineq, _ = transformed_linear_constraints(scal, mop)

    # Variant 1
    # In theory, We approximate the nonlinear constraints via 
    # c(ξ) = m(x) + Dm(x)⋅(x - ξ)
    # Hence, when looking for a step `t` (taken from `x + n`),
    # we get the constraint approximations 
    # m(x) + Dm(x)⋅(n + t) ≦ 0 ⇔ Dm(x)t - (-m(x)-Dm(x)n) ≤ 0
    Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal,x)
    Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal,x)    
    m_eq = eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x_n)
    m_ineq = eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x_n)
    n = x_n .- x
    __b_eq = -m_eq .- Dm_eq*n
    __b_ineq = -m_ineq .- Dm_ineq*n
    
    #=
    # Variant 2 - disabled because of problems with `_intersect_bounds`
    # Here, we actually do the Taylor Expansion around `x_n=x+n` and get constraints 
    # m(x+n) + Dm(x+n)⋅t ≦ 0
    # The only reason to do so is to directly use `b = -m(x_n)` and `A = Dm(x_n)`:    
    Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal,x_n)    # TODO these should be returned by `compute_normal_step` 
    Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal,x_n)    
    m_eq = -eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x_n)
    m_ineq = -eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x_n)
    =#

    A_eq = vcat( _A_eq, Dm_eq )
    b_eq = vcat( _b_eq, __b_eq )
    A_ineq = vcat( _A_ineq, Dm_ineq )
    b_ineq = vcat( _b_ineq, __b_ineq )

    # compute steepest descent at `x+n`
    d, ω = _steepest_descent_direction( x_n, ∇m, lb, ub, A_eq, b_eq, A_ineq, b_ineq, desc_cfg.normalize)
    return ω, d
end

function compute_descent_step(desc_cfg :: SteepestDescentConfig, mop :: AbstractMOP, scal :: AbstractVarScaler, 
        x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
        data_base, sc :: SurrogateContainer, algo_config :: AbstractConfig, ω, d; kwargs...
    )

    @logmsg loglevel4 "Calculating steepest stepsize."

    x = get_x_scaled( x_it )
    x_n = get_x_scaled( x_it_n )
    
    lb, ub = full_bounds_internal(scal)
    if x ≈ x_n
        Δ = get_delta( x_it )
        lb_eff, ub_eff = _local_bounds(x, Δ, lb, ub )
    else
        # if a normal step was taken, we have to respect the trust region boundary
        # with respect to x but measure Δ from x+n.
        get_delta( x_it )
        lb_eff, ub_eff = _local_bounds( x, get_delta(x_it), lb, ub )
        Δ = intersect_box( x_n, d, lb_eff, ub_eff; return_vals = :pos )
    end

    # scale direction for backtracking as in paper
    # σ is meant to be an inital stepsize for d
    norm_d = norm(d, Inf)
    σ = if Δ <= 1
        # ⇒ if ‖d‖ ≤ Δ, set σ = 1 ≤ Δ/‖d‖
        # ⇒ if Δ < ‖d‖, set σ = Δ/‖d‖ < 1, such that ‖σd‖ = Δ
        min( Δ/norm_d, 1 )
    else
        # Δ > 1
        if norm_d ≈ 1
            # construct matrices for `_intersect_bounds(…)`, to find maximum 
            # σ ≥ 0 such that A( x_n + σd ) - b ≤ 0

            # We need the linear constraint matrices to find a stepsize `σ` with 
            # A * (x_n + σd) - b ≦ 0
            _A_eq, _b_eq, _A_ineq, _b_ineq = transformed_linear_constraints(scal, mop)
            
            # σ_lin = _intersect_bounds(x_n, d, lb_eff, ub_eff, A_eq, b_eq, A_ineq, b_ineq; ret_mode = :pos)

            # Variant 1 - paper variant
            # Dm(x)(n + σd) + m(x) ≤ 0
            # x + n + σ d ≤ ub ⇔ n + σd ≤ ub .- x
            Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal, x)
            Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal, x)    
            m_eq = - eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x)
            m_ineq = - eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x)           
            n = x_n .- x
            #= 
            σ_nonlin = _intersect_bounds(
                n, d, lb_eff .- x, ub_eff .-x , 
                Dm_eq, m_eq, Dm_ineq, m_ineq; ret_mode = :pos
            )
            =#
            #=
            # Variant 2 
            # `d` was calculated using the linearization of the constraints around `x_n`:
            # Dm(x_n)*d + m(x_n) = A d - b ≦ 0.
            # x + n + σd ≤ ub ⇔ σd ≤ ub .- x .- n
            Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal, x_n)    # TODO these should be returned by `compute_normal_step` 
            Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal, x_n)    
            m_eq = eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x_n)
            m_ineq = eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x_n)
            σ_nonlin = _intersect_bounds(
                zeros_like(x), d, lb_eff .- x_n, ub_eff .- xn, 
                Dm_eq, m_eq, Dm_ineq, m_ineq; ret_mode = :pos
            )
            =#
            _intersect_bounds( 
                [x_n; n], [d;d], [lb_eff; lb_eff .- x], [ub_eff; ub_eff .- x],
                cat( _A_eq, Dm_eq; dims = (1,2) ), [_b_eq; m_eq], cat( _A_ineq, Dm_ineq; dims = (1,2) ), [_b_ineq; m_ineq ];
                ret_mode = :pos
            )
        else
            1
        end
    end
    
    if σ > desc_cfg.min_stepsize 
        x₊, mx₊, step = _backtrack( x_n, d, σ, ω, sc, desc_cfg, scal )
        return ω, x₊, mx₊, norm(step, Inf)
    end
    
    return 0, copy(x_n), eval_container_objectives_at_scaled_site( sc, scal, x_n ), 0
end 

# PASCOLETTI-SERAFINI

# Does NLopt honour precision?
@with_kw struct PascolettiSerafiniConfig <: AbstractDescentConfig
    
    "Local utopia point to guide descent search in objective space."
    reference_point :: Vector{Float64} = Float64[]

    "Objective space direction to guide local subproblem solution."
    reference_direction :: Vector{Float64} = Float64[]

    trust_region_factor :: Float64 = 1.0

    max_ps_problem_evals :: Int = -1
    max_ps_polish_evals :: Int = -1

    max_ideal_point_problem_evals :: Int = -1 

    "Algorithm used for the Pascoletti Serafini subproblem minimization."
    main_algo :: Symbol = :GN_ISRES 
    
    "Algorithm to compute local reference point if no point or direction is given."
    reference_algo :: Symbol = :GN_ISRES
    reference_trust_region_factor :: Float64 = 1.1
    
    "Specify local algorithm to polish Pascoletti-Serafini solution. Uses 1/4 of maximum allowed evals."
    ps_polish_algo :: Union{Nothing, Symbol} = nothing 

    @assert all( reference_direction .> 0 ) "The components of the `reference_direction` cannot be negative."
end

"""
    _get_global_dir( cfg, fx )

Return the objective space direction `r ≥ 0`.
If a direction is stored in `cfg`, then it is returned.
If a global ideal point `i` is stored in `cfg`, then `fx .- i` is returned.
Else `nothing` is returned.
"""
function _get_global_dir( cfg :: PascolettiSerafiniConfig, fx )
    if !isempty( cfg.reference_direction )
        return cfg.reference_direction
    elseif !isempty( cfg.reference_point ) 
        return fx .- cfg.reference_point
    else
        return nothing
    end
end

function _min_component( n_vars, x_scaled, lb, ub, opt_handle, eq_handles, ineq_handles, algo_name, MAX_EVALS )
    opt = NLopt.Opt( algo_name, n_vars )
    
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    
    opt.xtol_rel = 1e-3 
    opt.maxeval = MAX_EVALS
    
    opt.min_objective = opt_handle 
    for eq_handle in eq_handles 
        NLopt.equality_constraint!(opt, eq_handle)
    end
    for ineq_handle in ineq_handles
        NLopt.inequality_constraint!(opt, ineq_handle)
    end  
    minf, _ = NLopt.optimize( opt, x_scaled )
    return minf
end

function get_raw_optim_handles( mop, sc, scal )
    objf_handles = collect(get_objectives_optim_handles( sc, scal ))
    
    eq_handles = [ 
        collect(get_nl_eq_constraints_optim_handles(sc, scal)); 
        collect(get_eq_constraints_optim_handles( mop, scal ))
    ]
    ineq_handles = [ 
        collect(get_nl_ineq_constraints_optim_handles( sc, scal ));
        collect(get_ineq_constraints_optim_handles( mop, scal ))
    ]

    return objf_handles, eq_handles, ineq_handles
end

function compute_local_ideal_point( x_it_n, lb_eff, ub_eff, objf_handles, eq_handles, ineq_handles, MAX_EVALS, algo_name)
    x_n = get_x_scaled( x_it_n )
    n_vars = length(x_n)

    return [ 
        _min_component(n_vars, x_n, lb_eff, ub_eff, opt_handle, eq_handles, ineq_handles, 
            algo_name, MAX_EVALS ) for opt_handle = objf_handles 
    ]
end

function _ps_max_evals( desc_cfg, n_vars )
    # total number of sub solver evaluations
    MAX_EVALS = desc_cfg.max_ps_problem_evals < 0 ? 500 * (n_vars+1) : desc_cfg.max_ps_problem_evals
    
    # number of evaluations for the global optimization algo
    if isnothing( desc_cfg.ps_polish_algo )
        MAX_EVALS_global = MAX_EVALS 
        MAX_EVALS_local = 0
    else 
        if desc_cfg.max_ps_polish_evals < 0
            MAX_EVALS_global = Int( floor( MAX_EVALS*3/4 ) )
            MAX_EVALS_local = MAX_EVALS - MAX_EVALS_global
        else
            MAX_EVALS_global = MAX_EVALS 
            MAX_EVALS_local = desc_cfg.max_ps_polish_evals
        end
    end
    return MAX_EVALS_global, MAX_EVALS_local
end

function _get_ps_constraint_functions( objf_handles, mx, dir )
    return [
        function(χ, g)
            if !isempty(g)
                g[1] = -dir[l]
                _g = @view(g[2:end])
            else 
                _g = Float32[]
            end
            ret_val = objf_handle(χ[2:end], _g) - mx[l] - χ[1] * dir[l]
            return ret_val
        end for (l, objf_handle) = enumerate(objf_handles)
    ]
end

function _augment_constraint_functions( constraint_handles, x )
    return [
        function(χ, g)
            #ξ = χ[2:end] .+ x
            ξ = χ[2:end]
            if !isempty(g)
                g[1] = 0
                _g = @view(g[2:end])
            else
                _g = Float32[]
            end
            return c_handle(ξ, _g)
        end
    for c_handle = constraint_handles ]
end

"""
Return objective function for Pascoletti-Serafini, modifying gradient in place.
"""
function _get_ps_objective_func()
    function( χ, g )
        if !isempty(g)
            g[1] = 1
            g[2:end] .= 0
        end
        return χ[1]
    end
end

function _ps_optimization( t0, x, lb, ub, ps_constraints, ps_eq_constraints, ps_ineq_constraints, MAX_EVALS, algo )
    
    n_vars = length(x)

    opt = NLopt.Opt( algo, n_vars + 1 )
    opt.lower_bounds = [-1.0 ; lb ]
    opt.upper_bounds = [ 0.0 ; ub ]
    opt.xtol_rel = 1e-3
    opt.maxeval = MAX_EVALS
    
    opt.min_objective = _get_ps_objective_func()
    
    for constr_func in ps_constraints
        NLopt.inequality_constraint!( opt, constr_func )
    end

    
    for constr_func in ps_eq_constraints
        NLopt.equality_constraint!( opt, constr_func )
    end

    for constr_func in ps_ineq_constraints
        NLopt.inequality_constraint!( opt, constr_func )
    end
    
    @logmsg loglevel4 "Starting PS optimization."
    χ0 = [t0; x]
    τ, χ_min, ret = NLopt.optimize(opt, χ0 )
    
    @logmsg loglevel4 "Finished with $(ret) after $(opt.numevals) model evals."
    
    return τ, χ_min[2:end], ret 
end

function get_criticality( desc_cfg :: PascolettiSerafiniConfig, mop, scal, x_it, x_it_n, 
        data_base, sc, algo_config
    )
    
    @logmsg loglevel3 "Calculating Pascoletti-Serafini descent."
    x = get_x_scaled( x_it )
    x_n = get_x_scaled( x_it_n )
    Xet = eltype(x_n)
    fx_n = get_fx( x_it_n )
    Yet = eltype(fx_n)

    n_vars = length( x_n )
    
    _r = _get_global_dir( desc_cfg, fx_n )

    MAX_EVALS = desc_cfg.max_ideal_point_problem_evals < 0 ? 500 * (n_vars+1) : desc_cfg.max_ideal_point_problem_evals

    # prepare nonlinear optimization:
    objf_handles, eq_handles, ineq_handles = get_raw_optim_handles(mop, sc, scal)

    lb_eff, ub_eff = local_bounds( scal, x, get_delta(x_it) )
    # TODO trust region factor 
    r = if isnothing(_r)
        fx_n .- compute_local_ideal_point( x_it_n, lb_eff, ub_eff, objf_handles, 
            eq_handles, ineq_handles, MAX_EVALS, desc_cfg.reference_algo )
    else
        _r
    end

    @logmsg loglevel4 "Local image direction is $(_prettify(r))"

    mx = eval_container_objectives_at_scaled_site( sc, scal, x_n)

    # If any component is not positive, we are critical
    if any( r .<= 0 )
        return 0, copy(x_n), mx, 0
    end

    MAX_EVALS_global, MAX_EVALS_local = _ps_max_evals( desc_cfg, n_vars )
    ps_constraints = _get_ps_constraint_functions(objf_handles, mx, r)
    ps_eq_constraints = _augment_constraint_functions( eq_handles, x_n )
    ps_ineq_constraints = _augment_constraint_functions( ineq_handles, x_n )
    
    τ, x_min, ret = _ps_optimization(-.5f0, x_n, lb_eff, ub_eff, ps_constraints, ps_eq_constraints, 
        ps_ineq_constraints, MAX_EVALS_global, desc_cfg.main_algo)

    RET_FAIL = (ret == :FAILURE || isinf(τ) || isnan(τ) || any(isnan.(x_min)) )
    
    if MAX_EVALS_local > 0 && !RET_FAIL
        @logmsg loglevel4 "Local polishing for PS enabled."
       
        τ_loc, x_min_loc, ret_loc = _ps_optimization(τ, x_min, lb_eff, ub_eff,
            ps_constraints, ps_eq_constraints, ps_ineq_constraints, MAX_EVALS_local, desc_cfg.ps_polish_algo)

        if !( ret_loc == :FAILURE || isinf(τ_loc) || isnan(τ_loc) || any(isnan.(x_min_loc)) )
            τ, x_min = τ_loc, x_min_loc
        end
    end

    if RET_FAIL
        return 0, copy(x), eval_container_objectives_at_scaled_site(sc, scal, x_n), 0
    else
        ω = Xet(abs( τ ))
        x_trial = Xet.(x_min)
        mx₊ = eval_container_objectives_at_scaled_site( sc, scal, x_trial )
        sl = norm( x .- x_trial, Inf )

        return ω, (x_trial, mx₊, sl)
    end
end

# Directed Search
#=
function compute_descent_step(::Val{:directed_search}, algo_config :: AbstractConfig,
    mop :: AbstractMOP, id :: AbstractIterate, sc :: SurrogateContainer )
    pascoletti_serafini(Val(ds), algo_config, mop, id, sc )
end

function compute_descent_step(::Val{:ds}, algo_config :: AbstractConfig,
    mop :: AbstractMOP, id :: AbstractIterate, sc :: SurrogateContainer )
    @logmsg loglevel3 "Calculating Pascoletti-Serafini descent."
    x = xᵗ(id);
    fx = fxᵗ(id);
    n_vars = num_vars(mop);
    n_out = num_objectives(mop);
    
    # very similar to PS, but inverted sign(s)
    r = begin 
        if !isempty( reference_direction(algo_config) )
            dir = apply_internal_sorting( reference_direction(algo_config), mop )
            if all( dir .>= 0 )
                dir *= -1
            end
        elseif !isempty( reference_point(algo_config))
            dir = apply_internal_sorting(reference_point(algo_config), mop) .- fx
        else
            dir = _local_ideal_point(mop, algo_config, id, sc) .- fx
        end
        dir
    end

    @logmsg loglevel4 "Local image direction is $(_prettify(r))"

    if any( r .>= 0 )
        return 0, copy(x), eval_container_objectives_at_scaled_site(sc, x), 0
    end

    lb, ub = full_bounds_internal( mop );    
    mx = eval_container_objectives_at_scaled_site( sc, x );
    ∇m = eval_container_jacobian_at_scaled_site(sc, x);

    ∇m⁺ = pinv( ∇m )
    CONSTRAINED = !isempty(MOI.get(mop,MOI.ListOfConstraints()))
    if !CONSTRAINED
        dir = ∇m⁺*r;
    else
        dir_prob = JuMP.Model(LP_OPTIMIZER);
        JuMP.set_silent(dir_prob)
        
        # TODO are parameters sensible?
        JuMP.set_optimizer_attribute(dir_prob,"eps_rel",1e-5)
        JuMP.set_optimizer_attribute(dir_prob, "polish", true)

        # slightly adapted version of directed search (allow only non-anscent directions)
        JuMP.@variable(dir_prob, d[1:n_vars] )
        JuMP.@objective(dir_prob, Min, sum(( ∇m * d .- r ).^2) )
        JuMP.@constraint(dir_prob, norm_constraint, -1.0 .<= d .<= 1.0)
        JuMP.@constraint(dir_prob, descent, ∇m*d .<= 0)
        JuMP.@constraint(dir_prob, global_constraint, lb .<= x .+ d .<= ub )

        JuMP.optimize!(dir_prob)
        dir = JuMP.value.(d)
    end
    ω = - maximum(∇m * dir);

    # scale direction for backtracking as in paper
    norm_d = norm(dir,Inf);
    if norm_d > 0
        d_normed = dir ./ norm_d;
        σ = intersect_bounds( mop, x, Δᵗ(id), d_normed; return_vals = :pos )
        # Note: For scalar Δ the above should equal 
        # `σ = norm(d,Inf) < 1 || Δ <= 1 ? min( d, norm(d,Inf) ) : Δ`
        @assert σ >= 0

        x₊, mx₊, step = _backtrack( x, d_normed, σ, ω, sc, strict_backtracking(algo_config) );
        return ω, x₊, mx₊, norm(step, Inf)
    else
        return ω, copy(x), eval_container_objectives_at_scaled_site( sc, x ), 0
    end
end

TODO Re-enable Directed Search
=#

####### Symbol values as quick configuration 

function _cfg_from_symbol( desc_cfg :: Symbol, F :: Type )
    if desc_cfg == :steepest || desc_cfg == :sd || desc_cfg == :steepest_descent
        return SteepestDescentConfig()
    elseif desc_cfg == :ps || desc_cfg == :pascoletti_serafini 
        return PascolettiSerafiniConfig()
    end
end

# DefaultDescent -> Fallback to SteepestDescent
function compute_descent_step( desc_cfg :: Symbol, mop, scal, x_it, x_it_n, data_base, sc, algo_config, args...; kwargs... )
    F = eltype(get_x(x_it))
    true_cfg = _cfg_from_symbol( desc_cfg, F )
    return compute_descent_step( true_cfg, mop, scal, x_it, x_it_n, data_base, sc, algo_config, args...; kwargs... )
end

# DefaultDescent -> Fallback to SteepestDescent
function get_criticality( desc_cfg :: Symbol, mop, scal, x_it, x_it_n, data_base, sc, algo_config, args...; kwargs... )
    F = eltype(get_x(x_it))
    true_cfg = _cfg_from_symbol( desc_cfg, F )
    return get_criticality( true_cfg, mop, scal, x_it, x_it_n, data_base, sc, algo_config, args...; kwargs... )
end

###################
function compute_normal_step( mop :: AbstractMOP, scal :: AbstractVarScaler, x_it :: AbstractIterate,
    data_base, sc :: SurrogateContainer, algo_config :: AbstractConfig;
    variable_radius :: Bool = false )
    
    x = get_x_scaled( x_it )
    Xet = eltype(x)
    κ_Δ = filter_kappa_delta( algo_config )

    n_vars = length(x)

    A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints( scal, mop )
    l_e = get_eq_const( x_it )  # == A_eq * x - b_eq
    l_i = get_ineq_const( x_it ) # == A_ineq * x - b_ineq 

    Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal,x)    
    Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal,x)    
    m_eq = eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x)
    m_ineq = eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x)

    opt_problem = JuMP.Model( LP_OPTIMIZER )
    JuMP.set_silent(opt_problem)

    JuMP.set_optimizer_attribute( opt_problem, "eps_rel", Xet(1e-5) );
    JuMP.set_optimizer_attribute( opt_problem, "polish", true );

    JuMP.@variable(opt_problem, n[1:n_vars])
    JuMP.@variable(opt_problem, α >= 0 )

    if variable_radius
        JuMP.@variable(opt_problem, 0 <= del <= delta_max(algo_config) )
        JuMP.@objective( opt_problem, Min, del )
        
        # this must be fullfilled in order for `n` to be compatible
        # results from `min(1, κ_μ Δ^μ) = 1`.
        # if this is not the case, `min(1, κ_μ*Δ^μ) ≤ 1` still makes it
        # a necessary condition
        JuMP.@constraint( opt_problem, α <= κ_Δ * del )
    else
        JuMP.@objective(opt_problem, Min, α )
        # NOTE we don't use the trust region constraint normally so that in case
        # of a linearly constrained problem we can use `n` as `r`
        # and else use `n` as an initial guess for the nonlinear restoration
    end

    JuMP.@constraint( opt_problem, -α .<= n)
    JuMP.@constraint( opt_problem, n .<= α)

    lb, ub = full_bounds_internal( scal )
    JuMP.@constraint( opt_problem, lb .<= x .+ n .<= ub )

    # A (x + n) - b ≦ 0 ⇔ An -b + Ax ≦ 0 ⇔ An - (b-Ax) ≦ 0 ⇔ An - (-l) ≦ 0
    JuMP.@constraint(opt_problem, linear_eq_const, A_eq * n .+ l_e .== 0)
    JuMP.@constraint(opt_problem, linear_ineq_const, A_ineq * n .+ l_i .<= 0)
    
    JuMP.@constraint(opt_problem, nl_eq_const, Dm_eq * n .+ m_eq .== 0)
    JuMP.@constraint(opt_problem, nl_ineq_const, Dm_ineq * n .+ m_ineq .<= 0)

    JuMP.optimize!(opt_problem)
    
    if JuMP.termination_status(opt_problem) == JuMP.MathOptInterface.INFEASIBLE
        return fill(MIN_PRECISION(NaN64), n_vars), -MIN_PRECISION(Inf)  # this should happen anyways and triggers restoration
    end

    _Δ = !variable_radius ? -1 : JuMP.value( del )
    _n = JuMP.value.(n)
    n = Xet.(_project_into_box( x .+ _n, lb, ub ) .- x)     # we _project_into_box because it might happen that there is some small constraint violation
    return n, Xet(_Δ)
end