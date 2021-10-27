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
#  ::AbstractSuperDB, sc :: SurrogateContainer )`
# should be defined, because this is then called internally
# 
# Return: a criticality measure the new trial point (same eltype as x),
# the new trial point model values and the steplength (inf-norm)

# method to only compute criticality
function get_criticality( mop :: AbstractMOP, scal :: AbstractVarScaler, x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
        data_base :: AbstractSuperDB, sc :: SurrogateContainer, algo_config :: AbstractConfig;
        kwargs...
    )

    return get_criticality( descent_method( algo_config ), mop, scal, x_it, x_it_n, data_base, sc, algo_config; kwargs... ) 
end

# methods to compute a local descent step
function compute_descent_step( mop :: AbstractMOP, scal :: AbstractVarScaler, x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
       data_base :: AbstractSuperDB, sc :: SurrogateContainer, algo_config :: AbstractConfig, args...; kwargs...
    )
    return compute_descent_step( descent_method( algo_config ), mop, scal, x_it, x_it_n, data_base, sc, algo_config, args... ; kwargs...)
end

# fallback, if `get_criticality` allready does the whole job: simply return `ω` and `data`
function compute_descent_step( cfg :: AbstractDescentConfig, mop :: AbstractMOP, scal :: AbstractVarScaler, x_it :: AbstractIterate, x_it_n :: AbstractIterate, 
        data_base :: AbstractSuperDB, sc :: SurrogateContainer, algo_config :: AbstractConfig; kwargs...
    )
    ω, data = get_criticality(cfg, mop, scal, x_it, x_it_n, data_base, sc, algo_config; kwargs...)
    return (ω, data...)
end

# ## STEEPEST DESCENT AND BACKTRACKING

@with_kw struct SteepestDescentConfig <: AbstractDescentConfig

    "Require a descent in all model objective components."
    strict_backtracking :: Bool = true 

    armijo_const_rhs :: Float64 = 1e-3
    armijo_const_shrink :: Float64 = .5 

    max_loops :: Int = 30
    min_stepsize :: Float64 = -1.0
end

"Provided `x` and the (surrogate) jacobian `∇F` at `x`, as well as bounds `lb` and `ub`, return steepest multi descent direction."
function _steepest_descent_direction(
    x :: AbstractVector{F}, ∇F :: Mat, lb :: Vec, ub :: Vec,
    A_eq = [], b_eq = [], A_ineq = [], b_ineq = [],
    ) where F <: AbstractFloat
    
    n = length(x);
    try 
        opt_problem = JuMP.Model( OSQP.Optimizer );
        JuMP.set_silent(opt_problem)

        JuMP.set_optimizer_attribute( opt_problem, "eps_rel", 1e-5 )
        JuMP.set_optimizer_attribute( opt_problem, "polish", true )

        JuMP.@variable(opt_problem, α)  # criticality measure with flipped sign
        JuMP.@objective(opt_problem, Min, α)

        JuMP.@variable(opt_problem, d[1:n]) # steepest descent direction 
      
        JuMP.@constraint(opt_problem, descent_constraints, ∇F*d .<= α)
        JuMP.@constraint(opt_problem, norm_constraints, -1 .<= d .<= 1 )

        # original problem constraints (have to be transformed beforehand)
        JuMP.@constraint(opt_problem, global_scaled_var_bounds, lb .<= x .+ d .<= ub);
        if !isempty(A_eq)
            @assert size(A_eq, 1) == length(b_eq) "Equality constraint dimension mismatch."
            JuMP.@constraint(opt_problem, linear_eq_constraints,  A_eq*d .+ b_eq .== 0 )
        end
        
        if !isempty(A_ineq)
            @assert size(A_ineq, 1) == length(b_ineq) "Equality constraint dimension mismatch."
            JuMP.@constraint(opt_problem, linear_ineq_constraints, A_ineq*d .+ b_ineq .<= 0 )
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
function _backtrack( x :: AbstractVector{F}, dir, ω, sc, cfg, scal ) where F<:AbstractFloat

    MIN_STEPSIZE = cfg.min_stepsize >= 0 ? cfg.min_stepsize : eps(F)
    MAX_LOOPS = cfg.max_loops
    strict_backtracking = cfg.strict_backtracking 
    α = cfg.armijo_const_shrink
    c = cfg.armijo_const_rhs

    step_size = 1 

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

    x_n = get_x_scaled(x_it_n)
    x = get_x_scaled(x_it)

    ∇m = eval_container_objectives_jacobian_at_scaled_site(sc, scal, x)

    lb, ub = full_bounds_internal( scal )
    
    # For computing a step `t`, starting at `x + n`, the linear constraints are 
    # A⋅(x + n + t) + b ≤ 0
    # As `x` and `n` are constant we use `b̃ = Ax + An + b` 
    # which happens to be saved in `x_it_n` for constraints 
    # A ⋅ t + b̃ ≦ 0
    _b_eq = get_eq_const(x_it_n)   # A(x+n) + b
    _b_ineq = get_ineq_const(x_it_n)
    _A_eq, _, _A_ineq, _ = transformed_linear_constraints(scal, mop)
   
    # In theory, We approximate the nonlinear constraints via 
    # c(ξ) = m(x) + Dm(x)⋅(x - ξ)
    # Hence, when looking for a step `t` (taken from `x + n`),
    # we get the constraint approximations 
    # m(x) + Dm(x)⋅(n + t) ≦ 0

    # Here, we actually do the Taylor Expansion around `x+n` and get constraints 
    # m(x + n) + Dm(x+n)⋅t ≦ 0
    # The only reason to do so is to directly use `b = m(x_n)` and `A = Dm(x_n)`:    
    Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal,x_n)    # TODO these should be returned by `compute_normal_step` 
    Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal,x_n)    
    m_eq = eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x_n)
    m_ineq = eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x_n)
    
    A_eq = vcat( _A_eq, Dm_eq )
    b_eq = vcat( _b_eq, m_eq )
    A_ineq = vcat( _A_ineq, Dm_ineq )
    b_ineq = vcat( _b_ineq, m_ineq )

    # compute steepest descent at `x+n`
    d, ω = _steepest_descent_direction( x_n, ∇m, lb, ub, A_eq, b_eq, A_ineq, b_ineq)
    return ω, d
end

function compute_descent_step( desc_cfg :: SteepestDescentConfig, mop, scal, x_it, x_it_n, 
        data_base, sc, algo_config, ω, d        
    )

    @logmsg loglevel4 "Calculating steepest stepsize."

    x = get_x_scaled( x_it )
    x_n = get_x_scaled( x_it_n )
    
    lb_eff, ub_eff = full_lower_bounds_internal(scal)
    if x ≈ x_n
        Δ = get_delta( x_it )
    else
        # if a normal step was taken, we have to respect the trust region boundary
        # with respect to x_n = x + n
        lb_eff, ub_eff = _local_bounds( x, get_delta(x_it), lb, ub )
        Δ = _intersect_box( x_n, d, lb_eff, ub_eff; ret_mode = :pos )
    end

    # scale direction for backtracking as in paper
    norm_d = norm(d, Inf)
    σ = if Δ <= 1
        min( Δ/norm_d, 1 )
    else
        # Δ > 1
        if norm_d ≈ 1
            # We need the linear constraint matrices to find a stepsize `σ` with 
            # A * x_n + σt + b ≦ 0
            _A_eq, _b_eq, _A_ineq, _b_ineq = transformed_linear_constraints(scal, mop)
            
            # Again we use a linearization of the constraints around `x_n` to find `σ` with
            # m(x_n) + Dm(x_n)*σt ≦ 0:
            Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal, x_n)    # TODO these should be returned by `compute_normal_step` 
            Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal, x_n)    
            m_eq = eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x_n)
            m_ineq = eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x_n)

            A_eq = vcat( _A_eq, Dm_eq )
            b_eq = vcat( _b_eq, m_eq )
            A_ineq = vcat( _A_ineq, Dm_ineq )
            b_ineq = vcat( _b_ineq, m_ineq )

            _intersect_bounds(x_n, d, lb_eff, ub_eff, A_eq, b_eq, A_ineq, b_ineq)
        else
            1
        end
    end
    
    if σ > desc_cfg.min_stepsize 
        x₊, mx₊, step = _backtrack( x_n, σ * d, ω, sc, desc_cfg, scal )
        return ω, x₊, mx₊, norm(step, Inf)
    end
    
    return 0, copy(x_n), eval_container_objectives_at_scaled_site( sc, scal, x_n ), 0
end 


function compute_descent_step( desc_cfg :: SteepestDescentConfig, mop, scal, x_it, x_it_n, data_base, sc, algo_config; kwargs...)
    ω, d = get_criticality( desc_cfg, mop, scal, x_it, x_it_n, data_base, sc, algo_config; kwargs... )
    return compute_descent_step(desc_cfg, mop, scal, x_it, x_it_n, data_base, sc, algo_config, ω, d; kwargs... )
end

# PASCOLETTI-SERAFINI
#=
# TODO would it matter to allow single precision reference_point?
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
end

function _get_local_dir( cfg :: PascolettiSerafiniConfig, fx, sorting_indices )
    if !isempty( cfg.reference_direction )
        return cfg.reference_direction[ sorting_indices ]
    elseif !isempty( cfg.reference_point ) 
        return fx .- cfg.reference_point[ sorting_indices ]
    else
        return nothing
    end
end

function _min_component( x, n_vars, lb, ub, algo_name, MAX_EVALS, opt_handle )
    opt = NLopt.Opt( algo_name, n_vars )
    opt.lower_bounds = lb
    opt.upper_bounds = ub
    opt.xtol_rel = 1e-3 
    opt.maxeval = MAX_EVALS
    #opt.min_objective = get_optim_handle( sc, mop, l )
    opt.min_objective = opt_handle 
    minf, _ = NLopt.optimize( opt, x );
    return minf
end


function _local_ideal_point( x :: AbstractVector{F}, lb, ub, optim_handles, max_evals, algo_name ) where F
    @logmsg loglevel4 "Computing local ideal point. This can take a bit…"
    
    num_objectives = length( optim_handles )
    n_vars = length(x)
    
    MAX_EVALS = max_evals < 0 ? 500 * (n_vars+1) : max_evals
    
    # preallocate local ideal point:
    ȳ = fill( -F(Inf), num_objectives )

    # minimize each individual scalar surrogate output 
    for (l, opt_handle) in enumerate( optim_handles )
        ȳ[l] = _min_component(x, n_vars, lb, ub, algo_name, MAX_EVALS, opt_handle)
    end
    return ȳ
end


function get_criticality( desc_cfg :: PascolettiSerafiniConfig, mop, iter_data, data_base, sc, algo_config )
    
    @logmsg loglevel3 "Calculating Pascoletti-Serafini descent."
    x = get_x( iter_data )
    fx = get_fx( iter_data )
    Δ = get_delta( iter_data )
    n_vars = length( x )
    n_out = num_objectives(mop);
    
    r = _get_local_dir( desc_cfg, fx, mop )
    if isnothing(r)
        opt_handles = [ get_optim_handle(sc, l) for l = 1 : num_objectives ]
        lb, ub = local_bounds(mop, x, desc_cfg.reference_trust_region_factor * Δ )
        ip = _local_ideal_point(x,lb, ub, opt_handles, desc_cfg.MAX_EVALS, desc_cfg.algo_name)
        r = fx .- ip 
    end

    @logmsg loglevel4 "Local image direction is $(_prettify(r))"

    # If any component is not positive, we are critical
    if any( r .<= 0 )
        return 0, copy(x), eval_container_objectives_at_scaled_site(sc, x), 0
    end

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

    mx = eval_container_objectives_at_scaled_site( sc, x );
    lb, ub = local_bounds(mop, x, decs_cfg.trust_region_factor * Δ)
    
    τ, χ_min, ret = _ps_optimization(sc, desc_cfg.main_algo, lb, ub,
        MAX_EVALS_global, [-0.5;x], mx, r, n_vars, n_out)

    if MAX_EVALS_local > 0
        @logmsg loglevel4 "Local polishing for PS enabled."
       
        τₗ, χ_minₗ, retₗ = _ps_optimization(sc, desc_cfg.ps_polish_algo, lb, ub,
            MAX_EVALS_local, χ_min, mx, r, n_vars, n_out);
        if !(retₗ == :FAILURE || isinf(τₗ) || isnan(τₗ) || any(isnan.(χ_minₗ)) )
            τ, χ_min, ret = τₗ, χ_minₗ, retₗ
        end
    end

    if (ret == :FAILURE || isinf(τ) || isnan(τ) || any(isnan.(χ_min)) )
        return 0, copy(x), eval_container_objectives_at_scaled_site(sc, x), 0
    else
        ω = abs( τ );
        x₊ = χ_min[2 : end];
        mx₊ = eval_container_objectives_at_scaled_site( sc, x₊ );
        sl = norm( x .- x₊, Inf );

        return ω, x₊, mx₊, sl
    end
end

"Construct and solve Pascoletti Serafini subproblem using surrogates from `sc`."
function _ps_optimization( sc :: SurrogateContainer, algo :: Symbol, 
    lb :: Vec, ub :: Vec, MAX_EVALS :: Int, χ0 :: AbstractVector{F}, 
    mx :: Vec, r :: Vec, n_vars :: Int, n_out :: Int ) where F
    
    opt = NLopt.Opt( algo, n_vars + 1 );
    opt.lower_bounds = [-1.0 ; lb ];
    opt.upper_bounds = [ 0.0 ; ub ];
    opt.xtol_rel = 1e-3;
    opt.maxeval = MAX_EVALS;
    opt.min_objective = _get_ps_objective_func();
    
    #NLopt.inequality_constraint!( opt, _get_ps_constraint_func(sc,mx,r), 1e-12 );
    for l = 1 : n_out 
        NLopt.inequality_constraint!( opt, _get_ps_constraint_func(sc, mx, r, l), eps(F) )
    end

    @logmsg loglevel4 "Starting PS optimization."
    τ, χ_min, ret = NLopt.optimize(opt, χ0 );
    @logmsg loglevel4 "Finished with $(ret) after $(opt.numevals) model evals."
    
    return τ, χ_min, ret 
end

#=
NOTE Does not work using the whole jacobian somehow
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
        return eval_container_objectives_at_scaled_site(sc, χ[2:end]) .- mx .- χ[1] .* dir
    end
end
=#

"""
    _get_ps_constraint_func( sc :: SurrogateContainer, mx, dir, l )

Return the `l`-th (possibly non-linear) constraint function 
for Pascoletti-Serafini.
`dir` .>= 0 is the image direction;
`χ = [t;x]` is the augmented variable vector;

"""
function _get_ps_constraint_func( sc :: SurrogateContainer, mx, dir, l )
    return function(χ, g)
        if !isempty(g)
            g[1] = -dir[l]
            g[2:end] .= get_gradient(sc, χ[2:end], l)
        end
        ret_val = eval_container_objectives_at_scaled_site(sc, χ[2:end], l) - mx[l] - χ[1] * dir[l]
        return ret_val
    end
end

"""
Return objective function for Pascoletti-Serafini, modifying gradient in place.
"""
function _get_ps_objective_func()
    function( χ, g )
        if !isempty(g)
            g[1] = 1.0;
            g[2:end] .= 0.0;
        end
        return χ[1]
    end
end
=#

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
        dir_prob = JuMP.Model(OSQP.Optimizer);
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
    data_base :: AbstractSuperDB, sc :: SurrogateContainer, algo_config :: AbstractConfig;
    variable_radius :: Bool = false )
    
    x = get_x_scaled( x_it )
    κ_Δ = filter_kappa_delta( algo_config )

    n_vars = length(x)

    A_eq, b_eq, A_ineq, b_ineq = transformed_linear_constraints( scal, mop )
    l_e = get_eq_const( x_it )  # == A_eq * x + b_eq
    l_i = get_ineq_const( x_it ) # == A_ineq * x + b_ineq 

    Dm_eq = eval_container_nl_eq_constraints_jacobian_at_scaled_site(sc,scal,x)    
    Dm_ineq = eval_container_nl_ineq_constraints_jacobian_at_scaled_site(sc,scal,x)    
    m_eq = eval_container_nl_eq_constraints_at_scaled_site(sc,scal,x)
    m_ineq = eval_container_nl_ineq_constraints_at_scaled_site(sc,scal,x)

    opt_problem = JuMP.Model( OSQP.Optimizer )
    JuMP.set_silent(opt_problem)

    JuMP.set_optimizer_attribute( opt_problem, "eps_rel", 1e-5 );
    JuMP.set_optimizer_attribute( opt_problem, "polish", true );

    JuMP.@variable(opt_problem, n[1:n_vars])
    JuMP.@variable(opt_problem, α >= 0 )

    if variable_radius
        JuMP.@variable(opt_problem, 0 <= del <= get_delta_max(algo_config) )
        JuMP.@objective( opt_problem, Min, del )
        
        # this must be fullfilled in order for `n` to be compatible
        # results from `min(1, κ_μ Δ^μ) = 1`.
        # if this is not the case, `min(1, κ_μ*Δ^μ) \le 1` still makes it
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

    JuMP.@constraint(opt_problem, linear_eq_const, l_e .+ A_eq * n .== 0)
    JuMP.@constraint(opt_problem, linear_ineq_const, l_i .+ A_ineq * n .<= 0)
    
    JuMP.@constraint(opt_problem, nl_eq_const, Dm_eq * n .+ m_eq .== 0)
    JuMP.@constraint(opt_problem, nl_ineq_const, Dm_ineq * n .+ m_ineq .<= 0)

    JuMP.optimize!(opt_problem)
    
    if JuMP.termination_status(opt_problem) == JuMP.MathOptInterface.INFEASIBLE
        return fill(MIN_PRECISION, n_vars)  # this should happen anyways and triggers restoration
    end

    _Δ = !variable_radius ? -1 : JuMP.value( del )
    return JuMP.value.(n), _Δ
end