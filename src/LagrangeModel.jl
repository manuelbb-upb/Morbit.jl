using DynamicPolynomials

using FileIO;

@with_kw mutable struct LagrangePoly 
    p :: AbstractPolynomialLike 
    grad_poly :: Union{Nothing, Vector{<:AbstractPolynomialLike}} = nothing
    res :: Result = init_res(Res)  
    # TODO check if `res` is a reference (good) or copy (bad for memory)
    # if the latter, store only values at interpolation sites                                
end
Broadcast.broadcastable(lp :: LagrangePoly ) = Ref(lp);

@with_kw mutable struct LagrangeModel <: SurrogateModel
    n_vars :: Int;  # stored for convenience
    n_out :: Int;
    basis :: Vector{LagrangePoly} = LagrangePoly[];
    vals :: RVecArr = RVec[];
    fully_linear :: Bool = false;
    out_indices :: Vector{Int} = Int[];
end

@with_kw mutable struct LagrangeConfig <: SurrogateConfig
    degree :: Int = 2;
    θ_enlarge :: Real = 2;

    "Acceptance Parameter in first Poisedness-Algorithm."
    ε_accept :: Real = 1e-8;
    "Quality Parameter in Λ-Poisedness Algorithm."
    Λ :: Real = 1.5;

    allow_not_linear :: Bool = false;

    optimized_sampling :: Bool = true;

    # # if optimized_sampling = false, shall we try to use saved sites?
    save_path :: Union{AbstractString, Nothing} = nothing;
    io_lock :: Union{Nothing, ReentrantLock, Threads.SpinLock} = nothing;

    algo1_max_evals :: Union{Nothing,Int} = nothing;    # nothing = automatic
    algo2_max_evals :: Union{Nothing,Int} = nothing;

    algo1_solver :: Symbol = :LD_MMA
    algo2_solver :: Symbol = algo1_solver;

    max_evals :: Int64 = typemax(Int64);
end

# overwrite lock and unlock so we can use `nothing` as a lock
function Base.lock(::Nothing) end
function Base.unlock(::Nothing) end

max_evals( cfg::LagrangeConfig)::Int=cfg.max_evals;
function max_evals!(cfg::LagrangeConfig, N :: Int)::Nothing
    cfg.max_evals = N 
end
fully_linear( lm::LagrangeModel ) :: Bool = lm.fully_linear;

combinable(::LagrangeConfig) ::Bool = true;

function combine(cfg1 :: LagrangeConfig, cfg2 :: LagrangeConfig) :: LagrangeConfig
    cfg1
end

@with_kw mutable struct LagrangeMeta <: SurrogateMeta
    interpolation_indices :: Vector{<:Union{Nothing,Int}} = Union{Nothing,Int}[];
    #lagrange_basis :: Vector{Any} = [];
end

function _init_model( cfg :: LagrangeConfig, objf :: AbstractObjective,
    mop :: AbstractMOP, id :: AbstractIterData, ac :: AbstractConfig ) :: Tuple{LagrangeModel, LagrangeMeta}
    lm = LagrangeModel(; 
        n_vars = num_vars(objf),
        n_out = num_outputs(objf),
        out_indices = copy(output_indices( objf, mop)),
        basis = _canonical_basis( num_vars( mop ), cfg.degree ),
    );
    # prepare initial basis of polynomial space 
    lmeta = LagrangeMeta();
    return update_model(lm, objf, lmeta, mop, id, ac;
        ensure_fully_linear = true)
end

function _eval_new_sites!( lm :: LagrangeModel, lmeta :: LagrangeMeta, 
        mop :: AbstractMOP, id :: AbstractIterData )  :: Nothing

    basis_results = [lp.res for lp ∈ lm.basis];
    _eval_and_store_new_results!(id, basis_results, mop);
    #@show  [ get_id(res) for res ∈ basis_results ];
    lmeta.interpolation_indices = [ convert(Union{Int,Nothing},get_id(res)) for res ∈ basis_results ];
    nothing 
end

function _set_gradient!(lp :: LagrangePoly) :: Nothing
    lp.grad_poly = differentiate.(lp.p, DynamicPolynomials.variables(lp.p));
    nothing
end

function _update_model_optimized( lm :: LagrangeModel, objf :: AbstractObjective, lmeta :: LagrangeMeta,
    mop :: AbstractMOP, id :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false):: Tuple{LagrangeModel, LagrangeMeta}
    cfg = model_cfg(objf) :: LagrangeConfig;
    
    make_basis_poised!( 
        lm.basis, id, mop ; 
        Δ_factor = cfg.θ_enlarge, ε_accept = cfg.ε_accept,
        max_solver_evals = cfg.algo1_max_evals,
        solver = cfg.algo1_solver
    );
    
    if ensure_fully_linear || !cfg.allow_not_linear
        make_basis_lambda_poised!(
            lm.basis, id, mop;
            Δ_factor = cfg.θ_enlarge, Λ = cfg.Λ, max_solver_evals = cfg.algo2_max_evals, 
            solver = cfg.algo2_solver
        );
        lm.fully_linear = true;
    end
    
    _eval_new_sites!( lm, lmeta, mop, id);
    _set_gradient!.(lm.basis);

    return lm, lmeta;
end 

@memoize IdDict function _get_unit_basis( n_vars :: Int, cfg :: LagrangeConfig ) :: Vector{LagrangePoly}
    return _get_unit_basis( Val(!isnothing( cfg.save_path ) ) , n_vars, cfg ) 
end

function _get_unit_basis( use_saved :: Val{true}, n_vars :: Int, cfg :: LagrangeConfig )
    load_successful = true;
    unit_sites = [];
    lock(cfg.io_lock) 
    try 
        file_dict = load(cfg.save_path);
        unit_sites = file_dict["sites"];
    catch
        stacktrace( catch_backtrace() )
        @warn "Could not load files from file. Calculating a poised set."
        load_successful = false;
    end
    
    if load_successful
        # check if the sites are any goodif length( unit_sites[1] ) != n_vars || 
        length( unit_sites ) != binomial( n_vars + cfg.degree, n_vars )
        if any( any( ( s .< 0 ) .| ( s .> 1 ) ) for s ∈ unit_sites )
            @warn "Loaded sites not suited for interpolation."
            load_successful = false;
        end
    end

    if !load_successful
        unit_basis = _get_unit_basis( Val(false), n_vars, cfg )
    else 
        @logmsg loglevel4 "Loaded sites from $(cfg.save_path)"
        unit_basis = copy( _canonical_basis( n_vars, cfg.degree ) );
        for (i,x) ∈ enumerate(unit_sites)
            unit_basis[i].res = init_res( Res, x );
            _normalize!(unit_basis[i]);
            _orthogonalize!(unit_basis, i);
        end
    end

    unlock(cfg.io_lock)
        
    return unit_basis;
end

function _get_unit_basis( use_saved :: Val{false}, n_vars :: Int, cfg :: LagrangeConfig )
    unit_basis = _unit_basis( n_vars, cfg.degree; 
        solver1 = cfg.algo1_solver, solver2 = cfg.algo2_solver,
        max_evals1 = cfg.algo1_max_evals, max_evals2 = cfg.algo2_max_evals,
        Λ = cfg.Λ
    );
    if !isnothing( cfg.save_path )
        #lock(cfg.io_lock) 
        # better wrap the outer call in io_lock for parallelization
        try 
            unit_sites = [ get_site(ub.res) for ub ∈ unit_basis ];
            save(cfg.save_path, Dict( "sites" => unit_sites ) );
        catch
            stacktrace( catch_backtrace() )
            @warn "Could not save unit sites."
        end
        #unlock(cfg.io_lock);
    end
    return unit_basis;
end

 function _update_model_unoptimized(
    lm :: LagrangeModel, objf :: AbstractObjective, lmeta :: LagrangeMeta,
    mop :: AbstractMOP, id :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false ):: Tuple{LagrangeModel, LagrangeMeta}
    
    cfg = model_cfg(objf);
    unit_basis = _get_unit_basis( num_vars(mop), cfg ); # this is memoized
    lb_eff, ub_eff = local_bounds(mop, xᵗ(id), Δᵗ(id) * cfg.θ_enlarge);
    lm.basis = _scale_unit_basis( unit_basis, lb_eff, ub_eff );
    lm.fully_linear = true;

    _eval_new_sites!( lm, lmeta, mop, id);
    _set_gradient!.(lm.basis);
    return  lm, lmeta ;
end

function _scale_unit_basis( unit_basis :: Vector{LagrangePoly}, lb :: RVec, ub :: RVec )
    χ = variables( unit_basis[1].p );

    # combine the polyonmial from unit_basis (working on [0,1]^n)
    # with a poly that maps the current trust region to [0,1]^n 
    scaling_poly = (χ .- lb) ./ (ub .- lb)

    scaled_basis = LagrangePoly[];
    for up ∈ unit_basis
        lp = LagrangePoly(;
            p = subs( up.p, χ => scaling_poly ),
            # we `unscale` the sites from [0,1]^n to correspond to sites in 
            # current trust region
            res = init_res( Res, _unscale( get_site( up.res ), lb, ub ) )
        );
        push!(scaled_basis, lp);
    end
    return scaled_basis 
end

function update_model( lm :: LagrangeModel, objf :: AbstractObjective, lmeta :: LagrangeMeta,
    mop :: AbstractMOP, id :: AbstractIterData, algo_config :: AbstractConfig; 
    ensure_fully_linear :: Bool = false):: Tuple{LagrangeModel, LagrangeMeta}
    
    @logmsg loglevel3 "Building LagrangeModel with indices $(output_indices(objf, mop))."
    cfg = model_cfg(objf);
    if cfg.optimized_sampling
        @info "optimized"
        lm,lmeta = _update_model_optimized(lm,objf,lmeta,mop,id,algo_config; ensure_fully_linear);
    else
        lm,lmeta = _update_model_unoptimized(lm,objf,lmeta,mop,id,algo_config; ensure_fully_linear);
    end
    
    # TODO remove this
    #=
    oi = lm.out_indices #output_indices( objf, mop )
    for lp in lm.basis 
        try
            @assert eval_poly( lp, get_site(lp.res) ) ≈ 1
            @assert all(get_value(lp.res)[oi] .≈ eval_models( lm, get_site(lp.res) ))
        catch e
            @warn "Imprecise Lagrange Models."
            @info get_value(lp.res)[oi] .- eval_models( lm, get_site(lp.res) )
        end
    end
    =#
    return lm, lmeta;
end

function improve_model( lm :: LagrangeModel, objf :: AbstractObjective, lmeta :: LagrangeMeta,
    mop :: AbstractMOP, id :: AbstractIterData, :: AbstractConfig; ensure_fully_linear :: Bool = false):: Tuple{LagrangeModel, LagrangeMeta}

    @logmsg loglevel3 "Performing an improvement step for LagrangeModel with indices $(output_indices(objf, mop))."

    cfg = model_cfg( objf );
    make_basis_lambda_poised!( lm.basis, id, mop; 
        Δ_factor = cfg.θ_enlarge, Λ = cfg.Λ, 
        max_solver_evals = cfg.algo2_max_evals, solver = cfg.algo2_solver 
    );
    lm.fully_linear = true;
    _eval_new_sites!( lm, lmeta, mop, id);
    _set_gradient!.(lm.basis)
    return lm, lmeta
end

function eval_models( lm :: LagrangeModel, x̂ :: RVec, ℓ :: Int ) :: Real
    sum( get_value(lp.res)[lm.out_indices[ ℓ ]] * eval_poly(lp.p, x̂) for lp ∈ lm.basis)    
end

function eval_models( lm :: LagrangeModel, x̂ :: RVec ) :: RVec
    sum( get_value(lp.res)[ lm.out_indices ] * eval_poly(lp.p, x̂) for lp ∈ lm.basis)    
end

function get_gradient( lm :: LagrangeModel, x̂ :: RVec, ℓ :: Int ) :: RVec
    sum( get_value(lp.res)[lm.out_indices[ ℓ ]] * eval_poly(lp.grad_poly, x̂) for lp ∈ lm.basis )
end

function get_jacobian( lm :: LagrangeModel, x̂ :: RVec ) :: RMat
    grad_evals = [ eval_poly(lp.grad_poly, x̂) for lp ∈ lm.basis ];
    return Matrix(transpose( hcat(
        [ sum( get_value(lm.basis[i].res)[ lm.out_indices[ ℓ ] ] * grad_evals[i] 
            for i = eachindex(grad_evals) )               
                for ℓ = 1 : lm.n_out ]... 
    )));
end

#@memoize IdDict
function _unit_basis( n_vars :: Int, degree :: Int ;
        solver1 :: Symbol, solver2 :: Symbol, max_evals1 :: Union{Nothing,Int}, 
        max_evals2 :: Union{Nothing,Int}, Λ :: Real ) :: Vector{LagrangePoly}
    lb = zeros(n_vars);
    ub = ones(n_vars);

    basis = _canonical_basis( n_vars, degree );
    
    candidates = [init_res(Res, lb .+ .5 .* (ub .- lb) ),];
    _make_poised!( basis, lb, ub, candidates; ε_accept = 0, max_solver_evals = max_evals1, solver = solver1 );
    _make_lambda_poised!( basis, lb, ub; Λ = Λ, max_solver_evals = max_evals2, solver = solver2);
    return basis
end

function _make_poised!( basis :: Vector{LagrangePoly}, lb :: RVec, ub :: RVec, candidates :: Vector{<:Result};
    ε_accept ::Real, solver :: Symbol, max_solver_evals :: Union{Nothing,Int}, loop_start :: Int = 1 ) :: Nothing
    n_vars = length(lb);
    if isnothing( max_solver_evals)
        max_solver_evals = 300*(n_vars+1);   # TODO is sensible?
    end
    p = length( basis )
    for i = loop_start : p 
        X = get_site.( candidates );
        l_max, j = isempty( X ) ? (0, 0) : findmax( abs.( eval_poly.( basis[i], X) ) );
        if l_max > ε_accept
            @logmsg loglevel4 " 1.$(i)) Recycling a point from the database."
            res = candidates[j];
            deleteat!( candidates, j);
        else
            @logmsg loglevel4 " 1.$(i)) Computing a poised point by Optimization."
            opt = NLopt.Opt( solver , n_vars )
            opt.lower_bounds = lb;
            opt.upper_bounds = ub;
            opt.maxeval = max_solver_evals;
            opt.xtol_rel = 1e-3;
            opt.max_objective = (x,g) -> abs( eval_poly( basis[i], x) )
            x₀ = _rand_box_point(lb, ub);
            (_, x, ret) = NLopt.optimize(opt, x₀)
        
            res = init_res(Res, x);
        end
        basis[i].res = res;
        _normalize!(basis[i]);
        _orthogonalize!(basis, i);
    end
    nothing;
end 

function make_basis_poised!( basis :: Vector{LagrangePoly}, id :: AbstractIterData, mop :: AbstractMOP;
    Δ_factor :: Real = 1, ε_accept :: Real = 1e-8, solver :: Symbol = :GN_AGS, max_solver_evals :: Union{Int,Nothing} = nothing )
    @logmsg loglevel4 "Finding a poised set..."
    
    x = xᵗ( id );
    N = length(x);
    
    # always include current iterate
    @assert xᵗ_index( id ) isa XInt
    basis[1].res = get_result( id, xᵗ_index(id));
    _normalize!(basis[1]);
    _orthogonalize!(basis, 1);

    # find all points in database in current trust region
    lb_eff, ub_eff = local_bounds(mop, x, Δᵗ(id) * Δ_factor);
    box_indices = find_points_in_box(
        id, lb_eff, ub_eff;
        exclude_indices = [xᵗ_index(id)] 
    );
    
    _make_poised!( basis, lb_eff, ub_eff, get_result.(id, box_indices); ε_accept, solver, max_solver_evals, loop_start = 2 );
    nothing
end

function _make_lambda_poised!( basis :: Vector{LagrangePoly}, lb :: RVec, ub :: RVec;
    Λ :: Real, max_solver_evals :: Union{Int,Nothing}, solver :: Symbol ) :: Nothing
    
    N = length(lb);
    if isnothing( max_solver_evals)
        max_solver_evals = 300*(N+1);   # TODO is sensible?
    end

    iter_counter = 1;
    while !isinf(Λ)
        # determine a swap index iₖ, 1 ≤ iₖ ≤ length(basis):
        # maximize absolute value of each basis polynomial within trust region
        # if we find a maximizer xᵢ with |lᵢ| > Λ then we can replace the
        # site with index i with xᵢ 
        # iₖ is set so as to favor the argmax of all |lᵢ|
        # or to correspond to unevaluated sites and not the current iterate if possible
        iₖ = -1;
        xₖ = similar(lb);
        Λ_max = -Inf;
        swap_id = nothing;
        for (i,lp) = enumerate(basis)
            opt = NLopt.Opt( solver, N);
            opt.lower_bounds = lb;
            opt.upper_bounds = ub;
            opt.maxeval = max_solver_evals;
            opt.xtol_rel = 1e-3;
            opt.max_objective = (x,g) -> abs( eval_poly( lp, x) )
            x₀ = _rand_box_point(lb, ub);
            (abs_lᵢ, xᵢ, _) = NLopt.optimize(opt, x₀);
            
            if abs_lᵢ > Λ # found a potential swap candidate
                res_id = get_id(lp.res)
                if isa(swap_id, XInt) || abs_lᵢ > Λ_max || isnothing( res_id )
                    Λ_max = abs_lᵢ 
                    iₖ = i;
                    xₖ[:] = xᵢ;
                    swap_id = res_id
                    if isnothing(res_id)
                        # break if we have a suitable swapping point that is not yet 
                        # in the database (i.e. not evaluated yet)
                        break;
                    end
                end
            end
        end#for 
        if iₖ > 0
            # perform a point swap and normalize
            @logmsg loglevel4 " 2.$(iter_counter)) Replacing point at index $(iₖ)."
            basis[iₖ].res = init_res(Res, xₖ);
            _normalize!(basis[iₖ])            
            _orthogonalize!(basis, iₖ);
       else
            # finished, there is no poly with abs value > Λ 
            return nothing
        end#if
        iter_counter += 1;
    end#while
    nothing
end

function make_basis_lambda_poised!( basis :: Vector{LagrangePoly}, id :: AbstractIterData, mop :: AbstractMOP;
        Δ_factor :: Real = 1, Λ :: Real = 1.5, max_solver_evals :: Union{Int,Nothing} = nothing,
        solver :: Symbol = :GN_AGS ) :: Nothing
    @logmsg loglevel4 "Making the set $(Λ)-poised..."
    
    lb_eff, ub_eff = local_bounds(mop, xᵗ(id), Δᵗ(id) * Δ_factor);
    _make_lambda_poised!( basis, lb_eff, ub_eff; Λ, max_solver_evals, solver );
end


# helper function to easily evaluate polynomial
function eval_poly(p :: AbstractPolynomialLike, x :: RVec)
    return p( variables(p) => x ) 
end

function eval_poly( lp :: LagrangePoly, x :: RVec )
    eval_poly( lp.p, x )
end

function _orthogonalize!( basis :: Vector{LagrangePoly}, i :: Int ) :: Nothing
    p = length(basis)
    y = get_site( basis[i].res );
    for j=1:p
        if j≠i 
            basis[j].p -= (eval_poly( basis[j], y )* basis[i].p )
        end
    end
    nothing 
end

function _normalize!( lp :: LagrangePoly ) :: Nothing
    x = get_site(lp.res);
    @assert !isempty(x);
    lp.p /= eval_poly( lp, x);
    nothing 
end

# helper function to easily evaluate polynomial array for gradient
function eval_poly(p::Vector{<:AbstractPolynomialLike}, x :: RVec)
    [g( variables(p) => x ) for g ∈ p]
end

# helper function for initial monomial basis
@doc """
Return array of solution vectors [x_1, …, x_len] to the equation
``x_1 + … + x_len = rhs``
where the variables must be non-negative integers.
"""
function non_negative_solutions( rhs :: Int, len :: Int )
    if len == 1
        return rhs
    else
        solutions = [];
        for i = 0 : rhs
            for shorter_solution ∈ non_negative_solutions( i, len - 1)
                push!( solutions, [ rhs-i; shorter_solution ] )
            end
        end
        return solutions
    end
end


@doc "Factorial of a multinomial."
_multifactorial( arr :: Vector{Int} ) =  prod( factorial(α) for α in arr )

# @memoize IdDict
# TODO memoization caused a lot of trouble here in parallel usage.
# if important: use IdDicts and Memoization.jl
function _canonical_basis( n_vars :: Int, degree :: Int ) :: Vector{LagrangePoly}
    basis = LagrangePoly[];
    @polyvar χ[1:n_vars]
    for d = 0 : degree
        for multi_exponent ∈ non_negative_solutions( d, n_vars )
            poly = 1 / _multifactorial( multi_exponent ) * prod( χ.^multi_exponent );
            push!(basis, LagrangePoly(; p = poly ));
        end
    end
    return basis
end 