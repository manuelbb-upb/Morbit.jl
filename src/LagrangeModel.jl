using DynamicPolynomials

#using FileIO, JLD2;

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
    ε_accept :: Real = 1e-6;
    "Quality Parameter in Λ-Poisedness Algorithm."
    Λ :: Real = 1.5;

    allow_not_linear :: Bool = false;

    # optimized_sampling :: Bool = true;

    # # if optimized_sampling = false, shall we try to use saved sites?
    # use_saved_sites :: Bool = true;
    # io_lock :: Union{Nothing, ReentrantLock, Threads.SpinLock} = nothing;

    algo1_max_evals :: Union{Nothing,Int} = nothing;    # nothing = automatic
    algo2_max_evals :: Union{Nothing,Int} = nothing;

    max_evals :: Int64 = typemax(Int64);
end

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
        out_indices = output_indices( objf, mop),
    );
    # prepare initial basis of polynomial space 
    @polyvar χ[1:lm.n_vars]
    for d = 0 : cfg.degree
        for multi_exponent ∈ non_negative_solutions( d, lm.n_vars )
            poly = 1 / _multifactorial( multi_exponent ) * prod( χ.^multi_exponent );
            push!(lm.basis, LagrangePoly(; p = poly ));
        end
    end     
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

function update_model( lm :: LagrangeModel, objf :: AbstractObjective, lmeta :: LagrangeMeta,
    mop :: AbstractMOP, id :: AbstractIterData, :: AbstractConfig; 
    ensure_fully_linear :: Bool = false):: Tuple{LagrangeModel, LagrangeMeta}
    
    @logmsg loglevel3 "Building LagrangeModel with indices $(output_indices(objf, mop))."
    cfg = model_cfg(objf) :: LagrangeConfig;
    
    make_basis_poised!( 
        lm.basis, id, mop ; 
        Δ_factor = cfg.θ_enlarge, ε_accept = cfg.ε_accept,
        max_solver_evals = cfg.algo1_max_evals
    );
    
    if ensure_fully_linear || !cfg.allow_not_linear
        make_basis_lambda_poised!(
            lm.basis, id, mop;
            Δ_factor = cfg.θ_enlarge, Λ = cfg.Λ, max_solver_evals = cfg.algo2_max_evals
        );
        lm.fully_linear = true;
    end

    _eval_new_sites!( lm, lmeta, mop, id);
    _set_gradient!.(lm.basis);

    # TODO remove this
    #=
    oi = output_indices(objf, mop);
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
    make_basis_lambda_poised!( lm.basis, objf, mop; 
        Δ_factor = cfg.θ_enlarge, Λ = cfg.Λ, max_solver_evals = cfg.algo2_max_evals );
    lm.fully_linear = true;
    _eval_new_sites!( lm, lmeta, mop, id);
    _set_gradient!.(lm.basis)
    return lm, lmeta
end


function eval_models( lm :: LagrangeModel, x̂ :: RVec, ℓ :: Int ) :: Real
    sum( get_value(lp.res)[ ℓ ] * eval_poly(lp.p, x̂) for lp ∈ lm.basis)    
end

function eval_models( lm :: LagrangeModel, x̂ :: RVec ) :: RVec
    sum( get_value(lp.res)[ lm.out_indices ] * eval_poly(lp.p, x̂) for lp ∈ lm.basis)    
end

function get_gradient( lm :: LagrangeModel, x̂ :: RVec, ℓ :: Int ) :: RVec
    sum( get_value(lp.res)[ℓ] * eval_poly(lp.grad_poly, x̂) for lp ∈ lm.basis )
end

function get_jacobian( lm :: LagrangeModel, x̂ :: RVec ) :: RMat
    grad_evals = [ eval_poly(lp.grad_poly, x̂) for lp ∈ lm.basis ];
    return Matrix(transpose( hcat(
        [ sum( get_value(lm.basis[i].res)[ ℓ ] * grad_evals[i] 
            for i = eachindex(grad_evals) )               
                for ℓ = 1 : lm.n_out ]... )));
end

function make_basis_poised!( basis :: Vector{LagrangePoly}, id :: AbstractIterData, mop :: AbstractMOP;
    Δ_factor :: Real = 1, ε_accept :: Real = 1e-3, max_solver_evals :: Union{Int,Nothing} = nothing )
    @logmsg loglevel4 "Finding a poised set..."
    
    x = xᵗ( id );
    N = length(x);
    if isnothing( max_solver_evals)
        max_solver_evals = 300*(N+1);   # TODO is sensible?
    end

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
    
    p = length( basis );
    for i = 2 : p
        Y = get_sites(id, box_indices );
        lyᵢ, jᵢ = isempty(Y) ? (0,0) : findmax( abs.( eval_poly.(basis[i], Y) ) );
        if lyᵢ > ε_accept
            @logmsg loglevel4 " 1.$(i)) Recycling a point from the database."
            yᵢ = Y[ jᵢ ];
            db_id = box_indices[jᵢ];
            deleteat!( box_indices, jᵢ );
        else
            @logmsg loglevel4 " 1.$(i)) Computing a poised point by Optimization."
            opt = NLopt.Opt(NLopt.:LN_BOBYQA, N)
            opt.lower_bounds = lb_eff;
            opt.upper_bounds = ub_eff;
            opt.maxeval = max_solver_evals;
            opt.xtol_rel = 1e-3;
            opt.max_objective = (x,g) -> abs( eval_poly( basis[i], x) )
            y₀ = _rand_box_point(lb_eff, ub_eff);
            (_, yᵢ, ret) = NLopt.optimize(opt, y₀)
            #push!(new_sites, yᵢ)
            db_id = nothing;
        end

        # Set site & id, then normalize and orthogonalize basis
        change_site!(basis[i].res, yᵢ);
        change_id!(basis[i].res, db_id);
        _normalize!(basis[i]);
        _orthogonalize!( basis, i)    
    end
     
    nothing
end

function make_basis_lambda_poised!( basis :: Vector{LagrangePoly}, id :: AbstractIterData, mop :: AbstractMOP;
    Δ_factor :: Real = 1, Λ :: Real = 1.5, max_solver_evals :: Union{Int,Nothing} = nothing )
    @logmsg loglevel4 "Making the set $(Λ)-poised..."
    x = xᵗ( id );


    N = length(x);
    if isnothing( max_solver_evals)
        max_solver_evals = 300*(N+1);   # TODO is sensible?
    end

    p = length( basis )
    lb_eff, ub_eff = local_bounds(mop, x, Δᵗ(id) * Δ_factor);
 
    iter_counter = 1;
    while !isinf(Λ)
        iₖ = -1;
        yₖ = similar(x);
        Λ_max = -Inf;
        swap_id = nothing;
        for (i,lp) = enumerate(basis)
            opt = NLopt.Opt(NLopt.:LN_BOBYQA, N);
            opt.lower_bounds = lb_eff;
            opt.upper_bounds = ub_eff;
            opt.maxeval = max_solver_evals;
            opt.xtol_rel = 1e-3;
            opt.max_objective = (x,g) -> abs( eval_poly( lp, x) )
            y₀ = _rand_box_point(lb_eff, ub_eff);
            (abs_lᵢ, yᵢ, _) = NLopt.optimize(opt, y₀);
            if abs_lᵢ > Λ
                res_id = get_id(lp.res)
                if isa(swap_id, XInt) || abs_lᵢ > Λ_max || isnothing( res_id )
                    Λ_max = abs_lᵢ 
                    iₖ = i;
                    yₖ[:] = yᵢ;
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
            # there was a polynomial with abs value > Λ 
            # perform a point swap and normalize
            @logmsg loglevel4 " 2.$(iter_counter)) Replacing point at index $(iₖ)."
            change_site!(basis[iₖ].res , yₖ);
            change_id!(basis[iₖ].res, nothing);
            _normalize!(basis[iₖ])            
            _orthogonalize!(basis, iₖ);
       else
            # finished, there is no poly with abs value > Λ 
            return nothing
        end#if
        iter_counter += 1;
    end#while
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

function _normalize!( lp :: LagrangePoly )
    x = get_site(lp.res);
    @assert !isempty(x);
    lp.p /= eval_poly( lp, x);
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