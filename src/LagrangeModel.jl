using DynamicPolynomials
import NLopt;

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
    mop :: AbstractMOP, id :: AbstractIterData ) :: Tuple{LagrangeModel, LagrangeMeta}
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
    return update_model(lm, objf, lmeta, mop, id; 
        ensure_fully_linear = !cfg.allow_not_linear)
end

function _point_in_box( x̂ :: RVec, lb :: RVec, ub :: RVec ) :: Bool 
    return all( lb .<= x̂ .<= ub )
end

"Indices of sites in database that lie in box with bounds `lb` and `ub`."
function find_points_in_box( id :: AbstractIterData, lb :: RVec, ub :: RVec;
    filter_x :: Bool = true ) :: Vector{Int}
    if !filter_x
        return [i for i = eachindex(db(id)) if _point_in_box(get_site(id, i), lb, ub) ]
    else
        xind = xᵗ_index(id);
        return [i for i = eachindex(db(id)) if i != xind && 
            _point_in_box(get_site(id, i), lb, ub) ]
    end
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
    mop :: AbstractMOP, id :: AbstractIterData; ensure_fully_linear :: Bool = false):: Tuple{LagrangeModel, LagrangeMeta}
    
    @info("Building Lagrange Models...");
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

    # todo remove this
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

    return lm, lmeta;
end

function improve_model( lm :: LagrangeModel, objf :: AbstractObjective, lmeta :: LagrangeMeta,
    mop :: AbstractMOP, id :: AbstractIterData; ensure_fully_linear :: Bool = false):: Tuple{LagrangeModel, LagrangeMeta}

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
    sum( lp.value[ℓ] * eval_poly(lp.grad_poly, x̂) for lp ∈ lm.basis )
end

function get_jacobian( lm :: LagrangeModel, x̂ :: RVec ) :: RMat
    grad_evals = [ eval_poly(lp.grad_poly, x̂) for lp ∈ lm.basis ];
    return Matrix(transpose( hcat(
        [ sum( get_value(lm.basis[i].res)[ ℓ ] * grad_evals[i] 
            for i = eachindex(grad_evals) )               
                for ℓ = 1 : lm.n_out ]... )));
end

_rand_box_point(lb::RVec, ub::RVec)::RVec = lb .+ (ub .- lb) .* rand(length(lb));

function make_basis_poised!( basis :: Vector{LagrangePoly}, id :: AbstractIterData, mop :: AbstractMOP;
    Δ_factor :: Real = 1, ε_accept :: Real = 1e-3, max_solver_evals :: Union{Int,Nothing} = nothing )
    @info("Finding a poised set...")
    
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
    box_indices = find_points_in_box(id, lb_eff, ub_eff; filter_x = true);
    
    p = length( basis );
    for i = 2 : p
        Y = get_sites(id, box_indices );
        lyᵢ, jᵢ = isempty(Y) ? (0,0) : findmax( abs.( eval_poly.(basis[i], Y) ) );
        if lyᵢ > ε_accept
            @info "\t 1.$(i)) Recycling a point from the database."
            yᵢ = Y[ jᵢ ];
            db_id = box_indices[jᵢ];
            deleteat!( box_indices, jᵢ );
        else
            @info "\t 1.$(i)) Computing a poised point by Optimization."
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
    @info("Making the set $(Λ)-poised...")
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
            @info "\t2.$(iter_counter)) Replacing point at index $(iₖ)."
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

#=
@doc """
    find_poised_set(ε_accept, start_basis, point_database, x, Δ, x_index, box_lb, box_ub)

    Using `ε_accept` as a small positive threshold, apply Algorithm 6.2 
    (see Conn, “Introduction to Derivative-Free Optimization”) to obtain
    a set poised for polynomial interpolation.
    
    Return arrays `new_sites` of sites that have to be newly sampled,
    `recycled_indices` to indicate which sites in `point_database` can 
    be used again and `lagrange_basis` as a vector of polyonmial basis functions
    on those sites combined.
"""
function find_poised_set(ε_accept :: R where R<:Real, start_basis :: Vector{Any}, 
        point_database :: RVecArr,
        x :: RVec, Δ :: Real, x_index :: Int, box_lb :: RVec,
        box_ub :: RVec; max_solver_evals = nothing :: Union{Nothing, Int})
    # Algorithm 6.2 (p. 95, Conn)
    # # select or generate a poised set suited for interpolation
    # # also computes the lagrange basis functions

    n_vars = length(x);
    if isnothing( max_solver_evals ) max_solver_evals = 300*(n_vars + 1); end
    rand_box_point() = box_lb .+ (box_ub .- box_lb) .* rand(n_vars);
    
    # find currently valid points
    # putting x_index first makes it be accepted as an interpolation site
    box_indices = find_points_in_box( x, Δ, point_database, Val(false))
    setdiff!(box_indices, x_index)
    insert!(box_indices, 1, x_index)
    
    lagrange_basis = copy( start_basis )
    p = length( lagrange_basis );
    p_init = length(box_indices);

    new_sites = RVec[];
    recycled_indices = Int64[];

    for i = 1 : p
        Y = point_database[ box_indices ];
        lyᵢ, jᵢ = isempty(Y) ? (0,0) : findmax( abs.( eval_poly.(lagrange_basis[i],Y ) ) )
        if lyᵢ > ε_accept
            @info "\t 1) It. $i: Recycling a point from the database."
            yᵢ = Y[jᵢ]
            push!(recycled_indices, box_indices[jᵢ])
            deleteat!(box_indices, jᵢ)
        else
            @info "\t 1) It. $i: Computing a poised point by Optimization."
            opt = Opt(:LN_BOBYQA, n_vars)
            opt.lower_bounds = box_lb
            opt.upper_bounds = box_ub
            opt.maxeval = max_solver_evals; 
            opt.xtol_rel = 1e-3;
            opt.max_objective = (x,g) -> abs( eval_poly( lagrange_basis[i], x ) )
            y₀ = rand_box_point()
            (_, yᵢ, ret) = optimize(opt, y₀)
            push!(new_sites, yᵢ)
        end

        # Normalization
        lagrange_basis[i] /= eval_poly( lagrange_basis[i], yᵢ)

        # Orthogonalization
        for j = 1 : p
            if j ≠ i
                lagrange_basis[j] -= (eval_poly( lagrange_basis[j], yᵢ) * lagrange_basis[i])
            end
        end
    end
    return new_sites, recycled_indices, lagrange_basis
end

# little helper for improvement of non-linear models below 
# orthogonalize "start_basis" using "points"
function lagrange_basis_from_points( start_basis :: Vector{Any}, points :: RVecArr )
    lagrange_basis = copy( start_basis )
    p = length( lagrange_basis );

    if p != length( points )
        error("Lengths of initial basis array and point array don't match!")
    end  

    for i = 1 : p
        yᵢ = points[i];

        # Normalization
        lagrange_basis[i] /= eval_poly( lagrange_basis[i], yᵢ)

        # Orthogonalization
        for j = 1 : p
            if j ≠ i
                lagrange_basis[j] -= (eval_poly( lagrange_basis[j], yᵢ) * lagrange_basis[i])
            end
        end
    end
    return lagrange_basis
end

@doc """
    improve_poised_set!( lagrange_basis, new_sites, recycled_indices, 
        Λ, point_database, box_lb, box_ub )

    Improve a poised point set (given by `point_database[recycled_indices]` and 
    `new_sites`) so that it becomes `Λ`-poised - using Algorithm 6.3
    (see Conn, “Introduction to Derivative-Free Optimization”).  
    Modifies in-place the first arguments, i.e. `lagrange_basis`, `new_sites`
    and `recycled_indices`.
"""
function improve_poised_set!( lagrange_basis :: Vector{Any}, 
        new_sites :: RVecArr, recycled_indices :: Vector{Int},
        Λ :: Real, point_database :: RVecArr, box_lb :: RVec,
        box_ub :: RVec; max_solver_evals = nothing :: Union{Nothing, Int})
    
    n_vars = isempty(point_database) ? length(new_sites[1]) : length( point_database[1] );
    p = length(lagrange_basis);
    if isnothing( max_solver_evals ) max_solver_evals = 300*(n_vars + 1); end
    rand_box_point() = box_lb .+ (box_ub .- box_lb) .* rand(n_vars);

    print_counter = 0;
    while !isinf(Λ)
        print_counter += 1;

        Y = [
            point_database[ recycled_indices ];
            new_sites
        ]
        num_recycled = length(recycled_indices);

        # 1) Λ calculation
        Λₖ₋₁ = -Inf;    # max_i max_x |l_i(x)|
        iₖ = -1;        # index of point to swap if set is not Λ-poised
        yₖ = zeros(n_vars);    # replacement site if not Λ-poised
        for i = 1 : p
            opt = Opt(:LN_BOBYQA, n_vars)
            opt.lower_bounds = box_lb;
            opt.upper_bounds = box_ub;
            opt.maxeval = max_solver_evals; 
            opt.xtol_rel = 1e-3;
            opt.max_objective = (x,g) -> abs( eval_poly( lagrange_basis[i], x ) )
            y₀ = rand_box_point()
            (abs_lᵢ, yᵢ, _) = optimize(opt, y₀)

            update_Λₖ₋₁ = abs_lᵢ > Λₖ₋₁;
            if abs_lᵢ > Λ   
                # the algo works with any point satisfying `abs_lᵢ > Λ`
                # we increase `iₖ` if it was pointing to a recycled site 
                # or to favor the argmax
                if iₖ <= num_recycled || update_Λₖ₋₁
                    iₖ = i;
                    yₖ[:] = yᵢ[:];
                end
                
                # update the max
                # note that it is ok to do so only if abs_lᵢ > Λ because we test Λₖ₋₁ > Λ and …
                # … if no abs_li is bigger than Λ then the max cannot be bigger neither
                if update_Λₖ₋₁
                    Λₖ₋₁ = abs_lᵢ
                end
            end
        end

        # 2) Point swap
        if Λₖ₋₁ > Λ
            @info("\t It. $print_counter: Λₖ₋₁ is $Λₖ₋₁. Performing a point swap for index $iₖ…")
            if iₖ > num_recycled
                # delete the site referenced by iₖ from new_sites
                deleteat!( new_sites, iₖ - num_recycled)
            else
                # we have to sacrifice an old site :(
                deleteat!( recycled_indices, iₖ )
            end
            push!( new_sites, yₖ )
            # sort basis so that polynomial for iₖ is last
            lagrange_basis[ [ iₖ; iₖ+1:p ] ] = lagrange_basis[ [iₖ+1:p; iₖ ] ]
        else
            @info("\t Λₖ₋₁ is $(Λₖ₋₁) < Λ = $(Λ)!")
            return nothing            
        end

        # 3) Lagrange Basis update
        ## Normalization
        lagrange_basis[p] /= eval_poly( lagrange_basis[p], yₖ )

        # Orthogonalization
        for j = 1 : p-1
            lagrange_basis[ j ] -= ( eval_poly(lagrange_basis[j], yₖ) * lagrange_basis[ p ] )
        end
    end
    @info("\tFor a $Λ-poised set we can recycle $(length(recycled_indices)) sites.")
end

function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
    cfg :: LagrangeConfig, crit_flag :: Bool = true )
    if cfg.optimized_sampling
        build_model_optimized(ac, objf, cfg, crit_flag)
    else
        build_model_stencil(ac, objf, cfg);
    end
end 

function build_model_stencil(ac :: AlgoConfig, objf :: VectorObjectiveFunction,
    cfg :: LagrangeConfig, crit_flag :: Bool = true )
    @unpack ε_accept, θ_enlarge, Λ, allow_not_linear = cfg;
    @unpack n_vars, iter_data, problem, use_eval_database = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db = iter_data;
    
    if numevals(objf) < max_evals(cfg)
        # Find a nice point set in the unit hypercube
        stencil_sites = [];
        if isempty(cfg.stencil_sites)
            # maybe there is a precalculated set in data folder?
            if cfg.degree >= 2 && cfg.use_saved_sites
                lock( cfg.io_lock )
                try 
                    fn = joinpath(@__DIR__, "data", "lagrange_basis_$(n_vars)_vars.jld2" );
                    if isfile(fn)                    
                        precalculated_data = load(fn);
                        if precalculated_data["Λ"] <= Λ
                            @info "\tUsing saved sites and Lagrange Basis."
                            lagrange_basis = precalculated_data["lagrange_basis"];
                            stencil_sites = precalculated_data["sites"];
                        end
                    end# isfile
                catch
                finally
                    unlock( cfg.io_lock )
                end#try                     
            end#deg
        
            if isempty(stencil_sites)
                X = .5 .* ones(n_vars); # center point
                # find poiset set in hypercube
                new_sites, recycled_indices, lagrange_basis = find_poised_set( ε_accept, 
                    cfg.canonical_basis, [X,], X, 0.5, 1, zeros(n_vars), ones(n_vars);
                    max_solver_evals = 200*(n_vars+1)^cfg.degree );
                # make Λ poised
                improve_poised_set!(lagrange_basis, new_sites, recycled_indices, Λ, 
                    [X,], zeros(n_vars), ones(n_vars);
                    max_solver_evals = 200*(n_vars+1)^cfg.degree );
                stencil_sites = [ [X,][recycled_indices]; new_sites ];
            end

            # save pre-calculated Lagrange basis in config
            cfg.canonical_basis = lagrange_basis;
            cfg.stencil_sites = stencil_sites;
        end

        # scale sites to current trust region
        θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
        Δ_1 = θ * Δ;
        lb_eff, ub_eff = effective_bounds_vectors( x, Δ_1, Val(problem.is_constrained) )   
        
        # scale stencil sites to current trust region box
        new_sites = ( ξ -> lb_eff .+ (ub_eff .- lb_eff) .* ξ ).( cfg.stencil_sites )

        # modify lagrange basis such that sites are unscaled to unit hypercube
        χ = variables( cfg.canonical_basis[1] );
        scaling_poly = (χ .- lb_eff) ./ (ub_eff .- lb_eff)
        lagrange_basis = [ subs(lp, χ => scaling_poly) for lp in cfg.canonical_basis  ]
        
        # evaluate and build model
        @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")

        new_indices = eval_new_sites( ac, new_sites );

        lmodel, lmeta = get_final_model( lagrange_basis, new_indices, 
            objf, iter_data, cfg.degree, true );

        if !use_eval_database
            reset_database!(iter_data);
        end
        return lmodel, lmeta
    end
end

function build_model_optimized( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: LagrangeConfig, crit_flag :: Bool = true )
    @unpack ε_accept, θ_enlarge, Λ, allow_not_linear = cfg;
    @unpack n_vars, iter_data, problem, use_eval_database = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db = iter_data;

    # use a slightly enlarged box …
    θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
    Δ_1 = θ * Δ;
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ_1, Val(problem.is_constrained) )   

    # find a poised set
    if numevals(objf) < max_evals(cfg) 
        new_sites, recycled_indices, lagrange_basis = find_poised_set( ε_accept, 
            cfg.canonical_basis, sites_db, x, Δ, x_index, lb_eff, ub_eff;
            max_solver_evals = 200*(n_vars+1)^cfg.degree );
    end

    fully_linear = false;
    # make the set Λ poised for full linearity
    if !allow_not_linear || crit_flag && numevals(objf) < max_evals(cfg)
        improve_poised_set!(lagrange_basis,new_sites, recycled_indices, Λ, sites_db, lb_eff, ub_eff;
            max_solver_evals = 200*(n_vars+1)^cfg.degree );
        fully_linear = true
    end
    
    # evaluate
    @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")
    new_indices = eval_new_sites( ac, new_sites );

    lmodel, lmeta = get_final_model( lagrange_basis, [recycled_indices; new_indices], 
        objf, iter_data, cfg.degree, fully_linear );

    if !use_eval_database && fully_linear
        reset_database!(iter_data);
    end

    return lmodel, lmeta
end

@doc "Return vector of evaluations for output `ℓ` of a (vector) Lagrange Model
`lm` at scaled site `ξ`."
function eval_models( lm :: LagrangeModel, ξ :: RVec, ℓ :: Int64 )
   eval_poly( lm.lagrange_models[ℓ], ξ ) 
end

function get_gradient( lm :: LagrangeModel, ξ :: RVec, ℓ :: Int64 )
    grad_poly = differentiate.( lm.lagrange_models[ℓ], variables(lm.lagrange_models[ℓ] ) )
    grad = eval_poly(grad_poly, ξ)
    try 
        @assert length(grad) != 0;
    catch
        @show ℓ, ξ
        @show lm.lagrange_models[ℓ]
        @show variables( lm.lagrange_models[ℓ] )
        @show grad_poly
    end
    return grad;
end

function eval_models( lm :: LagrangeModel, ξ :: RVec)
    vcat( [ eval_models(lm, ξ, ℓ) for ℓ = 1 : num_outputs(lm) ]... )
end

function get_jacobian( lm :: LagrangeModel, ξ :: RVec)
    transpose( hcat( [ get_gradient(lm, ξ, ℓ) for ℓ = 1 : num_outputs(lm) ]... ) )
end

function make_linear!( lm::LagrangeModel, lmeta::LagrangeMeta, ac::AlgoConfig, 
    objf::VectorObjectiveFunction, cfg::LagrangeConfig )
    return improve!( lm, lmeta, ac, objf, cfg);
end

function improve!( lm::LagrangeModel, lmeta::LagrangeMeta, ac::AlgoConfig, 
    objf::VectorObjectiveFunction, cfg::LagrangeConfig )
    @unpack θ_enlarge, Λ, allow_not_linear = cfg;
    @unpack iter_data, problem, use_eval_database = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db = iter_data;

    # use a slightly enlarged box …
    θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
    Δ_1 = θ * Δ;
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ_1, Val(problem.is_constrained) )   

    # reuse the poised set from before
    new_sites = RVec[];
    recycled_indices = lmeta.interpolation_indices;
    #lagrange_basis = lmeta.lagrange_basis;
    lagrange_basis = lagrange_basis_from_points( cfg.canonical_basis, sites_db[ recycled_indices ] );

    # make the set Λ poised for full linearity
    improve_poised_set!(lagrange_basis, new_sites, recycled_indices, Λ, sites_db, lb_eff, ub_eff)
    
    # evaluate
    @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")
    new_indices = eval_new_sites( ac, new_sites );

    lmodel, lmeta2 = get_final_model( lagrange_basis, [recycled_indices; new_indices], 
        objf, iter_data, cfg.degree, true );
   
    as_second!(lm, lmodel);
    as_second!(lmeta, lmeta2);

    if !use_eval_database
        reset_database!(iter_data);
    end

    return isempty(new_sites);
end

# build interpolating model to return
function get_final_model( lagrange_basis :: Vector{T} where T, 
        interpolation_indices :: Union{Vector{Int}, AbstractRange}, 
        objf :: VectorObjectiveFunction, iter_data :: IterData, 
        degree :: Int, fully_linear :: Bool)
    
    lmeta = LagrangeMeta( 
        #lagrange_basis = lagrange_basis,
        interpolation_indices = interpolation_indices 
    );

    t_vals = get_training_values( objf, lmeta, iter_data);

    lagrange_models = sum( lagrange_basis[i] .* t_vals[i] for i = eachindex(lagrange_basis) );

    lmodel = LagrangeModel(
        n_out = num_outputs( objf ),
        degree = degree,
        lagrange_models = lagrange_models,
        fully_linear = fully_linear
    );
    return lmodel, lmeta 
end

=#