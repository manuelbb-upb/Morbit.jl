# This file defines the required methods for vector-valued Lagrange models.
#
# It is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.
using DynamicPolynomials
using FileIO, JLD2;

@with_kw mutable struct LagrangeModel <: SurrogateModel
    n_out :: Int64 = -1;
    degree :: Int64 = 1;

    lagrange_models :: Vector{Any} = [];
    fully_linear :: Bool = false;
end
Broadcast.broadcastable( lm :: LagrangeModel ) = Ref(lm);

@with_kw mutable struct LagrangeConfig <: ModelConfig
    degree :: Int64 = 1;
    θ_enlarge :: Float64 = 2;

    ε_accept :: Float64 = 1e-6;
    Λ :: Float64 = 1.5;

    allow_not_linear :: Bool = false;

    optimized_sampling :: Bool = true;

    # the basis is set by `prepare!`
    canonical_basis :: Union{ Nothing, Vector{Any} } = nothing
    # fields to enable unoptimized (stencil) sampling
    stencil_sites :: Vector{Vector{Float64}} = [];

    # if optimized_sampling = false, shall we use saved sites?
    use_saved_sites :: Bool = true;
    io_lock :: Union{Nothing, ReentrantLock, Threads.SpinLock} = nothing;

    max_evals :: Int64 = typemax(Int64);
end

function Base.deepcopy( cfg :: LagrangeConfig)
    new_cfg = LagrangeConfig();
    for fname in fieldnames(LagrangeConfig)
        if fname != :io_lock
            setfield!(new_cfg, fname, deepcopy(getfield(cfg, fname)) );
        else
            setfield!(new_cfg, fname, getfield(cfg, fname) );
        end
    end
    return new_cfg
end

#Base.lock( f, ::Nothing ) = try f() catch end;
function Base.lock(::Nothing) end
function Base.unlock(::Nothing) end


@with_kw mutable struct LagrangeMeta <: SurrogateMeta
    interpolation_indices :: Vector{Int64} = [];
    #lagrange_basis :: Vector{Any} = [];
end

# helper function to easily evaluate polynomial
function eval_poly(p ::T where T<:Union{Monomial, Polynomial, Term, PolyVar},
        x :: Vector{R} where R<:Real)
    p( variables(p) => x ) 
end
# helper function to easily evaluate polynomial array for gradient
function eval_poly(p::Vector{T} where T<:Union{Monomial, Polynomial, Term, PolyVar},
    x :: Vector{R} where R<:Real)
    [g( variables(p) => x ) for g in p]
end

# Overwrite Type methods
fully_linear( lm :: LagrangeModel ) = lm.fully_linear;
max_evals( cfg :: LagrangeConfig ) = cfg.max_evals;

# helpers for the modifying functions (improve!, make_linear!)
@doc "Modify first meta data object to equal second."
function as_second!(destination :: T, source :: T ) where{T<:LagrangeMeta}
    @unpack_LagrangeMeta source;
    @pack_LagrangeMeta! destination;
    return nothing
end
function as_second!(destination :: T, source :: T ) where{T<:LagrangeModel}
    @unpack_LagrangeModel source;
    @pack_LagrangeModel! destination;
    return nothing
end

@doc "Factorial of a multinomial."
multifactorial( arr :: Vector{Int} ) =  prod( factorial(α) for α in arr )

@doc "Set field `:canonical_basis` of `cfg::LagrangeConfig` object."
function prepare!(objf :: VectorObjectiveFunction, cfg :: LagrangeConfig, ac::AlgoConfig )
    n = ac.n_vars;
    D = cfg.degree;
    p = binomial( n + D, n)
    # Generate the canonical basis for space of polynomial of degree at most D
    cfg.canonical_basis = Any[];
    @polyvar χ[1:n];
    for d = 0 : D
        for multi_exponent in non_negative_solutions( d, n )
            poly = 1/multifactorial(multi_exponent) * prod( χ.^multi_exponent );
            push!(cfg.canonical_basis, poly)
        end
    end
    return nothing
end

@doc "Return list of training values for current Lagrange Model described by `cfg` and `meta`."
function get_training_values( objf :: VectorObjectiveFunction, meta :: LagrangeMeta, id :: IterData )
    [ deepcopy(val[objf.internal_indices]) for val ∈ id.values_db[ meta.interpolation_indices ] ]
end

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
        point_database :: Vector{Vector{R}} where R<:Real,
        x :: Vector{R} where R<:Real, Δ :: Float64, x_index :: Int, box_lb :: Vector{R} where R<:Real,
        box_ub :: Vector{R} where R<:Real; max_solver_evals = nothing :: Union{Nothing, Int})
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

    new_sites = Vector{Vector{Float64}}();
    recycled_indices = Int64[];

    for i = 1 : p
        Y = point_database[ box_indices ];
        lyᵢ, jᵢ = isempty(Y) ? (0,0) : findmax( abs.( eval_poly.(lagrange_basis[i],Y ) ) )
        if lyᵢ > ε_accept
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
function lagrange_basis_from_points( start_basis :: Vector{Any}, points :: Vector{Vector{R}} where R<:Real )
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
        new_sites :: Vector{Vector{R}} where R<:Real, recycled_indices :: Vector{Int},
        Λ :: Float64,
        point_database :: Vector{Vector{R}} where R<:Real, box_lb :: Vector{R} where R<:Real,
        box_ub :: Vector{R} where R<:Real; max_solver_evals = nothing :: Union{Nothing, Int})
       
    
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
        yₖ = zeros(Float64, n_vars);    # replacement site if not Λ-poised
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
                # … if no abs_li is bigger than Λ then the max cannot be bigger as well
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
            @info("\t Λₖ₋₁ is $Λₖ₋₁ < Λ = $(Λ)!")
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
function eval_models( lm :: LagrangeModel, ξ :: Vector{Float64}, ℓ :: Int64 )
   eval_poly( lm.lagrange_models[ℓ], ξ ) 
end

function get_gradient( lm :: LagrangeModel, ξ :: Vector{Float64}, ℓ :: Int64 )
    grad_poly = differentiate.( lm.lagrange_models[ℓ], variables(lm.lagrange_models[ℓ] ) )
    return eval_poly(grad_poly, ξ)
end

function eval_models( lm :: LagrangeModel, ξ :: Vector{Float64})
    vcat( [ eval_models(lm, ξ, ℓ) for ℓ = 1 : lm.n_out ]... )
end

function get_jacobian( lm :: LagrangeModel, ξ :: Vector{Float64})
    transpose( hcat( [ get_gradient(lm, ξ, ℓ) for ℓ = 1 : lm.n_out ]... ) )
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
    new_sites = Vector{Vector{Float64}}();
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

    lagrange_models = sum( lagrange_basis[i] .* get_training_values( objf, lmeta, iter_data)[i] 
        for i = eachindex(lagrange_basis) );

    lmodel = LagrangeModel(
        n_out = objf.n_out,
        degree = degree,
        lagrange_models = lagrange_models,
        fully_linear = fully_linear
    );
    return lmodel, lmeta 
end