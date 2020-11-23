# This file defines the required methods for vector-valued Lagrange models.
#
# It is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.
using DynamicPolynomials

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
    [ val[objf.internal_indices] for val ∈ id.values_db[ meta.interpolation_indices ] ]
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
        box_ub :: Vector{R} where R<:Real )
    # Algorithm 6.2 (p. 95, Conn)
    # # select or generate a poised set suited for interpolation
    # # also computes the lagrange basis functions

    n_vars = length(x);
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
            opt.maxeval = 300;
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
        box_ub :: Vector{R} where R<:Real )
    
    n_vars = isempty(point_database) ? length(new_sites[1]) : length( point_database[1] );
    p = length(lagrange_basis);
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
            opt.maxeval = 300;
            opt.max_objective = (x,g) -> abs( eval_poly( lagrange_basis[i], x ) )
            y₀ = rand_box_point()
            (abs_lᵢ, yᵢ, _) = optimize(opt, y₀)

            update_Λₖ₋₁ = abs_lᵢ > Λₖ₋₁;

            if abs_lᵢ > Λ   
                # the algo works with any point satisfying `abs_lᵢ > Λ`
                # we increase `iₖ` if it was pointing to a recycled site 
                # or to choose favor the argmax
                if iₖ <= num_recycled || update_Λₖ₋₁
                    iₖ = i;
                    yₖ[:] = yᵢ[:];
                end
            end
            if update_Λₖ₋₁
                Λₖ₋₁ = abs_lᵢ
            end
        end

        # 2) Point swap
        if Λₖ₋₁ ≥ Λ
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
    @unpack n_vars, iter_data, problem = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db = iter_data;

    # Find a nice point set in the unit hypercube
    if isempty(cfg.stencil_sites)
        X = .5 .* ones(n_vars); # center point
        # find poiset set in hypercube
        new_sites, recycled_indices, lagrange_basis = find_poised_set( ε_accept, 
            cfg.canonical_basis, [X,], X, Δ, 1, zeros(n_vars), ones(n_vars) );
        # make Λ poised
        improve_poised_set!(lagrange_basis, new_sites, recycled_indices, min( 1.2, Λ ), 
            [X,], zeros(n_vars), ones(n_vars) );

        # save pre-calculated Lagrange basis in config
        cfg.canonical_basis = lagrange_basis;
        cfg.stencil_sites = [ [X,][recycled_indices]; new_sites ];
    end

    # scale sites to current trust region
    θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
    Δ_1 = θ * Δ;
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ_1, Val(problem.is_constrained) )   
    
    # scale stencil sites to current trust region box
    new_sites = ( ξ -> lb_eff .+ (ub_eff .- lb_eff) .* ξ ).( cfg.stencil_sites )

    # modfy lagrange basis such that sites are unscaled to unit hypercube
    χ = variables( cfg.canonical_basis[1] );
    scaling_poly = (χ .- lb_eff) ./ (ub_eff .- lb_eff)
    lagrange_basis = [ subs(lp, χ => scaling_poly) for lp in cfg.canonical_basis  ]
    
    # evaluate and build model
    @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")

    new_indices = eval_new_sites( ac, new_sites );

    lmodel, lmeta = get_final_model( lagrange_basis, new_indices, 
        objf, iter_data, cfg.degree, true );
    return lmodel, lmeta
end

function build_model_optimized( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: LagrangeConfig, crit_flag :: Bool = true )
    @unpack ε_accept, θ_enlarge, Λ, allow_not_linear = cfg;
    @unpack n_vars, iter_data, problem = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db = iter_data;

    # use a slightly enlarged box …
    θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
    Δ_1 = θ * Δ;
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ_1, Val(problem.is_constrained) )   

    # find a poised set
    new_sites, recycled_indices, lagrange_basis = find_poised_set( ε_accept, 
        cfg.canonical_basis, sites_db, x, Δ, x_index, lb_eff, ub_eff );

    fully_linear = false;
    # make the set Λ poised for full linearity
    if !allow_not_linear || crit_flag
        improve_poised_set!(lagrange_basis,new_sites, recycled_indices, Λ, sites_db, lb_eff, ub_eff)
        fully_linear = true
    end
    
    # evaluate
    @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")
    new_indices = eval_new_sites( ac, new_sites );

    lmodel, lmeta = get_final_model( lagrange_basis, [recycled_indices; new_indices], 
        objf, iter_data, cfg.degree, fully_linear );

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
    @unpack iter_data, problem = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db = iter_data;

    # use a slightly enlarged box …
    θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
    Δ_1 = θ * Δ;
    lb_eff, ub_eff = effective_bounds_vectors( x, Δ_1, Val(problem.is_constrained) )   

    # reuse the poised set from before
    new_sites = Vector{Vector{Float64}}();
    recycled_indices = lmeta.interpolation_indices;
    lagrange_basis = lmeta.lagrange_basis;

    # make the set Λ poised for full linearity
    improve_poised_set!(lagrange_basis, new_sites, recycled_indices, Λ, sites_db, lb_eff, ub_eff)
    
    # evaluate
    @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")
    new_indices = eval_new_sites( ac, new_sites );

    lmodel, lmeta2 = get_final_model( lagrange_basis, [recycled_indices; new_indices], 
        objf, iter_data, cfg.degree, true );
   
    as_second!(lm, lmodel);
    as_second!(lmeta, lmeta2);
    return isempty(new_sites);
end

# build interpolating model to return
function get_final_model( lagrange_basis :: Vector{T} where T, 
        interpolation_indices :: Union{Vector{Int}, AbstractRange}, 
        objf :: VectorObjectiveFunction, iter_data :: IterData, 
        degree :: Int, fully_linear :: Bool)
    lmeta = LagrangeMeta( 
        lagrange_basis = lagrange_basis,
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