# This file defines the required methods for vector-valued Taylor models.
#
# It is included from within "Surrogates.jl".
# We therefore can refer to other data structures used there.

fully_linear( lm :: LagrangeModel ) = lm.fully_linear;

# make the canonical polynomial basis available in `LagrangeConfig`
function prepare!(objf :: VectorObjectiveFunction, cfg :: LagrangeConfig, ac::AlgoConfig )
    n = ac.n_vars;
    D = cfg.degree;
    p = binomial( n + D, n)
    # Generate the canonical basis for space of polynomial of degree at most D
    cfg.canonical_basis = Vector{Polynomial{n}}(undef, p);
    index = 1
    for d = 0 : D
        for sol in non_negative_solutions( d, n )
            m = Polynomials.MultiIndex( sol )
            poly = Polynomial( 1 / factorial( m ), m )
            cfg.canonical_basis[index] = poly
            index += 1
        end
    end
    return nothing
end

function non_lagrange_points( lm :: LagrangeMeta, id :: IterData )
    setdiff( 1 : length(id.sites_db), keys(lm.basis_data) )
end

@doc "Return list of training values for current Lagrange Model described by `cfg` and `meta`."
function get_training_values( objf :: VectorObjectiveFunction, meta :: LagrangeMeta, id :: IterData )
    [ val[objf.internal_indices] for val ∈ id.values_db[ meta.interpolation_indices ] ]
end

function build_model( ac :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: LagrangeConfig, crit_flag :: Bool = true )
    @unpack ε_accept, θ_enlarge, Λ = cfg;
    @unpack n_vars, iter_data, problem = ac;
    @unpack Δ, x, x_index, f_x, sites_db, values_db, model_meta = iter_data;

    # NOTE We could save a bit on the recomputation of the Lagrange Basis
    # by recycling the old basis. However keeping track of the indices becomes
    # somewhat difficult.

    θ = sensible_θ(Val(problem.is_constrained), θ_enlarge, x, Δ )
    Δ_1 = θ * Δ;

    lb_eff, ub_eff = effective_bounds_vectors( x, Δ, Val(problem.is_constrained) )
    rand_box_point() = lb_eff .+ (ub_eff .- lb_eff) .* rand(n_vars);

    # find currently valid points
    # putting x_index first makes it be accepted as an interpolation site
    box_indices = find_points_in_box( x, Δ, sites_db, Val(false))
    setdiff!(box_indices, x_index)
    insert!(box_indices, 1, x_index)


    # Algorithm 6.2 (p. 95, Conn)
    # # select or generate a poised set suited for interpolation
    # # also computes the lagrange basis functions
    lagrange_basis = copy( cfg.canonical_basis )
    p = length( lagrange_basis );
    p_init = length(box_indices);

    accepted_indices = zeros(Bool, p_init);
    new_sites = Vector{Vector{Float64}}();

    for i = 1 : p
        Y = [
            sites_db[ box_indices[ .!(accepted_indices) ] ];
            new_sites
        ]
        lyᵢ, jᵢ = isempty(Y) ? (0,0) : findmax( abs.( lagrange_basis[i].(Y)) )
        if lyᵢ > ε_accept
            yᵢ = Y[jᵢ]
            accepted_indices[jᵢ] = true;    # exclude from further investigation
        else
            @info "\t It. 1:$i: Computing a poised point by Optimization."
            opt = Opt(:LN_BOBYQA, 2)
            opt.lower_bounds = lb_eff
            opt.upper_bounds = ub_eff
            opt.min_objective = (x,g) -> lagrange_basis[i](x)
            y₀ = rand_box_point()
            (_, yᵢ, _) = optimize(opt, y₀)

            push!(new_sites, yᵢ)
        end

        # Normalization
        lagrange_basis[i] /= lagrange_basis[i](yᵢ)

        # Orthogonalization
        for j = 1:p
            if j≠i
                lagrange_basis[j] -= (lagrange_basis[j](yᵢ) * lagrange_basis[i])
            end
        end
    end

    # We now have a point set suited for unique interpolation…
    # But "poisedness" does not suffice for full linearity
    fully_linear = false;
    recycled_indices = box_indices[ accepted_indices ];

    # We hence need
    # Algorithm 6.3 (p.96 Conn)
    # to guarantee Λ-poisedness
    while true
        Y = [
            sites_db[ recycled_indices ];
            new_sites
        ]
        num_recycled = length(recycled_indices);
        @show recycled_indices
        @show length(new_sites)
        @show length(Y)

        # 1) Λ calculation
        Λₖ₋₁ = -Inf;    # max_i max_x |l_i(x)|
        iₖ = -1;        # index of point to swap if set is not Λ-poised
        yₖ = zeros(Float64, n_vars);    # replacement site if not Λ-poised
        for i = 1 : p
            opt = Opt(:LN_BOBYQA, 2)
            opt.lower_bounds = lb_eff
            opt.upper_bounds = ub_eff
            opt.max_objective = (x,g) -> abs(lagrange_basis[i](x))
            y₀ = rand_box_point()
            (abs_lᵢ, yᵢ, _) = optimize(opt, y₀)

            update_Λₖ₋₁ = abs_lᵢ > Λₖ₋₁;
            if abs_lᵢ > Λ
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
            @info("\t Λₖ₋₁ is $Λₖ₋₁. Performing a point swap for index $iₖ.")
            if iₖ > num_recycled
                # delete the site referenced by iₖ from new_sites
                deleteat!( new_sites, iₖ - num_recycled)
            else
                # we have to sacrifice an old site :(
                deleteat!( recycled_indices, iₖ )
            end
            push!( new_sites, yₖ )
        else
            fully_linear = true;
            break;
        end

        # 3) Lagrange Basis update
        ## Normalization
        lagrange_basis[iₖ] /= lagrange_basis[iₖ]( yₖ )

        # Orthogonalization
        for j = 1 : p
            if j ≠ iₖ
                lagrange_basis[ j ] -= ( lagrange_basis[j]( yₖ) * lagrange_basis[ iₖ ] )
            end
        end
    end

    @info("\tFor a $Λ-poised set we can recycle $(length(recycled_indices)) sites.")
    @info("\tNeed to evaluate at $(length(new_sites)) additional sites.")

    new_indices = eval_new_sites( ac, new_sites );

    lmeta = LagrangeMeta( interpolation_indices = [recycled_indices; new_indices] )

    lmodel = LagrangeModel(
        n_out = objf.n_out,
        degree = cfg.degree,
        lagrange_basis = lagrange_basis,
        coefficients = get_training_values(objf, lmeta, iter_data),
        fully_linear = fully_linear
    )

    return lmodel, lmeta
end

@doc "Return vector of evaluations for output `ℓ` of a (vector) Lagrange Model
`lm` at scaled site `ξ`."
function eval_models( lm :: LagrangeModel, ξ :: Vector{Float64}, ℓ :: Int64 )
    sum( lm.coefficients[i][ℓ] * lm.lagrange_basis[i](ξ)
        for i = eachindex(lm.coefficients) )
end

function eval_models( lm :: LagrangeModel, ξ :: Vector{Float64})
    vcat( [ eval_models(lm, ξ, ℓ) for ℓ = 1 : lm.n_out ]... )
end

function get_gradient( lm :: LagrangeModel, ξ :: Vector{Float64}, ℓ :: Int64 )
    sum( lm.coefficients[i][ℓ] * gradient(lm.lagrange_basis[i], (ξ))
        for i = eachindex(lm.coefficients) )
end

function get_jacobian( lm :: LagrangeModel, ξ :: Vector{Float64})
    transpose( hcat( [ get_gradient(lm, ξ, ℓ) for ℓ = 1 : lm.n_out ]... ) )
end

make_linear!( ::LagrangeModel, ::LagrangeMeta, ::AlgoConfig, ::VectorObjectiveFunction, ::LagrangeConfig, ::Bool ) = false;
improve!( ::LagrangeModel, ::LagrangeMeta, ::AlgoConfig, ::VectorObjectiveFunction, ::LagrangeConfig )  = false;
