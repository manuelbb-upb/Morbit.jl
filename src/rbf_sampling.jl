# This file is NOT included from main module "Morbit.jl"
# Instead it is included in "RBFModel.jl" to provide
# `build_model`, `make_linear!` and `improve!`

using LinearAlgebra: norm, I, qr, cholesky, givens, pinv, Hermitian

include("PointSampler.jl")
using .PointSampler: MonteCarloThDesign, monte_carlo_th
using Base.Iterators: drop

# using `intersect_bounds` and `effective_bounds_vectors` from 'constraints.jl'

@doc "Return indices of sites in current database used for training."
function rbf_training_indices( mi :: RBFMeta )
    return [
        mi.center_index;
        mi.round1_indices;
        mi.round2_indices;
        mi.round3_indices;
        mi.round4_indices;
    ]
end

@doc "Return indices of sites in `id.sites_db`not used for training."
function non_rbf_training_indices( mi :: RBFMeta, id :: IterData )
    setdiff( 1 : length(id.sites_db), rbf_training_indices( mi ) )
end

@doc "Return list of training sites for current RBF Model described by `cfg` and `meta`."
function get_training_sites( meta :: RBFMeta, id :: IterData )
    return id.sites_db[ rbf_training_indices(meta) ]
end

@doc "Return list of training values for current RBF Model described by `cfg` and `meta`."
function get_training_values( objf :: VectorObjectiveFunction, meta :: RBFMeta, id :: IterData )
    [val[objf.internal_indices] for val ∈ id.values_db[ rbf_training_indices( meta ) ]]
end

@doc "Initialize an RbfModel from the information in `cfg::RbfConfig` and `meta::RBFMeta`."
function RBFModel( cfg :: RbfConfig, objf :: VectorObjectiveFunction, meta :: RBFMeta, id :: IterData )
   @unpack kernel, shape_parameter, polynomial_degree = cfg;
   shape_param = isa( shape_parameter, Function ) ? shape_parameter( id.Δ ) : shape_parameter;

   model = RBFModel(
       training_sites = get_training_sites( meta, id ),
       training_values = get_training_values( objf, meta, id ),
       kernel = kernel,
       shape_parameter = shape_param,
       polynomial_degree = polynomial_degree,
       fully_linear = meta.fully_linear
   );
   return model
end

function Z_from_Y( Y :: Array{Float64,2} )
    Q,_ = qr(Y);
    Z = Q[:, size(Y,2) + 1 : end];
    if size(Z,2) > 0
        Z ./= norm.( eachcol(Z), Inf )';
    end
    return Z
end

@doc """Find sites in provided array of sites that are well suited for polynomial interpolation.

If each site is in `\\mathbb R^n` than at most `n+1` sites will be returned."""
function find_affinely_independent_points(
       sites_array:: Vector{Vector{Float64}}, x :: Vector{Float64}, Δ :: Float64,
       θ_pivot ::Float64 , x_in_sites :: Bool, Y :: TY where TY <: Array{Float64,2},
       Z :: TZ where TZ <: Array{Float64,2} )

   n_vars = length(x);

   candidate_indices = find_points_in_box(x, Δ, sites_array, Val(x_in_sites));
   accepted_flags = zeros(Bool, length(candidate_indices));

   n_start = size(Y,2);
   n_missing = n_vars - n_start;

   # TODO PARALLELIZE inner LOOP
   min_projected_value = Δ * θ_pivot #*sqrt(n_vars);   # the projected value of any site (scaled by 1/(c_k*Δ) has to be >= θ_pivot)
   for iter = 1 : length(candidate_indices)

       largest_value = -Inf
       best_index = 0

       for index = findall( .!(accepted_flags ) )
           site_index = candidate_indices[ index ]
           site = sites_array[ site_index ] - x;   # TODO perform this before the loops
           proj_onto_Z = Z*(Z'site);       # Z'site gives the coefficient of each basis column in Z, left-multiplying by Z means summing
           site_value = norm( proj_onto_Z, Inf );

           if site_value > largest_value
               largest_value = site_value;
               best_index = index;
           end
       end # NOTE The Matlab version did not use nested For-Loops to find the *best* points, but simply used *sufficiently good* points

       # check measure for affine independence and add best point if admissible
       if largest_value > min_projected_value
           accepted_flags[best_index] = true;
           Y = hcat(Y, sites_array[ candidate_indices[best_index] ] .- x );
           Z = Z_from_Y(Y)
       end

       # break if enough points are found
       if sum( accepted_flags ) == n_missing
           break;
       end
   end

   if isempty(Z)
       Z = Matrix{typeof(x[1])}(undef, n_vars, 0);
   end

   @info("\t Found $(size(Y,2) - n_start) affinely independent sites in box with Δ=$Δ for pivot value $min_projected_value.")
   return (candidate_indices[ accepted_flags ], Y, Z)
end


function find_affinely_independent_points(
       sites_array:: Vector{Vector{Float64}}, x :: Vector{Float64}, Δ :: Float64 = 0.1,
       θ_pivot :: Float64 = 1e-3, x_in_sites :: Bool = false )
   n_vars = length(x);

   Y = Matrix{Float64}(undef, n_vars, 0);    # initializing a 0-column matrix enables hstacking without problems
   Z = zeros(Float64, n_vars, n_vars ) + I(n_vars);

   find_affinely_independent_points( sites_array, x, Δ, θ_pivot, x_in_sites, Y, Z)
end

@doc "Add points to model `m` and then train `m`."
function add_points!( m :: RBFModel, objf::VectorObjectiveFunction,
        meta_data :: RBFMeta, cfg :: RbfConfig, config_struct :: AlgoConfig )
    @unpack Δ_max, iter_data , problem = config_struct;
    @unpack θ_enlarge_2 = cfg;
    @unpack sites_db, x = iter_data;

    # Check whether there is anything left to do
    unused_indices = non_rbf_training_indices( meta_data, iter_data );
    if isempty(unused_indices)
        if isnothing(m.function_handle)
            train!(m)
        end
        return
    end

    ϑ_enlarge_2 = sensible_θ( Val(problem.is_constrained), θ_enlarge_2, x, Δ_max )
    big_Δ = ϑ_enlarge_2 * Δ_max;
    candidate_indices = unused_indices[ find_points_in_box( x, big_Δ, sites_db[ unused_indices ], Val(false) )]  # find points in box with radius 'big_Δ'
    reverse!(candidate_indices)

    # Prepare constants
    if cfg.max_model_points <= 0
        cfg.max_model_points = max( config_struct.n_vars^2, 2*config_struct.n_vars) + 1;
    end

    add_points!(m, objf, meta_data, cfg, candidate_indices, config_struct, Val(is_valid(m)))
end

@doc "Add points to an RBFModel `m` and train the model. Points are chosen near the current iterate."
function add_points!( m :: RBFModel, objf :: VectorObjectiveFunction, meta_data :: RBFMeta, cfg :: RbfConfig,
    candidate_indices :: Vector{Int64}, config_struct :: AlgoConfig, chol :: Val{false} )
   @unpack n_vars, Δ_max, iter_data, problem = config_struct;
   @unpack max_model_points, use_max_points, θ_enlarge_2 = cfg;
   @unpack x, Δ, sites_db, values_db = iter_data;

   @info "\t•Box search for more points."

   num_candidates = length(candidate_indices);
   if num_candidates <= max_model_points
       chosen_indices = candidate_indices
   else
       chosen_indices = reverse(candidate_indices)[1:max_model_points]
   end

   @info("\t\tFound $(length(chosen_indices)) additional training sites")
   push!(meta_data.round4_indices, chosen_indices...)

   training_indices = rbf_training_indices(meta_data)
   N = max_model_points - length(training_indices)
   if use_max_points && N > 0

       ϑ_enlarge_2 = sensible_θ( Val(problem.is_constrained), θ_enlarge_2, x, Δ_max )
       Δ_2 = ϑ_enlarge_2 * Δ_max;

       @info("Trying to actively sample $N additional sites to use full number of allowed sites.")
       lb_eff, ub_eff = effective_bounds_vectors(x, Δ_2, Val(problem.is_constrained));
       seeds = sites_db[training_indices];
       additional_sites = monte_carlo_th( max_model_points, lb_eff, ub_eff; seeds = seeds )[ n_training_sites + 1 : end]
       # batch evaluate
       new_indices = eval_new_sites( config_struct, additional_sites )
       push!(meta_data, round4_indices, new_indices...)
   end
   reset_m = RBFModel( cfg, objf, meta_data, iter_data )    # Reset to make sure all sites are included and scaled properly
   train!(reset_m)
   as_second!(m,reset_m)
   reset_m = nothing;
end


@doc "Add points to an RBFModel `m` and train the model. Points are chosen according
to the procedure described in (Wild,2008) by backtracking and Cholesky factorizations."
function add_points!( m :: RBFModel, objf :: VectorObjectiveFunction,
        meta_data :: RBFMeta, cfg :: RbfConfig, candidate_indices :: Vector{Int64},
        config_struct :: AlgoConfig, chol :: Val{true} )
    @unpack n_vars, Δ_max, iter_data, problem = config_struct;
    @unpack max_model_points, use_max_points, θ_enlarge_2, θ_pivot_cholesky = cfg;
    @unpack x, Δ, sites_db, values_db = iter_data;

    @info "\tBacktracking search for more points."

    ϑ_enlarge_2 = sensible_θ( Val(problem.is_constrained), θ_enlarge_2, x, Δ_max )
    Δ_2 = ϑ_enlarge_2 * Δ_max;

    # Initialize matrices
    Φ = get_Φ( m )     # TODO change when shape_parameter is made more adaptive
    Π = get_Π( m )
    Q, R = qr( Π' );
    R = [
       R;
       zeros( size(Q,1) - size(R,1), size(R,2) )
    ]
    Z = Q[:, min_num_sites( m ) + 1 : end ] # orthogonal basis for right kernel of Π, should be empty at this point i.e. == Matrix{Float64}(undef, size_Y + 1, 0)

    ZΦZ = Hermitian(Z'Φ*Z);

    L = cholesky( ZΦZ ).L     # should also be empty at this point
    Lⁱ = inv(L);

    φ₀ = Φ[1,1]; # constant value

    print_counter = 0;
    for i in candidate_indices
       y = sites_db[i];
       τ_y², Qy, Ry, Zy, Ly, Lyⁱ, Φy, Πy = test_y(m, y, φ₀, θ_pivot_cholesky, Q, R, Z, L, Lⁱ, Φ, Π )

       if !isempty(Qy)
           Q, R, Z, L, Lⁱ, Φ, Π = Qy, Ry, Zy, Ly, Lyⁱ, Φy, Πy
           print_counter += 1;
           push!(m.training_sites, y)
           push!(meta_data.round4_indices, i)
       end
    end
    @info("\t\tFound $print_counter additional sites.")

    N = max_model_points - length(m.training_sites)
    if use_max_points && N > 0
       @info("Trying to actively sample $N additional sites to use full number of allowed sites.")
       lb_eff, ub_eff = effective_bounds_vectors(x, Δ_2, Val(problem.is_constrained));
       seeds = m.training_sites;
       additional_sites = Vector{Vector{Float64}}();
       n_seeds = length(seeds)
       for y ∈ drop( MonteCarloThDesign( min( n_seeds + N , max(100, n_seeds + 30*N)) , lb_eff, ub_eff, seeds ), n_seeds )  # NOTE factor 30 chosen at random
           τ_y², Qy, Ry, Zy, Ly, Lyⁱ, Φy, Πy = test_y(m, y, φ₀, θ_pivot_cholesky, Q, R, Z, L, Lⁱ, Φ, Π )
           if !isempty(Qy)
               Q, R, Z, L, Lⁱ, Φ, Π = Qy, Ry, Zy, Ly, Lyⁱ, Φy, Πy
               push!(additional_sites, y)
               push!(m.training_sites, y)
               N -= 1
           end

           if N == 0
               break;
           end
       end
       # batch evaluate
       new_indices = eval_new_sites( config_struct, additional_sites )
       push!(meta_data.round4_indices, new_indices...)
    end
    reset_m = RBFModel( cfg, objf, meta_data, iter_data )    # Reset to make sure all sites are included and scaled properly
    train!(reset_m, Q, R, Z, L, Φ)
    as_second!(m,reset_m)
    reset_m = nothing;
end

function test_y(m, y, φ₀, θ_pivot_cholesky, Q, R, Z, L, Lⁱ, Φ, Π )
   φy = φ(m, y)
   Φy = [
       [Φ φy];
       [φy' φ₀]
   ]

   πy = Π_col( m, y );
   Ry = [
       R ;
       πy'
   ]

   # perform some givens rotations to turn last row in Ry to zeros
   row_index = size( Ry, 1)
   G = Matrix(I, row_index, row_index)
   for j = 1 : size(R,2)  # column index
       g = givens( Ry[j,j], Ry[row_index, j], j, row_index )[1];
       Ry = g*Ry;
       G = g*G;
   end
   Gᵀ = transpose(G)
   g̃ = Gᵀ[1 : end-1, end];   #last column
   ĝ = Gᵀ[end, end];

   Qg = Q*g̃;
   v_y = Z'*( Φ*Qg + φy .* ĝ );
   σ_y = Qg'*Φ*Qg + (2*ĝ)* φy'*Qg + ĝ^2*φ₀;

   τ_y² = σ_y - norm( Lⁱ * v_y, 2 )^2
   if τ_y² > θ_pivot_cholesky
       τ_y = sqrt(τ_y²)

       Qy = [
           [ Q zeros( size(Q,1), 1) ];
           [ zeros(1, size(Q,2)) 1.0 ]
       ] * Gᵀ


       z = [
           Q * g̃;
           ĝ
       ]
       Zy = [
           [
               Z;
               zeros(1, size(Z,2))
           ] z
       ]

       Lyⁱ = [
           [Lⁱ zeros(size(Lⁱ,1),1)];
           [ -(v_y'Lⁱ'Lⁱ)./τ_y 1/τ_y ]
       ];

       Ly = [
           [ L zeros(size(L,1), 1) ];
           [ v_y'Lⁱ' τ_y ]
       ]
       #=
       ZΦZ = [
           [ ZΦZ v_y ];
           [ v_y' σ_y ]
       ]
       =#

       Πy = [ Π πy ];
   else
       Qy = Ry = Zy = Ly = Lyⁱ = Φy = Πy = [];
   end

   return τ_y², Qy, Ry, Zy, Ly, Lyⁱ, Φy, Πy
end


################ formerly file "training.jl" ###################################

@doc "Return a new site (possibly satisfying box constraints) along direction `d`
eminating from `x` in box with radius `Δ`."
function sample_new(constrained_flag :: Bool, x :: Vector{R}, Δ :: Float64,
        direction :: Vector{D}, min_pivot_value :: Float64) where{ R <: Real ,D<:Real }

    stepsizes = intersect_bounds(x, direction, Δ, constrained_flag)      # how much to move into positive, negative direction?

    best_stepsize = let abssteps = abs.(stepsizes), i = argmax( abssteps );
        sign( stepsizes[i] ) * min( Δ, abssteps[i] )
    end

    if abs( best_stepsize ) >= min_pivot_value
        return new_site = x .+ best_stepsize .* direction
    else
        return []
    end
end

@doc "Return `N` new samples along the directions given by the columns of `Z`.
Return also (QR) updated matrices `Y` and `Z`."
function get_new_sites( :: Val{:orthogonal}, N :: Int64, x :: Vector{Float64}, Δ :: Float64,
        θ_pivot :: Float64, Y :: TY, Z :: TZ, constrained_flag :: Bool,
        seeds :: Vector{Vector{Float64}} = Vector{Vector{Float64}}() )where{
            TY <: AbstractArray, TZ <: AbstractArray }
    n_vars = length(x);
    N = min( N, size(Z,2) );

    additional_sites = Vector{Vector{Float64}}();
    min_pivot = Δ * θ_pivot # * sqrt(n_vars);
    @info("\t Sampling at $(N) new sites in Δ = $Δ, pivot value is $min_pivot.")
    for i = 1 : N
        new_site = sample_new(constrained_flag, x, Δ, Z[:,1], min_pivot)
        if isempty( new_site )
            break;
        end

        push!(additional_sites, new_site);
        Y = hcat( Y, new_site - x );
        Z = Z[:,2 : end] #Z_from_Y(Y);
    end
    return additional_sites, Y, Z
end

@doc "Use an iterator to sample `N` sites from a space-filling design (Crombecq, 2011).
The sites must be sufficiently linearly independent. Return also updated `Y` and `Z`."
function get_new_sites( :: Val{:monte_carlo}, N :: Int64, x :: Vector{Float64}, Δ :: Float64,
        θ_pivot :: Float64, Y :: TY, Z :: TZ, constrained_flag :: Bool,
        seeds :: Vector{Vector{Float64}} = Vector{Vector{Float64}} ) where{
            TY <: AbstractArray, TZ <: AbstractArray }
    n_vars = length(x);
    lb_eff, ub_eff = effective_bounds_vectors(x, Δ, Val(constrained_flag));

    additional_sites = Vector{Vector{Float64}}();
    min_pivot = Δ * θ_pivot #* sqrt(n_vars)
    @info("\t Sampling at $(N) new sites in Δ = $Δ, pivot value is $min_pivot.")
    n_seeds = length(seeds)
    for true_site ∈ drop( MonteCarloThDesign( n_seeds + 30 * N, lb_eff, ub_eff, seeds ), n_seeds ) # TODO the factor 30 was chosen at random
        site = true_site .- x;
        piv_val = norm(Z*(Z'site),Inf)
        if piv_val >= min_pivot
            Y = hcat(Y, site )
            Z = Z_from_Y( Y )
            push!(additional_sites, true_site);
            N -= 1
        end
        if N == 0
            break;
        end
    end

    # if not enough sites have been found, use orthogonal sampling
    n_still_missing = N - length(additional_sites)
    if n_still_missing > 0
        even_more_sites, Y, Z = get_new_sites( Val(:orthogonal), n_still_missing, x, Δ, θ_pivot, Y, Z, constrained_flag,
                [seeds; additional_sites] )
        push!(additional_sites, even_more_sites...)
    end

    return additional_sites, Y, Z
end

@doc """Build a new RBF model at iteration site x employing 4 steps:
    1) Find affinely independent points in slighly enlarged trust region.
    2) Find additional points in larger trust region until there are n+1 points.
    3) Sample at new sites to have at least n + 1 interpolating sites in trust region.
    4) Use up to p_max points to further improve the model.

    Newly sampled sites/values are added to the respective arrays.
"""
function build_rbf_model( config_struct :: AlgoConfig, objf :: VectorObjectiveFunction,
        cfg :: RbfConfig, criticality_round :: Bool = false )
    @info "Building an RBFModel for (internal indices) $(objf.internal_indices)."

    @unpack θ_enlarge_1, θ_enlarge_2, θ_pivot, sampling_algorithm, require_linear = cfg;
    @unpack n_vars, Δ_max, max_evals, problem = config_struct;
    @unpack iter_data = config_struct;
    @unpack x, x_index, Δ, sites_db = iter_data;

    # initalize new RBFModel
    meta_data = RBFMeta( center_index = x_index )

    other_indices = non_rbf_training_indices( meta_data, iter_data );

    ϑ_enlarge_1 = sensible_θ( Val(problem.is_constrained), θ_enlarge_1, x, Δ )
    ϑ_enlarge_2 = sensible_θ( Val(problem.is_constrained), θ_enlarge_2, x, Δ_max )

    Δ_1 = ϑ_enlarge_1 * Δ;
    Δ_2 = ϑ_enlarge_2 * Δ_max;

    # ============== Round 1 ======================#
    # find good points in database within slightly enlarged trust region
    new_indices, Y, Z = find_affinely_independent_points( sites_db[other_indices], x, Δ_1, θ_pivot, false);
    new_indices = other_indices[ new_indices ];

    meta_data.round1_indices = new_indices;
    setdiff!(other_indices, new_indices);

    meta_data.Y = Y;
    meta_data.Z = Z;  # columns contain model-improving directions

    # ============== Round 2 ======================#
    n_missing = n_vars - length(new_indices)
    if n_missing == 0
        @info("\tThe model is fully linear.")
        meta_data.fully_linear = true;
    else
        # n_missing > 0 ⇒ we have to search some more
        if !(criticality_round || require_linear)

            θ_pivot_2 = Δ_1/Δ_2 * θ_pivot

            # find additional points in bigger trust region
            new_indices, Y, Z = find_affinely_independent_points( sites_db[other_indices], x, Δ_2, θ_pivot_2, false, Y, Z )
            new_indices = other_indices[ new_indices ];

            if length(new_indices) > 0
                @info("\tThe model is not fully linear.")
                meta_data.round2_indices = new_indices;
                meta_data.fully_linear = false;
                setdiff!(other_indices, new_indices);
                n_missing -= length(new_indices);
            else
                meta_data.fully_linear = true;   # is made fully linear in Round 3
            end
        else
            meta_data.fully_linear = true; # is made fully linear in Round 3
        end
    end
    # ============== Round 3 ======================#
    # if there are still sites missing then sample them now

    if n_missing > 0
        if meta_data.fully_linear
            @info "\tThe model is (hopefully) made fully linear by sampling at $n_missing sites."
        else
            @info "\tThere are still $n_missing sites missing. Sampling..."
        end

        n_evals_left = min( max_evals, cfg.max_evals ) - numevals(objf) - 1;
        n_missing = Int64(min( n_missing , n_evals_left ));

        additional_sites, Y, Z = get_new_sites(Val(sampling_algorithm),
            n_missing, x, Δ_1, θ_pivot, Y, Z, problem.is_constrained,
            get_training_sites( meta_data, iter_data)
            )
        if length(additional_sites) < n_missing
            return rebuild_rbf_model(config_struct, objf, cfg)
        end

        meta_data.fully_linear = n_missing <= n_evals_left;

        additional_site_indices = eval_new_sites( config_struct, additional_sites )
        push!(meta_data.round3_indices, additional_site_indices...)
    end

    # ============== Round 4 ======================#

    m = RBFModel( cfg, objf, meta_data, iter_data);
    add_points!(m, objf, meta_data, cfg, config_struct)

    @info("\tModel$(m.fully_linear ? " " : " not ")linear!")
    return m, meta_data
end

@doc "Return a new RBFModel using samples along the coordinate axis at the current iterate."
function rebuild_rbf_model( config_struct :: AlgoConfig,
        objf :: VectorObjectiveFunction, cfg :: RbfConfig )
    @info "\tREBUILDING model along coordinate axes."
    @unpack n_vars, max_evals, iter_data, problem = config_struct
    @unpack θ_pivot, θ_enlarge_1 = cfg;
    @unpack x, x_index, Δ, sites_db = iter_data;

    meta_data = RBFMeta( center_index = x_index )
    ϑ_enlarge_1 = sensible_θ( Val(problem.is_constrained), θ_enlarge_1, x, Δ )
    Δ_1 = ϑ_enlarge_1 * Δ;
    min_pivot = Δ*θ_pivot;

    # sample along carthesian coordinates
    ## collect sites in an array (new_sites) to profit from batch evaluation afterwards
    Y = Matrix{Float64}(undef, n_vars, 0);
    n_evals_left = min(max_evals, cfg.max_evals) - numevals(objf) - 1;
    n_steps = Int64( min( n_vars, n_evals_left ) );
    new_sites = Vector{Vector{Float64}}();
    for i = 1 : n_steps
        direction = zeros(Float64, n_vars);
        direction[i] = 1.0;
        new_site = sample_new(true, x, Δ_1, direction, min_pivot)
        if !isempty(new_site)
            Y = hcat(Y, new_site .- x);
            push!(new_sites, new_site);
        else
            @info "Cannot rebuild a fully linear model, too near to boundary."
        end
    end

    # (batch) evaluate at new sites and push to database
    new_indices = eval_new_sites( config_struct, new_sites )
    fully_linear = n_vars == length(new_indices) ? true : false;

    # update model meta data and save indices of new sites
    push!(meta_data.round3_indices, new_indices...);
    meta_data.fully_linear = fully_linear;

    # build preliminary surrogate models
    m = RBFModel( cfg, objf, meta_data, iter_data);

    ## backtracking search in unused points
    add_points!(m, objf, meta_data, cfg, config_struct)

    return m, meta_data
end

# simply a wrapper to have a unified function signature
# # actual function below
function make_linear!(m :: RBFModel, meta_data :: RBFMeta, config_struct :: AlgoConfig,
        objf :: VectorObjectiveFunction, cfg :: RbfConfig, criticality_round :: Bool )
    make_linear!(m, meta_data, config_struct, objf, cfg, Val(criticality_round))
end

@doc "Perform several improvement steps on 'm::RBFModel'." #using the 'improve!' method until the model is fully linear on θ_enlarge_1*Δ."
function make_linear!(m :: RBFModel, meta_data :: RBFMeta, config_struct :: AlgoConfig,
        objf :: VectorObjectiveFunction, cfg :: RbfConfig)
    @unpack max_evals = config_struct;
    @unpack sites_db = config_struct.iter_data;

    evals_left = min(max_evals, cfg.max_evals) - numevals(objf) - 1;
    n_improvement_steps = Int64(min(size(meta_data.Z,2), evals_left));

    # number of columns of Z is the number of missing sites for full linearity
    # perform as many improvement steps and modify data in place
    for i = 1 : n_improvement_steps
        improvement_flag = improve!(m, meta_data, config_struct, objf, cfg; retrain = false )
        if !improvement_flag
            @info("Rebuilding close to boundary.")
            # build a new fully linear model for smaller trust region
            new_model, new_meta = rebuild_rbf_model( config_struct, objf, cfg )
            as_second!(m, new_model);       # modify old model to equal new model
            as_second!(meta_data, new_meta)
            new_model = nothing; new_meta = nothing; # for garbage collection
            break
        end
    end
    train!(m);
    return n_improvement_steps > 0
end


@doc "Perform ONE improvement step along model improving direction (first column of Z, where Z is an orthogonal to the site matrix Y).

The model parameters 'Y' and 'z' and its fully_linear flag are potentially modified.
Returns the newly sampled site and its value vector as given by f."
function improve!( m :: RBFModel, meta_data :: RBFMeta, config_struct :: AlgoConfig,
        objf :: VectorObjectiveFunction, cfg :: RbfConfig; retrain :: Bool = true )
    @unpack iter_data, problem, n_vars = config_struct;
    @unpack x, Δ, sites_db, values_db = iter_data;
    @unpack θ_pivot, θ_enlarge_1 = cfg;

    @info("\tImproving RBF Model with internal indices $(objf.internal_indices).")

    # Y and Z matrix are already associated with model
    Y = meta_data.Y;
    Z = meta_data.Z;

    num_Z_cols = size(Z,2)

    if num_Z_cols > 0  # should always be true during optimization routine, but usefull for make_linear!

        # add new site from model improving direction
        z = Z[:,1];     # improving direction
        min_pivot = Δ * θ_pivot # *sqrt(n_vars)
        ϑ_enlarge_1 = sensible_θ( Val(problem.is_constrained), θ_enlarge_1, x, Δ )
        Δ_1 = ϑ_enlarge_1 * Δ;  # maximum steplength

        @info("\t\tSampling with radius $Δ_1.")

        new_site = sample_new(problem.is_constrained, x, Δ_1, z, min_pivot)
        if isempty( new_site )
            @info "\t COULD NOT SAMPLE IN BOX!"
            new_model, new_meta = rebuild_rbf_model( config_struct, objf, cfg )
            as_second!(m, new_model);
            as_second!(meta_data, new_meta);
            new_model = nothing; new_meta = nothing;
            return false        # to tell 'make_linear!' that improvement was not successful
        end

        # update Y and Z matrices
        meta_data.Y = hcat( Y, new_site .- x );    # add new column to matrix Y
        meta_data.Z = Z[:,2:end]; #Z_from_Y(meta_data.Y);

        # evaluate at new site and store in database
        new_val = eval_all_objectives(problem, new_site );  # unscaling is taken care of
        push!(sites_db, new_site);
        push!(values_db, new_val);

        # update RBFMeta data and model
        push!(meta_data.round3_indices, numevals(objf));
        push!(m.training_sites, new_site);
        push!(m.training_values, new_val[objf.internal_indices] )

        if retrain train!(m); end
        if num_Z_cols == 1
            meta_data.fully_linear = m.fully_linear = true;
            @info("\t\tModel is now fully linear.")
        else
            meta_data.fully_linear = m.fully_linear = false;
            @info("\t\tModel is not fully linear and there still is an improving direction.")
        end
        return true
    else
        @warn "Empty return from improvement!"
        return false
    end
end
