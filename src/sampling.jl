 #randlin(n) = 1 .- sqrt.( 1 .- rand(n) );   # sample randomly according to pdf 2(1-x)
randquad(n) = -( 1 .- rand(n) ).^(1/3) .+ 1; # sample randomly according to pdf 3(x-1)^2
randquad() = randquad(1)[end];
randquart(n) = - ( 1 .- rand(n) ).^(1/5) .+ 1;
randquart() = - ( 1 - rand() )^(1/5) + 1

function RBFModel( config_struct :: AlgoConfig )
    @unpack rbf_kernel, rbf_shape_parameter, rbf_poly_deg, iter_data = config_struct;
    @unpack sites_db = iter_data;

    training_indices = rbf_training_indices( iter_data );
    return RBFModel(
        training_sites = sites_db[ training_indices ],
        training_values = get_training_values( config_struct, training_indices ),
        kernel = rbf_kernel,
        shape_parameter = rbf_shape_parameter( config_struct ),
        polynomial_degree = config_struct.rbf_poly_deg,
        fully_linear = iter_data.model_meta.model_info.fully_linear
    )
end

function reset_training_values!( m::RBFModel, config_struct :: AlgoConfig )
    ti = rbf_training_indices( config_struct.iter_data )
    m.training_values = get_training_values( config_struct, ti );

    m.rbf_coefficients = Matrix{Float64}(undef,0,0);
    m.poly_coefficients = Matrix{Float64}(undef,0,0);
    m.function_handle = nothing;
    return ti
end

function reset_training_data!( m::RBFModel, config_struct :: AlgoConfig )
    training_indices = reset_training_values!(m,config_struct)
    m.training_sites = config_struct.iter_data.sites_db[training_indices];
end


@doc "Return indices of sites in `sites_array` so that `x .- Δ <= site <= x .+ Δ` and exclude index of `x` if contained in `sites_array`."
function find_points_in_box( x, Δ, sites_array, filter_x :: Val{true} )
    x_lb = x .- Δ;
    x_ub = x .+ Δ;
    candidate_indices = findall( site -> all( x_ub .>= site .>= x_lb ) && !isapprox(x, site, rtol = 1e-14), sites_array);   # TODO check isapprox relative tolerance
end

@doc "Return indices of sites in `sites_array` so that `x .- Δ <= site <= x .+ Δ` and assume that `sites_array` does not contain `x`."
function find_points_in_box( x, Δ, sites_array, filter_x :: Val{false} )
    x_lb = x .- Δ;
    x_ub = x .+ Δ;
    candidate_indices = findall( [ all( x_lb .<= site .<= x_ub ) for site ∈ sites_array ] )
end

function eval_new_sites( config_struct :: AlgoConfig, additional_sites :: Vector{Vector{Float64}})
    @info("Evaluating at $(length(additional_sites)) new sites.")
    @unpack iter_data = config_struct;
    additional_values = eval_all_objectives.(config_struct, additional_sites)
    additional_indices = let nold = length(iter_data.sites_db), nnew = length(additional_sites);
        nold + 1 : nold + nnew
    end

    push!(iter_data.sites_db, additional_sites...)
    push!(iter_data, additional_values...)
    return additional_indices
end

@doc """Find sites in provided array of sites that are well suited for polynomial interpolation.

If each site is in `\\mathbb R^n` than at most `n+1` sites will be returned."""
function find_affinely_independent_points(
        sites_array:: Vector{Vector{Float64}}, x :: Vector{Float64}, Δ :: Float64,
        θ_pivot ::Float64 , x_in_sites :: Bool, Y :: TY , Z :: TZ
        ) where{ TY, TZ <: AbstractArray{Float64} }
    n_vars = length(x);

    candidate_indices = find_points_in_box(x, Δ, sites_array, Val(x_in_sites));
    accepted_flags = zeros(Bool, length(candidate_indices));

    n_missing = n_vars - size(Y,2);

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
            Y = hcat(Y, sites_array[ candidate_indices[best_index]] .- x );
            Q,_ = qr(Y);
            Z = Q[:, size(Y,2) + 1 : end];
            if size(Z,2) > 0
                Z ./= norm.( eachcol(Z), Inf )';
            end
        end

        # break if enough points are found
        if sum( accepted_flags ) == n_missing
            break;
        end
    end

    if isempty(Z)
        Z = Matrix{typeof(x[1])}(undef, n_vars, 0);
    end
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


function add_points!( m :: RBFModel, config_struct :: AlgoConfig )
    add_points!(m, config_struct, Val(is_valid(m)))
end

function add_points!( m :: RBFModel, config_struct :: AlgoConfig, :: Val{false} )
    @unpack n_vars, use_max_points, max_model_points, θ_enlarge_2, θ_pivot_cholesky, Δ_max, iter_data, problem = config_struct;
    @unpack x, Δ, sites_db, values_db, model_meta = iter_data;

    # Check whether there is anything left to do
    unused_indices = non_rbf_training_indices( iter_data );
    if isempty(unused_indices)
        if isnothing(m.function_handle)
            train!(m)   # TODO think about this
        end
        return
    end
    @info "Box search for more points."
    # Prepare constants
    if max_model_points <= 0
        max_model_points = 2 * n_vars^2 + 1;
    end
    big_Δ = θ_enlarge_2 * Δ;

    candidate_indices = unused_indices[ find_points_in_box( x, big_Δ, sites_db[ unused_indices ], Val(false) )]  # find points in box with radius 'big_Δ'
    reverse!(candidate_indices)

    num_candidates = length(candidate_indices);
    if num_candidates <= max_model_points
        chosen_indices = candidate_indices
    else
        chosen_indices = Int64[];
        while length(chosen_indices) < max_model_points
            ci = ceil(Int64, randquart() * (num_candidates-1));
            ind = candidate_indices[ ci ]
            if ind ∉ chosen_indices
                push!(chosen_indices, ind)
            end
        end
    end

    @info("Found $(length(chosen_indices)) additional training sites")
    push!(model_meta.model_info.round4_indices, chosen_indices...)

    training_indices = rbf_training_indices(iter_data)
    N = max_model_points - length(training_indices)
    if use_max_points && N > 0
        @info("Trying to actively sample $N additional sites to use full number of allowed sites.")
        lb_eff, ub_eff = effective_bounds_vectors(x, θ_enlarge_2 * Δ_max, Val(problem.is_constrained));
        seeds = sites_db[training_indices];
        additional_sites = monte_carlo_th( max_model_points, lb_eff, ub_eff; seeds = seeds )[ n_training_sites + 1 : end]
        # batch evaluate
        new_indices = eval_new_sites( config_struct, additional_sites )
        push!(model_meta.model_info.round4_indices, new_indices...)
    end
    reset_training_data!(m, config_struct)  # NOTE IMPORTANT this requires model_meta.model_info to be correct!!
    train!(m)
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


function add_points!( m :: RBFModel, config_struct :: AlgoConfig, :: Val{true} )
    @unpack n_vars, max_model_points, θ_enlarge_2, θ_pivot_cholesky, Δ_max,
        use_max_points, iter_data, problem = config_struct;
    @unpack x, Δ, sites_db, values_db, model_meta = iter_data;

    # Check whether there is anything left to do
    unused_indices = non_rbf_training_indices( iter_data );
    if isempty(unused_indices)
        if isnothing(m.function_handle)
            train!(m)   # TODO think about this
        end
        return
    end
    if length(m.training_sites) < min_num_sites( m )
        @error("Function 'add_points!' should only be used if sufficiently many points for interpolation have been determined.")
    end

    @info "\tBacktracking search for more points."
    # Prepare constants
    if max_model_points <= 0
        max_model_points = 2 * n_vars^2 + 1;
    end
    big_Δ = θ_enlarge_2 * Δ;

    # find suitable site indices to test
    candidate_indices = unused_indices[ find_points_in_box( x, big_Δ, sites_db[ unused_indices ], Val(false) )]  # find points in box with radius 'big_Δ'
    reverse!(candidate_indices)     # now the more recently added sites are tested first; these sites tend to be closer to the current iterate

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
            push!(model_meta.model_info.round4_indices, i)
        end
    end
    @info("Found $print_counter additional sites.")

    N = max_model_points - length(m.training_sites)
    if use_max_points && N > 0
        @info("Trying to actively sample $N additional sites to use full number of allowed sites.")
        lb_eff, ub_eff = effective_bounds_vectors(x, θ_enlarge_2 * Δ_max, Val(problem.is_constrained));
        seeds = m.training_sites;
        additional_sites = Vector{Vector{Float64}}();

        for y ∈ drop( MonteCarloThDesign( 30 * N, lb_eff, ub_eff, seeds ), length(seeds) )  # NOTE factor 30 chosen at random
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
        push!(model_meta.model_info.round4_indices, new_indices...)
    end
    reset_training_data!(m, config_struct)
    train!(m, Q, R, Z, L, Φ)
end
