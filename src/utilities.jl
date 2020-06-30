 #randlin(n) = 1 .- sqrt.( 1 .- rand(n) );   # sample randomly according to pdf 2(1-x)
randquad(n) = -(-(rand(n) .- 1)).^(1/3) .+ 1; # sample randomly according to pdf 3(x-1)^2
randquad() = randquad(1)[end];

function find_points_in_box( x, Δ, sites_array )
    x_lb = x .- Δ;
    x_ub = x .+ Δ;  # in the most simple case of symmetrical boxes an absolute value test would also work but this approach can be extended to include global boundaries
    candidate_indices = findall( site -> all( x_ub .>= site .>= x_lb ) && !isapprox(x, site, rtol = 1e-14), sites_array);   # TODO check isapprox relative tolerance
end

@doc """Find sites in provided array of sites that are well suited for polynomial interpolation.

If each site is in `\\mathbb R^n` than at most `n+1` sites will be returned."""
function find_affinely_independent_points( sites_array::Array{Array{Float64,1},1}, x, Δ = 0.1; θ_pivot = 1e-3, Y = [], Z = [] )
    n_vars = length(x);

    candidate_indices = find_points_in_box(x, Δ, sites_array);
    accepted_flags = zeros(Bool, length(candidate_indices));

    # if Y and Z have been passed as kw arguments, the function is called a second time to complete a prior search on a bigger region
    if isempty( Y )
        # if Y has no columns then the standard basis spans its orthogonal complement
        Y = Matrix{typeof(x[1])}(undef, n_vars, 0);    # initializing a 0-column matrix enables hstacking without problems
        Z = zeros( n_vars, n_vars ) + I(n_vars);
        n_missing = n_vars;
    else
        n_missing = n_vars - size(Y,2);
    end

#    println("\t\t\tThere are $(length(candidate_indices)) candidates for radius $Δ.")

    min_projected_value = sqrt(n_vars) * Δ * θ_pivot;   # the projected value of any site (scaled by 1/(c_k*Δ) has to be >= θ_pivot)
#    println("\t\t\tMin projected value must be $min_projected_value")
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

# adds newly found sites and valus to model, update
function additional_points!( m::RBFModel, x :: Array{Float64,1}, n_exp, sites_array::Array{Array{Float64, 1},1}, values_array::Array{Array{Float64, 1},1}, Δ :: Float64, θ_pivot :: Float64, max_model_points :: Int64 = 0 )
    # follows notation of Wild, "Derivative free optimization algorithms for computationally expensive Functions"

    n_vars = length(x);

    if max_model_points == 0
        max_model_points = 2 * n_vars^2 + 1;
    end

    if isempty( sites_array )
        return []
    end

    #println("\tTrying to find $(max_model_points - n_vars - 1) more model enhancing sites out of unused $(length(sites_array)-1) sites in database.") # NOTE -1 for x itself

    # find all relevant points in trust region
    candidate_indices = find_points_in_box(x, Δ, sites_array);

    # sort candidate sites by distance
    distances = norm.( site - x for site = sites_array[candidate_indices] );
    sorting_index = sortperm( distances );
    candidate_indices = candidate_indices[ sorting_index ];
    #println("\tThere are $(length(candidate_indices)) candidates.")

    accepted_flags = zeros(Bool, length(candidate_indices));

    if m.fully_linear
        Y = m.tdata.Y; # does not work if model not fully linear !!!
    else
        Y = hcat( [v - m.training_sites[1] for v ∈ m.training_sites[2:end] ]... );
    end

    if size(Y,2) != n_vars
        @show size(Y)
        @warn("Function 'find_additional_points' should only be used if sufficiently many points for interpolation have been determined.")
    end
    size_Y = size(Y,2);

    Φ,φ  = build_Φ( m )     # TODO change when shape_parameter is made more adaptive

    Π = [ [zeros( size(Y,1),1) Y]; ones(1, size_Y + 1) ];  # polynomial basis matrix, (size_Y + 1) × (size_Y + 1) (columns are added)
    Q, R = qr( Π' );
    Z = Q[:, size(Π,1) + 1 : end]; # orthogonal basis for right kernel of Π, should be empty at this point i.e. == Matrix{Float64}(undef, size_Y + 1, 0)

    ZΦZ = Z'Φ*Z;    # empty now (maybe wrap in Hermitian )

    L = cholesky( ZΦZ ).L     # should also be empty at this point
    L_inv = inv(L);

    φ_0 = Φ[1,1]; # constant value defined in rbf.jl; the same in each iteration

    n_points = size_Y + 1;
    for i = eachindex(candidate_indices) # TODO consider nested for loop as in 'find_affinely_independent_points'
        site_index = candidate_indices[i]
        k = sum(accepted_flags)
        if k + n_points < max_model_points # i.e. if additional columns are allowed

            y = sites_array[ site_index ];
            φ_y = φ(y)
            y -= x; # translate into local coordinate system

            # update qr decomposition of Π_y = [Π [y;1]]

            #Π_y = [ Π [y;1] ];  # for testing and debugging
            Φ_y = [ [Φ φ_y]; [φ_y' φ_0] ];

            R_y = [ R; [y' 1] ]

            row = n_points + k + 1;
            G = zeros(row,row) + I(row);    # matrix of givens rotations to eliminate last row in R_y
            for j = 1 : size_Y + 1
                # eliminite element j in last row of R_y
                g = givens( R_y[j,j], R_y[row, j], j, row )[1];
                R_y = g*R_y;
                G = g*G;
            end
            Gᵀ = G';
            g̃ = Gᵀ[1 : end-1, end];   #last column
            ĝ = Gᵀ[end, end];

            Qg = Q*g̃;
            v_y = Z'*( Φ*Qg + φ_y .* ĝ );   # (0 x n_vars) * (  (n_vars + 1 x n_vars + 1) *  )
            σ_y = Qg'*Φ*Qg + (2*ĝ)* φ_y'*Qg + ĝ^2*φ_0;

            τ_y² = σ_y - norm( L_inv * v_y, 2 )^2
            if τ_y² > θ_pivot^2
                τ_y = sqrt(τ_y²)
                accepted_flags[i] = true;

                # update matrices
                R = R_y;

                # update orthogonal basis of left kernel to Π
                new_z = [ Q * g̃; ĝ ];
                Z = [ [Z; zeros(1, size(Z,2))] new_z] ;

                L = [
                    [L zeros(size(L,1), 1)];
                    [(L_inv * v_y)' τ_y]
                ];

                L_inv = [
                    [L_inv zeros(size(L_inv,1), 1)];
                    [-(v_y'L_inv'L_inv)./τ_y 1/τ_y]
                ];

                ZΦZ = [
                    [ZΦZ v_y];
                    [v_y' σ_y]
                ];

                Q_y = [
                    [Q zeros( size(Q,1), 1 )];
                    [zeros(1, size(Q,2)) 1 ]
                ]
                Q = Q_y * Gᵀ;  # TODO CHECK IF CORRECT!!

                Π = [ Π [y;1] ];
                #Y = hcat(Y,y);

                y = y+x;    # translate back into global coordinate system

                # update φ function to include new basis site
                push!(m.training_sites, y)
                push!(m.training_values, values_array[site_index][1:n_exp] )
                Φ,φ = build_Φ( m );
            end
        else
            break;
        end
    end

    return candidate_indices[accepted_flags], Y, Π, Q, R, Z, tril(L)
end
