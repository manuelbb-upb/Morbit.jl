function sample_new(constrained_flag :: Bool, x :: Vector{R}, Δ :: Float64, direction :: Vector{D}, min_pivot_value :: Float64, max_factor = 0.0) where{
    R,D<:Real
    }
    if max_factor == 0.0
        max_factor = Δ
    end

    #@assert isapprox(norm(direction, Inf),1) # TODO remove
    if constrained_flag
        σ₊, σ₋ = intersect_bounds(x, direction, Δ)      # how much to move into positive, negative direction?

        cap(σ) = abs(σ) > abs(max_factor) ? sign(σ) * abs(max_factor) : σ;

        # check whether we have sufficient independence in either admitted direction
        max_norm, arg_max = findmax( [ -σ₋ , σ₊] );

        if max_norm >= min_pivot_value
            if arg_max == 1
                new_site = x .+ cap(σ₋) .* direction
            else
                new_site = x .+ cap(σ₊) .* direction
            end
        else
            return []
        end
    else
        new_site = x + max_factor .* direction;
    end
    return new_site
end

function get_new_sites( :: Val{:orthogonal}, N :: Int64, x :: Vector{Float64}, Δ :: Float64,
        θ_pivot :: Float64, Y :: TY, Z :: TZ, constrained_flag :: Bool,
        seeds :: Vector{Vector{Float64}} = Vector{Vector{Float64}}() ) where{ TY, TZ <: AbstractArray }
    n_vars = length(x);
    additional_sites = Vector{Vector{Float64}}();
    min_pivot = Δ * θ_pivot# * sqrt(n_vars);

    @info("\t Sampling at $(N) new sites, pivot value is $min_pivot.")
    for i = 1 : N
        new_site = sample_new(constrained_flag, x, Δ, Z[:,1], min_pivot, Δ )
        if isempty( new_site )
            break;
        end

        push!(additional_sites, new_site);
        Y = hcat( Y, new_site - x );
        Z = Z[:, 2 : end];
    end
    return additional_sites, Y, Z
end

function get_new_sites( :: Val{:monte_carlo}, N :: Int64, x :: Vector{Float64}, Δ :: Float64,
        θ_pivot :: Float64, Y :: TY, Z :: TZ, constrained_flag :: Bool,
        seeds :: Vector{Vector{Float64}} = Vector{Vector{Float64}} ) where{ TY, TZ <: AbstractArray }
    n_vars = length(x);
    lb_eff, ub_eff = effective_bounds_vectors(x, Δ, Val(constrained_flag));

    additional_sites = Vector{Vector{Float64}}();
    min_pivot = Δ * θ_pivot #* sqrt(n_vars)

    for true_site ∈ drop( MonteCarloThDesign( 30 * N, lb_eff, ub_eff, seeds ), length(seeds) ) # TODO the factor 30 was chosen at random
        site = true_site .- x;
        piv_val = norm(Z*(Z'site),Inf)
        if piv_val >= min_pivot
            Y = hcat(Y, site )
            Q,_ = qr(Y);
            Z = Q[:, size(Y,2) + 1 : end];

            push!(additional_sites, true_site);
            N -= 1
        end
        if N == 0
            break;
        end
    end

    n_still_missing = N - length(additional_sites)
    if n_still_missing > 0
        even_more_sites, Y, Z = get_new_sites( Val(:orthogonal), n_still_missing, x, Δ, θ_pivot, Y, Z, constrained_flag,
                [seeds; additional_sites] )
        push!(additional_sites, even_more_sites...)
    end

    return additional_sites, Y, Z
end

function rebuild_model( config_struct :: AlgoConfig )
    @info "\tREBUILDING model along coordinate axes."
    @unpack n_vars, Δ_max, n_exp, n_cheap, θ_pivot, θ_enlarge_1, θ_enlarge_2,
        max_model_points, max_evals, rbf_poly_deg, rbf_kernel, rbf_shape_parameter = config_struct;

    @unpack problem = config_struct;

    @unpack iter_data = config_struct;
    @unpack x, f_x, x_index, Δ, sites_db, values_db = iter_data;

    iter_data.model_meta.model_info = ModelInfo( center_index = x_index )

    Δ_1 = θ_enlarge_1 * Δ;
    min_pivot = Δ*θ_pivot;

    # sample along carthesian coordinates
    ## collect sites in an array (new_sites) to profit from batch evaluation afterwards
    Y = Matrix{Float64}(undef, n_vars, 0);
    n_evals_left = max_evals - length(sites_db) - 1;
    n_steps = Int64( min( n_vars, n_evals_left ) );
    new_sites = Vector{Vector{Float64}}();
    for i = 1 : n_steps
        direction = zeros(Float64, n_vars);
        direction[i] = 1.0;
        new_site = sample_new(true, x, Δ_1, direction, min_pivot, Δ_1 )
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
    iter_data.model_meta.model_info.fully_linear = fully_linear;
    push!(iter_data.model_meta.model_info.round3_indices, new_indices...);

    # build preliminary surrogate models
    m = RBFModel(config_struct)

    ## backtracking search in unused points
    add_points!(m, config_struct)

    return m
end

@doc """Build a new RBF model at iteration site x employing 4 steps:
    1) Find affinely independent points in slighly enlarged trust region.
    2) Find additional points in larger trust region until there are n+1 points.
    3) Sample at new sites to have at least n + 1 interpolating sites in trust region.
    4) Use up to p_max points to further improve the model.

    Newly sampled sites/values are added to the respective arrays.
"""
function build_model( config_struct :: AlgoConfig, criticality_round :: Bool = false ) #, constrained_flag = false, criticality_round = false )

    if config_struct.n_exp > 0    # we only need the complicated sampling procedures for well-defined surrogates

        @unpack rbf_kernel, rbf_shape_parameter, rbf_poly_deg, θ_enlarge_1,
            θ_enlarge_2, θ_pivot, sampling_algorithm, problem = config_struct
        @unpack n_vars, Δ_max, max_evals = config_struct;
        @unpack iter_data = config_struct;
        @unpack x, x_index, Δ, sites_db = iter_data;

        # initalize new RBFModel
        iter_data.model_meta.model_info = ModelInfo( center_index = x_index )

        other_indices = non_rbf_training_indices( iter_data );

        Δ_1 = θ_enlarge_1 * Δ;
        Δ_2 = θ_enlarge_2 * Δ_max;

        # ============== Round 1 ======================#
        # find good points in database within slightly enlarged trust region
        new_indices, Y, Z = find_affinely_independent_points( sites_db[other_indices], x, Δ_1, θ_pivot, false);
        new_indices = other_indices[ new_indices ];
        @info("\tFound $(length(new_indices)) site(s) with indices $new_indices in first round with radius $Δ_1.")

        iter_data.model_meta.model_info.round1_indices = new_indices;
        setdiff!(other_indices, new_indices);

        iter_data.model_meta.model_info.Y = Y;
        iter_data.model_meta.model_info.Z = Z;  # columns contain model-improving directions

        # ============== Round 2 ======================#
        n_missing = n_vars - length(new_indices)
        if n_missing == 0
            @info("\tThe model is fully linear.")
            iter_data.model_meta.model_info.fully_linear = true;
        else
            # n_missing > 0 ⇒ we have to search some more
            if !criticality_round

                θ_pivot_2 = Δ_2/Δ_1 * θ_pivot

                @info("\tMissing $n_missing sites, searching in database for radius $Δ_2.")

                # find additional points in bigger trust region
                new_indices, Y, Z = find_affinely_independent_points( sites_db[other_indices], x, Δ_2, θ_pivot_2, false, Y, Z )
                new_indices = other_indices[ new_indices ];

                @info("\tFound $(length(new_indices)) site(s) in second round.")
                if length(new_indices) > 0
                    @info("\tThe model is not fully linear.")
                    iter_data.model_meta.model_info.round2_indices = new_indices;
                    iter_data.model_meta.model_info.fully_linear = false;
                    setdiff!(other_indices, new_indices);
                    n_missing -= length(new_indices);
                else
                    iter_data.model_meta.model_info.fully_linear = true;
                end
            else
                iter_data.model_meta.model_info.fully_linear = true;
            end
        end
        # ============== Round 3 ======================#
        # if there are still sites missing then sample them now

        if n_missing > 0
            if iter_data.model_meta.model_info.fully_linear
                @info "The model is (hopefully) made fully linear by sampling at $n_missing sites."
            else
                @info "There are still $n_missing sites missing. Sampling..."
            end

            n_evals_left = max_evals - length(sites_db) - 1;
            n_missing = Int64(min( n_missing , n_evals_left ));

            additional_sites, Y, Z = get_new_sites(Val(sampling_algorithm),
                n_missing, x, Δ_1, θ_pivot, Y, Z, problem.is_constrained,
                sites_db[rbf_training_indices(iter_data)]
                )
            if length(additional_sites) < n_missing
                return rebuild_model(config_struct)
            end

            iter_data.model_meta.model_info.fully_linear = n_missing <= n_evals_left;

            additional_site_indices = eval_new_sites( config_struct, additional_sites)
            push!(iter_data.model_meta.model_info.round3_indices, additional_site_indices...)
        end

        # ============== Round 4 ======================#

        m = RBFModel(config_struct);
        add_points!(m, config_struct)

        @info("\tModel$(m.fully_linear ? " " : " not ")linear!")

        return m
    else
        # if there are only cheap objectives return a NamedTuple instead of a surrogate
        return m = (
        function_handle = x -> Array{Float64,1}(),
        fully_linear = true ); # simply return named tuple
    end
end

# Phantom functions if there are only cheap objectives
function make_linear!(m :: NamedTuple, config_struct, crit :: Val{true}, constrained_flag = false, )
    return false
end

function make_linear!(m :: NamedTuple, config_struct, constrained_flag = false )
    return 0
end

@doc "Make the provided model fully linear assuming that the method is called from within the criticality loop of the main algorithm."
function make_linear!(m :: RBFModel, config_struct, crit :: Val{true})#, constrained_flag = false, )
    @unpack  θ_enlarge_1, rbf_shape_parameter = config_struct;
    @unpack Δ,model_meta = config_struct.iter_data;

    Δ_1 = Δ * θ_enlarge_1;
    allowed_flags_Y = norm.( eachcol(model_meta.model_info.Y), Inf ) .<= Δ_1    # are the 'linearizing' sites within Δ_1?
    #@show Δ_1

    if !all(allowed_flags_Y)
        new_model = build_model( config_struct, true );   # build a new fully linear model for smaller trust region
        as_second!(m, new_model);                                           # modify old model to equal new model
        return true  # tell main loop that model has changed
    else
        return false    # tell main loop that model has not changed and descent need not be recomputed
    end
end

@doc "Perform several improvement steps on 'm::RBFModel' using the 'improve!' method until the model is fully linear on θ_enlarge_1*Δ."
function make_linear!(m::RBFModel, config_struct) #, constrained_flag = false )
    @unpack max_evals = config_struct;
    @unpack sites_db , model_meta =config_struct.iter_data;

    evals_left = max_evals - length(sites_db) - 1;
    n_improvement_steps = Int64(min(size(model_meta.model_info.Z,2), evals_left));

    # number of columns of Z is the number of missing sites for full linearity
    # perform as many improvement steps and modify data in place
    for i = 1 : n_improvement_steps
        improvement_flag = improve!(m, config_struct)#, constrained_flag );
        if !improvement_flag
            new_model = rebuild_model(config_struct);
            as_second!(m, new_model)
            break
        end
    end
    return n_improvement_steps
end


@doc "Perform ONE improvement step along model improving direction (first column of Z, where Z is an orthogonal to the site matrix Y).

The model parameters 'Y' and 'z' and its fully_linear flag are potentially modified.
Returns the newly sampled site and its value vector as given by f."
function improve!( m::RBFModel, config_struct :: AlgoConfig)#, constrained_flag = false )
    @unpack n_vars, n_exp, θ_pivot, θ_enlarge_1 = config_struct;
    @unpack problem = config_struct;
    @unpack iter_data = config_struct;
    @unpack x, Δ, sites_db, values_db , model_meta = iter_data;

    # Y and Z matrix are already associated with model
    Y = model_meta.model_info.Y;
    Z = model_meta.model_info.Z;

    if size(Z,2) > 0  # should always be true during optimization routine, but usefull for make_linear!
        @info("\t\tSampling with radius $(θ_enlarge_1 * Δ).")

        # add new site from model improving direction
        z = Z[:,1];
        min_pivot = Δ*θ_pivot # *sqrt(n_vars)
        Δ_1 = θ_enlarge_1 * Δ;

        new_site = sample_new(problem.is_constrained, x, Δ_1, z, min_pivot , Δ_1)
        if isempty( new_site )
            @info "\t COULD NOT SAMPLE IN BOX!"
            new_model = rebuild_model( config_struct )
            as_second!(m, new_model);
            return false        # to tell 'make_linear!' that improvement was not successful
        end

        # update Y matrix
        Y = hcat( Y, new_site - x );    # add new column to matrix Y

        Z = Z[:,2:end];

        # save updated matrices to TrainingData object
        model_meta.model_info.Y = Y;
        model_meta.model_info.Z = Z;

        # update rbf model
        new_val = eval_all_objectives(problem, new_site );
        #=if isempty(size(new_val))
            new_val = [ new_val, ];
        end=#

        push!(sites_db, new_site);
        push!(iter_data, new_val);
        # update info
        push!(model_meta.model_info.round3_indices, length(sites_db));
        push!(m.training_sites, new_site);
        m.training_values = get_training_values( config_struct )

        if size(Z,2) == 0
            model_meta.model_info.fully_linear = m.fully_linear = true;
            @info("\t\tModel is now fully linear.")
        else
            model_meta.model_info.fully_linear = m.fully_linear = false;
            @info("\t\tModel is not fully linear and there is an improving direction.")
        end

        train!(m);
        return true

    else
        @warn "empty return from improvement!"
        return false
    end
end
