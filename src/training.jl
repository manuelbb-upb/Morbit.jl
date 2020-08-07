function sample_new(constrained_flag, x, Δ, direction, min_pivot_value, max_factor = 0.0)
    if max_factor == 0.0
        max_factor = Δ
    end

    #@assert isapprox(norm(direction, Inf),1) # TODO remove
    if constrained_flag
        σ₊, σ₋ = .9999 .* intersect_bounds(x, direction, Δ)      # how much to move into positive, negative direction?

        cap(σ) = abs(σ) > abs(max_factor) ? sign(σ) * abs(max_factor) : σ;

        # check whether we have sufficient independence in either admitted direction
        max_norm, arg_max =  findmax( [ -σ₋ , σ₊] );

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

function rebuild_model( config_struct :: AlgoConfig )
    @info "\tREBUILDING model along coordinate axes."
    @unpack n_vars, Δ_max, n_exp, n_cheap, θ_pivot, θ_enlarge_1, θ_enlarge_2, θ_pivot_cholesky, max_model_points, max_evals, rbf_poly_deg, rbf_kernel, rbf_shape_parameter = config_struct;

    @unpack problem = config_struct;

    @unpack iter_data = config_struct;
    @unpack x, f_x, Δ, sites_db, values_db = iter_data;

    m = RBFModel(
        kernel = rbf_kernel,
        shape_parameter = rbf_shape_parameter(Δ),
        fully_linear = true;
        polynomial_degree = rbf_poly_deg
        );

    Δ_1 = θ_enlarge_1 * Δ;
    Δ_2 = θ_enlarge_2 * Δ_max;
    min_pivot = Δ*θ_pivot;#sqrt(n_vars) * Δ * θ_pivot;

    # sample along carthesian coordinates
    Y = Matrix{Float64}(undef, n_vars, 0);
    n_evals_left = max_evals - length(sites_db) - 1;
    n_steps = Int64( min( n_vars, n_evals_left ) );
    new_sites = Vector{Vector{Float64}}();
    for i = 1 : n_steps
        direction = zeros(Float64, n_vars);
        direction[i] = 1.0;
        new_site = sample_new(true, x, Δ_1, direction, min_pivot, rand([-1.0, 1.0]) * Δ_1 )
        if !isempty(new_site)
            Y = hcat(Y, new_site .- x);
            push!(new_sites, new_site);
        else
            @info "Cannot rebuild a fully linear model, too near to boundary."
        end
    end
    new_values = eval_all_objectives.(problem, new_sites);
    push!(sites_db, new_sites...);
    push!(iter_data, new_values...);

    @info("\tSampled $n_steps new sites near variable boundaries.")

    if n_vars != n_steps
        Q, _ = qr(Y);
        Z = Q[:, size(Y,2) + 1 : end];
        m.fully_linear = false;
    else
        Z = Matrix{Float64}(undef, n_vars, 0);
    end

    new_tdata = TrainingData( Y = Y, Z = Z);

    new_indices = (length(sites_db) - n_steps + 1) : length(sites_db);
    old_indices = 1 : (length(sites_db) - n_steps);

    m.training_sites = [[x,]; sites_db[ new_indices ]];
    m.training_values = [[f_x[1:n_exp],]; [ v[1:n_exp] for v ∈ values_db[ new_indices ] ] ];
    m.tdata = new_tdata;

    # look for EVEN MORE model enhancing points in big radius, ignore points considered already
    more_site_indices, _, Π, Q, R, Z3, L = additional_points!( m, x, n_exp, sites_db[ old_indices ], values_db[ old_indices ], Δ_2, θ_pivot_cholesky, max_model_points ); # sites are added to m in-place
    @info("\tFound $(length(more_site_indices)) sites to further enhance the model.")
    train!(m,  Π, Q, R, Z3, L);

    # update model_info
    m.model_info.round1_indices = new_indices;
    m.model_info.round3_indices = more_site_indices;
    m.model_info.fully_linear = m.fully_linear;

    return m
end

@doc """Build a new RBF model at iteration site x employing 3 steps:
    1) Find affinely independent points in slighly enlarged trust region.
    2) Find additional points in larger trust region until there are n+1 points.
    3) Use up to p_max points to further improve the model.

    Newly sampled sites/values are added to the respective arrays.
"""
function build_model( config_struct :: AlgoConfig, constrained_flag = false, criticality_round = false )
    @unpack problem = config_struct;
    @unpack n_vars, n_exp, n_cheap, max_evals = config_struct;
    @unpack sampling_algorithm, use_max_points = config_struct;
    @unpack iter_data = config_struct;
    @unpack x, f_x, Δ, sites_db, values_db = iter_data;

    if n_exp > 0    # we only need the complicated sampling procedures for well-defined surrogates
        @unpack n_vars, rbf_kernel, rbf_shape_parameter, rbf_poly_deg, max_model_points, Δ_max, θ_enlarge_1, θ_enlarge_2, θ_pivot, θ_pivot_cholesky = config_struct;

        Δ_1 = θ_enlarge_1 * Δ;
        Δ_2 = θ_enlarge_2 * Δ_max;

        # ============== Round 1 ======================#
        # find good points in database within slightly enlarged trust region
        model_point_indices, Y1, Z1 = find_affinely_independent_points( sites_db, x, Δ_1, θ_pivot = θ_pivot )
        @info("\tFound $(length(model_point_indices)) site(s) in first round with radius $Δ_1.")

        # ============== Round 2 ======================#
        θ_pivot_2 = Δ_2/Δ_1 * θ_pivot
        Y2 = Y1;
        Z2 = Z1;

        if !isempty( Z1 ) && !criticality_round     # don't enhance in criticallity loop, because we need a fully linear model
            @info("\tMissing $(length(x) - length(model_point_indices)) sites, searching in database for radius $Δ_2.")

            # find additional points in bigger trust region
            other_sites_indices = setdiff(1 : length( sites_db), model_point_indices )   # ignore all points from first round

            additional_point_indices_sub, Y2, Z2 = find_affinely_independent_points( sites_db[other_sites_indices], x, Δ_2; θ_pivot = θ_pivot_2, Y = Y1, Z = Z1 )
            additional_point_indices = other_sites_indices[ additional_point_indices_sub ];

            @info("\tFound $(length(additional_point_indices)) site(s) in second round.")
            push!(model_point_indices, additional_point_indices...);
        end

        # ============== Round 3 ======================#
        # if there are still sites missing then sample them now
        min_pivot = sqrt(n_vars) * Δ * θ_pivot;

        # determine 'fully_linear' and 'new_tdata'
        n_missing = 0;
        if !isempty(Z1)
            if Y1 == Y2
                @info("\tNo second round points, model is made fully linear.")
                #fully_linear = true;
                Y = Y1;
                Z = Z2;
                n_missing = size( Z, 2 );
            else
                @info("\tThe model is not fully linear.")
                Y = Y2;
                Z = Z2;
                fully_linear = false;
                new_tdata = TrainingData(Y = Y1, Z = Z1);    # save data from round 1 for improvement steps
            end
        else
            @info("\tFound sufficiently many points in first sampling round.")
            fully_linear = true;
            Y = Y1;
            Z = Z2;
            new_tdata = TrainingData(Y = Y1, Z = Z1);
        end

        if n_missing > 0
        # sample at additional sites to obtain exactly n_vars (+1) training sites_array

            n_evals_left = max_evals - length(sites_db) - 1;
            n_missing = Int64(min( n_missing , n_evals_left ));

            fully_linear = n_missing <= n_evals_left;

            min_pivot = sqrt(n_vars) * Δ * θ_pivot;
            additional_sites = Array{Array{Float64,1},1}();

            if sampling_algorithm == :orthogonal

                @info("\t $(length(sites_db)) evals so far. Sampling at $(n_missing) new sites, pivot value is $min_pivot.")
                for i = 1 : n_missing
                    factor = rand([-1.0, 1.0]) * Δ_1;# (fully_linear ? Δ_1 : (Δ_1 + ( Δ_2 - Δ_1) * randquad()));   # TODO check if sensible: if model is *not* fully linear then samples are generated in a larger region
                    new_site = sample_new(constrained_flag, x, Δ_1, Z[:,1], min_pivot, factor )
                    if isempty( new_site )
                        return rebuild_model( config_struct );  # delegate to rebuild function
                    end

                    push!(additional_sites, new_site);
                    Y = hcat( Y, new_site - x );
                    Z = Z[:, 2 : end];
                end
                if !fully_linear
                    @warn "\tNot enough computational budget left to garantuee a unique model."
                end

            elseif sampling_algorithm == :monte_carlo

                seeds = sites_db[model_point_indices];      # TODO maybe include more points from Δ_2 ?
                lb_eff, ub_eff = effective_bounds_vectors(x, Δ);
                n_model_points = length( model_point_indices ) + 1
                for true_site ∈ drop( MonteCarloThDesign( 100 * n_model_points, lb_eff, ub_eff, seeds ), n_model_points )
                    site = true_site .- x;
                    if norm(Z*(Z'site),Inf) >= min_pivot
                        Y = hcat(Y, site )
                        Q,_ = qr(Y);
                        Z = Q[:, size(Y,2) + 1 : end];

                        push!(additional_sites, true_site);
                        n_missing -= 1
                    end
                    if n_missing == 0
                        break;
                    end
                end

                if n_missing > 0
                    return rebuild_model( config_struct );
                end
            end
            new_tdata = TrainingData( Y= Y, Z = Z); # TODO does it make sense for !fully_linear???

            additional_values = eval_all_objectives.( problem, additional_sites );
            additional_site_indices = length( sites_db ) + 1 : length( sites_db ) + length( additional_sites )  # for filtering out the new sites in 3rd search below

            push!(sites_db, additional_sites...)    # push new samples into database
            push!(iter_data, additional_values...)
        else
            additional_site_indices = Vector{Int64}();
        end

        round1_indices = model_point_indices;   # renaming for model info array
        push!(model_point_indices, additional_site_indices...);

        # ============== Round 4 ======================#
        # construct a preliminary (and untrained) rbf model
        training_sites = [[x, ]; sites_db[ model_point_indices ] ];
        training_values = [ v[1:n_exp] for v in [[f_x, ]; values_db[ model_point_indices ]] ];   # use only expensive objective values

        m = RBFModel(
            training_sites = training_sites,
            training_values = training_values,
            kernel = rbf_kernel,
            shape_parameter = rbf_shape_parameter(Δ),
            fully_linear = fully_linear,
            polynomial_degree = rbf_poly_deg,
            tdata = new_tdata
        );

        m.model_info.round1_indices = round1_indices;
        m.model_info.round2_indices = additional_site_indices;
        m.model_info.fully_linear = fully_linear;

        # look for EVEN MORE model enhancing points in big radius, ignore points considered already
        if length(training_sites) >= n_vars + 1
            unexplored_indices = setdiff( 1:length(sites_db), model_point_indices );
            more_site_indices, Y3, Π, Q, R, Z3, L = additional_points!( m, x, n_exp, sites_db[ unexplored_indices ], values_db[ unexplored_indices ], Δ_2, θ_pivot_cholesky, max_model_points ); # sites are added to m in-place
            @info("\tFound $(length(more_site_indices)) sites to further enhance the model.")

            n_points = length(m.training_sites);
            n_still_missing = Int64(min( max_model_points - n_points, max_evals - length(sites_db) - 1 ));
            if n_still_missing > 0 && use_max_points == true
                @info("\tActively sampling to the maximum number of points ($n_still_missing additional evaluations)!")
                lb_eff, ub_eff = effective_bounds_vectors(x, Δ_2);
                new_sites = monte_carlo_th( max_model_points, lb_eff, ub_eff; seeds = m.training_sites )[ n_points + 1 : end ];
                new_values = eval_all_objectives.( problem, new_sites );

                push!(sites_db, new_sites...);
                push!(m.training_sites, new_sites...);
                push!(values_db, new_values...);
                push!(m.training_values, [ v[1:n_exp] for v in new_values]...)
                m.model_info.round3_indices = [more_site_indices; length(sites_db) - n_still_missing : length(sites_db)]
                train!(m)
            else
                m.model_info.round3_indices = more_site_indices;
                # finally: train!
                train!(m, Π, Q, R, Z3, L);
            end
        else
            train!(m)   # look for least squares solution of underdetermined model
        end
        @info("\tModel$(fully_linear ? " " : " not ")linear!")

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
function make_linear!(m :: RBFModel, config_struct, crit :: Val{true}, constrained_flag = false, )
    @unpack  θ_enlarge_1, θ_enlarge_2, Δ_max, rbf_shape_parameter = config_struct;

    @unpack iter_data = config_struct;
    @unpack x, f_x, Δ, sites_db, values_db = iter_data;

    Δ_1 = Δ * θ_enlarge_1;
    Δ_2 = Δ_max * θ_enlarge_2;
    allowed_flags_Y = norm.( eachcol(m.tdata.Y), Inf ) .<= Δ_1    # are the 'linearizing' sites within Δ_1?
    #@show Δ_1

    if !all(allowed_flags_Y)
        new_model = build_model( config_struct, constrained_flag, true );   # build a new fully linear model for smaller trust region
        as_second!(m, new_model);                                           # modify old model to equal new model
        return true  # tell main loop that model has changed
    else
        return false    # tell main loop that model has not changed and descent need not be recomputed
    end
end


@doc "Perform several improvement steps on 'm::RBFModel' using the 'improve!' method until the model is fully linear on θ_enlarge_1*Δ."
function make_linear!(m::RBFModel, config_struct, constrained_flag = false )
    @unpack max_evals = config_struct;

    @unpack iter_data = config_struct;
    @unpack x, f_x, Δ, sites_db, values_db = iter_data;

    evals_left = max_evals - length(sites_db) - 1;
    n_improvement_steps = Int64(min(size(m.tdata.Z,2), evals_left));

    # number of columns of Z is the number of missing sites for full linearity
    # perform as many improvement steps and modify data in place
    for i = 1 : n_improvement_steps
        improvement_flag = improve!(m, config_struct, constrained_flag );
        if !improvement_flag
            break;
        end
    end
    return n_improvement_steps
end


@doc "Perform ONE improvement step along model improving direction (first column of Z, where Z is an orthogonal to the site matrix Y).

The model parameters 'Y' and 'z' and its fully_linear flag are potentially modified.
Returns the newly sampled site and its value vector as given by f."
function improve!( m::RBFModel, config_struct :: AlgoConfig, constrained_flag = false )
    @unpack n_vars, n_exp, θ_pivot, θ_enlarge_1 = config_struct;

    @unpack problem = config_struct;

    @unpack iter_data = config_struct;
    @unpack x, f_x, Δ, sites_db, values_db = iter_data;

    # Y and Z matrix are already associated with model
    @unpack Y, Z = m.tdata;

    if size(Z,2) > 0  # should always be true during optimization routine, but usefull for make_linear!
        @info("\t\tSampling with radius $(θ_enlarge_1 * Δ).")

        # add new site from model improving direction
        z = Z[:,1];

        new_site = sample_new(constrained_flag, x, θ_enlarge_1 * Δ, z, sqrt(n_vars) * Δ * θ_pivot,  rand([-1.0, 1.0]) * θ_enlarge_1 * Δ )
        if isempty( new_site )
            new_model = rebuild_model( config_struct )
            as_second!(m, new_model);
            return false        # to tell 'make_linear!' that improvement was not successful
        end

        # update Y matrix
        Y = hcat( Y, new_site - x );    # add new column to matrix Y

        Z = Z[:,2:end];

        # save updated matrices to TrainingData object
        @pack! m.tdata = Y, Z;

        # update rbf model
        new_val = eval_all_objectives(problem, new_site );
        if isempty(size(new_val))
            new_val = [ new_val, ];
        end
        push!(m.training_sites, new_site);
        push!(m.training_values, new_val[1:n_exp]);
        push!(sites_db, new_site);
        push!(iter_data, new_val);

        if size(Z,2) == 0
            m.fully_linear = true;
            @info("\t\tModel is now fully linear.")
        else
            m.fully_linear = false;
            @info("\t\tModel is not fully linear and there is an improving direction.")
        end

        # update info
        m.model_info.fully_linear = m.fully_linear;
        push!(m.model_info.round1_indices, length(sites_db));

        train!(m);
        return true

    else
        @warn "empty return from improvement!"
        return false
    end
end
