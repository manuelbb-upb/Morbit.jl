@doc "Scale a value vector (in place) via min-max-scaling."
function scale!( y :: Vector{T} where{T<:Real}, id :: IterData )
    if id.update_extrema
        y .-= id.min_value
        y ./= ( id.max_value .- id.min_value )
    end
end

function scale(y :: Vector{T} where{T<:Real}, id :: IterData, indices )
    Y = copy(y);
    if length(Y) != length(indices)
        Y = y[indices];
    end
    if id.update_extrema
        Y .-= id.min_value[indices]
        Y ./= (id.max_value[indices] - id.min_value[indices]);
    end
    return Y
end

@doc "Return a scaled value vector via min-max-scaling."
function scale( y :: Vector{T} where{T<:Real}, id :: IterData )
    Y = copy(y);
    if id.update_extrema
        Y .-= id.min_value
        Y ./= ( id.max_value .- id.min_value );
    end
    return Y
end

function scale(y :: Vector{T} where{T<:Real}, id :: IterData, indices :: Array{Int64,1})
    Y = copy(y);
    if id.update_extrema
        if length(Y) != length(indices)
            Y = y[indices];
        end
        Y .-= id.min_value[indices]
        Y ./= (id.max_value[indices] - id.min_value[indices]);
    end
    return Y
end

function scale_grad( g :: Vector{T} where{T<:Real}, id :: IterData, ℓ :: Int64 )
    if id.update_extrema
        g ./= ( id.max_value[ℓ] - id.min_value[ℓ])
    end
    return g
end

function get_data_values( id :: IterData, inds... )
    values = getindex( id.values_db, inds... )
    if id.update_extrema
        # feature_scaling was enabled
        return scale.(values, id )
    end
    return values
end

function get_training_values( config_struct :: AlgoConfig, inds ... )
    return expensive_components.( get_data_values( config_struct.iter_data, inds... ), config_struct )
end

function get_training_values(  config_struct :: AlgoConfig )
    ti = rbf_training_indices(config_struct.iter_data)
    get_training_values( config_struct, ti )
end

@doc "Specialized push! for IterData instances. Appends `args` to field `values_db` and updates `min_value` and `max_value`. Does not perform feature scaling!"
function push!( id :: IterData, new_vals... )
    if !isempty(new_vals)
        push!(id.values_db, new_vals... )
        if id.update_extrema
            if isempty(id.min_value)
                # no minimum vector calculated yet, take all sites and search
                id.min_value = isempty(id.values_db) ? [] : vec(minimum( hcat( id.values_db... ), dims = 2 ));
            else
                # search only new sites and compare to old minimum vector
                new_vals_min = vec(minimum( hcat( new_vals... ), dims = 2 ));
                id.min_value = vec(minimum( hcat( id.min_value, new_vals_min ), dims = 2 ));
            end
            if isempty(id.max_value)
                id.max_value = isempty(id.values_db) ? [] : vec(maximum( hcat( id.values_db... ), dims = 2 ));
            else
                new_vals_max = vec(maximum( hcat( new_vals... ), dims = 2 ));
                id.max_value = vec(maximum( hcat( id.max_value, new_vals_max ), dims = 2 ));
            end
        end
    end
end
