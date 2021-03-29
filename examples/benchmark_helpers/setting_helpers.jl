# This file contains utilities to facilities the use of 
# DataFrames.jl for generating benchmark settings
# The idea is to benchmark different "features" and obtain
# "measurements". Dependent features are supported.

using FileIO, DataFrames;

@doc """
    generate_all_settings( features_dict, dependent_features =  nothing )

    Return a DataFrame, the columns of which correspond to every possible 
    combination of feature values. 

    ## Arguments
    
    * `features_dict` is a Dict or NamedTuple. The keys must be Strings or 
      Symbols and provide a feature name. This is also the column name in the
      returned DataFrame. The values must be Vectors of possible feature values.
      E.g. if feature `"x"` can take values `[1,2,3]` provide an entry 
      `"x" => [1,2,3]`.
    * `dependent_features` (optional). A Vector of Dicts or NamedTuples, one
      for each **dependent** feature. A dependent feature depends on some features
      from `features_dict`; we call these **arguments**.
      Each `dependent_feature` is defined by an entry in `dependent_features` with 
      keys `"name"`, `"depends_on"` and `"values"`.
      The value of `"values"` must either be a function that takes as keyword arguments 
      the arguments and returns a vector of possible values.\n 
      It can also be a Dict/NamedTuple with keys being Tuples corresponding to all 
      possible value combinations of features in `depends_on` (in the order given there).

    ## Example  

    ```jldoctest
    features = Dict( 
        :X => [1,],
        :Y => [1,2,]
    );
    dep_features = [
        Dict(
            "name" => :Z,
            "depends_on" => [:X, :Y],
            "values" => function(;X,Y) fill( X+Y, X+Y ) end
        ),
    ];
    s = generate_all_settings(features, dep_features)

    # output
    
    5×3 DataFrame
     Row │ Y      X      Z     
         │ Int64  Int64  Int64 
    ─────┼─────────────────────
       1 │     1      1      2
       2 │     1      1      2
       3 │     2      1      3
       4 │     2      1      3
       5 │     2      1      3
    ```
"""
function generate_all_settings( features_dict :: Union{Dict, NamedTuple}, 
        dependent_features :: Union{Nothing,Vector{D}} where D<:Union{Dict, NamedTuple} = nothing
    )

    # turn entries of `features_dict` into Dict with key=feature name and 
    # value = single column DataFrame of values
    simple_dfs = Dict{Union{String},DataFrame}()
    for (feature, feature_values) ∈ pairs(features_dict)
        simple_dfs[string(feature)] = DataFrame( feature => feature_values )         
    end
    
    # combine to first settings DataFrame
    # rows = all possible combinations of feature values from `features_dict`
    settings_df = crossjoin( (simple_dfs[string(k)] for k in keys(features_dict))... ) 
    
    if !isnothing( dependent_features )
        for dependent_feature ∈ dependent_features
            # for each possible value combination of feature values of 
            # features in `dependent_feature["depends_on"]` put a dataframe 
            # into `dependent_dfs`; these are later stacked vertically
            dependent_dfs = DataFrame[];
            if length( dependent_feature["depends_on"] ) > 1
                input_args = crossjoin( 
                    (simple_dfs[ string(dependency) ] 
                        for dependency ∈ dependent_feature["depends_on"] )... 
                );
            else
                input_args = simple_dfs[ string(dependent_feature["depends_on"][1]) ]
            end


            for arg_row = eachrow( input_args ) 
                # a 1-col DF with colname=dependent_feature["name"] and values=f(;arg_row...)
                row_df = DataFrame( 
                    dependent_feature["name"] => let dp_vals = dependent_feature["values"];
                        if isa(dp_vals, Function) 
                            dp_vals(; arg_row... )
                        elseif isa( dp_vals, Union{Dict,NamedTuple} )
                            dp_vals[ Tuple(arg_row) ];
                        else
                            error( 
                                "Field `values` of dependent feature $(dependent_features["name"])
                                must be a function with kwargs or a Dict/NamedTuple with keys corresponding
                                to all possible value combinations of $(dependent_feature["depends_on"])."
                            );
                        end#if 
                    end#let             
                );
                # add redundant columns containing input_args information 
                # and colnames for joining with settings_df
                push!( dependent_dfs, crossjoin( DataFrame(arg_row), row_df ) )
            end
            
            settings_df = innerjoin( settings_df, vcat( dependent_dfs... ); 
                on = dependent_feature["depends_on"] 
            );
        end
    end
    return settings_df;
end

#=
features = Dict( 
    :X => [1,],
    :Y => [1,2,]
);
dep_features = [
    Dict(
        "name" => :Z,
        "depends_on" => [:X, :Y],
        "values" => function(;X,Y) fill( X+Y, X+Y ) end
    ),
];
s = generate_all_settings(features, dep_features)
=#

function feature_names( features_dict :: Union{Dict, NamedTuple}, 
        dependent_features :: Union{Nothing,Vector{D}} where D<:Union{Dict, NamedTuple} = nothing
    )
    return [
        string.(collect( keys( features_dict ) ));
        [ string(dep_feature["name"]) for dep_feature ∈ dependent_features ]
    ]
end

#%%
function unpack_feature( feature_name :: String, some_dict )
    feature_symbol = Symbol( feature_name );
    feature_symbol_symbol = :(Symbol( $feature_name ));
    @eval $feature_symbol = getindex($some_dict, $feature_symbol_symbol);
end

#%%
@doc """
    add_observation_columns!(df, observations ::Union{Dict, NamedTuple} )
   
Add one or several new empty column(s) to DataFrame `df` or overwrite if exists.
Columns are specified by `observations` where each key gives a column name and
each value the corresponding data type. 
Each column will have then have this data type in union with `Nothing`.
"""
function add_observation_columns!(df :: DataFrame, observations ::Union{Dict, NamedTuple} )    
    n_rows = size(df, 1);
    for (obs_name, obs_type) ∈ pairs(observations)
        df[!, string(obs_name)] = Vector{Union{obs_type,Missings.Missing}}(Missings.missing, n_rows)
    end
    nothing
end

#=
observations = (;
    :ω => Float64,
)
add_observation_columns!( s, observations );
=#

#%%
function load_previous_results( filename; result_key = "results" )
    file_data = load( filename );

    if !haskey( file_data, result_key )
        error("Cannot retrieve previous data.");
    end

    return file_data[result_key];
end

function scan_feature_values( df :: DataFrame, feature_name :: Union{Symbol, String})
    if feature_name in names(df)
        return unique( df[ !, feature_name ] );
    else
        @warn "No column $(feature_name) found in DataFrame."
        return []
    end
end

#%%
function scan_dependent_feature_values( df :: DataFrame, feature_name :: Union{Symbol, String},
        depends_on :: Union{S, Vector{S}} where S<:Union{Symbol, String} 
    )
    
    feature_name = string(feature_name);
    
    if feature_name in names(df)
        dep_args = isa( depends_on, Union{Symbol, String} ) ? [ string( depends_on ), ] : string.(depends_on);
        new_dep_dict = Dict()
        for sub_df in groupby( df, dep_args )
            new_dep_dict[ Tuple( sub_df[1, dep_args]) ] = unique( sub_df[:, feature_name ] );
        end
        return new_dep_dict
    else
        @warn "No column $(feature_name) found in DataFrame."
        return []
    end
end

function scan_dependent_feature_values(df :: DataFrame, feature :: Union{Dict, NamedTuple})
    return scan_dependent_feature_values( feature["name"], feature["depends_on"] )    
end

#=
z_vals = scan_dependent_feature_values( s, "Z", ["X","Y"])
=# 

#%%
function find_observation_rows( df :: DataFrame,
        observations :: Vector{<:Symbol} = Symbol[]
    )
    return findall( 
        .!(vec( 
            prod( hcat( ( ismissing.( df[:,obs] ) for obs in observations)... ), dims = 2 ) 
        ))
    );
end

find_observation_rows(df :: DataFrame, obs :: Union{Dict, NamedTuple}) = find_observation_rows( df, collect(keys(obs)) )

function fill_from_partial_results!(target :: DataFrame, source :: DataFrame, 
        feature_names :: Vector{ S } where S<:Union{Symbol, String}, 
        observations :: Union{Nothing, Union{Dict, NamedTuple} } = nothing,
        required_observations :: Union{Nothing, Union{Dict, NamedTuple} } = nothing;
    )
    
    if isnothing( observations )
        observations = intersect( setdiff( names(target), string.(feature_names) ), names( source ) )
    else
        observations = collect(keys(observations));
    end

    if !isnothing( observations )
        if isnothing( required_observations )
            required_observations = observations;
        end
        investigated_row_indices = Int[];
        
        non_missing_indices = find_observation_rows( source, required_observations );

        for non_missing_index in non_missing_indices
            src_row = source[ non_missing_index, : ]; 
            for target_row ∈ eachrow( target )
                if Tuple(target_row[feature_names]) == Tuple(src_row[feature_names]) 
                    target_row[observations] = src_row[observations];
                    push!( investigated_row_indices, rownumber(target_row) )
                    # break here if we expect/assume unique features
                end
            end
        end
        to_do_row_indices = setdiff( 1 : size(target,1), investigated_row_indices )
        println("Filled in data from $(length(investigated_row_indices)) rows.")
        return investigated_row_indices, to_do_row_indices
    end
    
    return nothing
end

function fill_from_partial_results!( target :: DataFrame, source :: DataFrame, 
        features_dict :: Union{Dict, NamedTuple}, 
        dependent_features :: Union{Nothing,Vector{D}} where D<:Union{Dict, NamedTuple} = nothing,
        observations :: Union{Nothing, Union{Dict, NamedTuple} } = nothing
    )
    feature_names = collect(keys( features_dict ));
    if !isnothing( dependent_features )
        push!( feature_names, [dep_feat["name"] for dep_feat ∈ dependent_features]... )
    end
    return fill_from_partial_results!( target, source, feature_names, observations)
end
#=
new_df = DataFrame( X=[1,1,2,2], Y=[1,2,1,2] )
add_observation_columns!( new_df, (; :Z => Int,))
src = DataFrame( X = [1,2], Y= [2,2], Z = [10,11])
fill_from_partial_results!( new_df, src, ["X", "Y"] )
=#


