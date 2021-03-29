function saveplot( file_name, fig )
    file_path = joinpath( splitpath(file_name)[1:end-1]... )
    if !isdir(file_path)
        mkpath( file_path )
    end
    save( file_name, fig )
end

function load_results( res_file; drop_missing = true )
    results_data = load( res_file );
    if drop_missing
        results = dropmissing(results_data["results"]);
    else
        results = results_data["results"]
    end
    results[!, :method] = string.(results[!,:method])
    return results
end
