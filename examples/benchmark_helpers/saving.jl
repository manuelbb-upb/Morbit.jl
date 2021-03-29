
using Random;
using FileIO, CSV, Dates;

# load body of this script to save with jld file
function script_content( file_path )
    return open( file_path, "r" ) do this_file 
        this_script = read( this_file, String );
    end
end

# define a function to save results
# `args` should be Dict or NamedTuple containing fields
# "outdir" and "filename"
function get_save_function( args; caller_path = @__FILE__, add_timestamp = true )

    this_script = script_content( caller_path );
    fn_rand_str = randstring(8)
    
    save_results = function(results_df; save_csv = true, save_jld = true) 

        if !ispath(args["outdir"])
            try 
                mkpath(args["outdir"])
            catch e
                @error "Could not make output path." exception=(e, catch_backtrace())
            end
        end
        
        if haskey( args, "filename") && args["filename"] != ""
            fn, fext = splitext( args["filename"] )
        else 
            fn = "results_$(fn_rand_str)_$(join(size(results_df),"x"))";
            if add_timestamp
                fn = string( fn, "_", Dates.format(now(), "dd_u_Y__HH_MM_SS"))
            end
            fext = "";
        end
        
        fncsv = (save_csv || lowercase(fext) == ".csv" ) ? string( fn, ".csv" ) : nothing
        fnjld = (save_jld || lowercase(fext) == ".jld2" ) ? string( fn, ".jld2" ) : nothing

        try 
            if !isnothing(fnjld)
                jldpath = joinpath(args["outdir"], fnjld);
                println("Saving jld2 results to $(jldpath).")
                save(jldpath, Dict("results" => results_df, "script" => this_script ));
            end
            if !isnothing(fncsv)
                csvpath = joinpath(args["outdir"], fncsv);
                println("Saving csv results $(csvpath).")
                CSV.write( csvpath, results_df );
            end
        catch e
            @error "Could not save results." exception=(e, catch_backtrace())
        end#try-catch

        return nothing
    end# function

    return save_results
end