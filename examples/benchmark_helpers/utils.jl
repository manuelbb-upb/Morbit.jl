
function parse_commandline()
    @eval begin 
        using ArgParse;
    end
    @eval begin
        s = ArgParseSettings(;suppress_warnings=true);
        @add_arg_table s begin 
            "--outdir"
                help = "specifies directory to store results in; defaults to '~/MORBIT_BENCHMARKS'"
                arg_type = String
                default = joinpath( ENV["HOME"], "MORBIT_BENCHMARKS" )
            "--filename"
                help = """
                name of results file; 
                    possible extensions are 'jld2' and 'csv',
                    if no extension is given, results are stored in both 'jld2' and 'csv'
                """
                arg_type = String
                default = ""
                required = false
            "--runs-per-setting"
                help = "number of random starts per setting."
                default = Threads.nthreads()
                arg_type = Int
            "--resume-from"
                help = "old result file to resume from."
                default = ""
                arg_type = String
            "--save-every"
                help = "save results every X iterations."
                default = 100
                arg_type = Int
            
            # optinal positional arguments if VSCode does something dumb
            "pos1"
                required = false
            "pos2"
                required = false 
            "pos3"
                required = false 
            "pos4"
                required = false 
            "pos5"
                required = false 
        end
        return parse_args(s)
    end
end