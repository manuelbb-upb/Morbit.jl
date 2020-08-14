
if isempty(ARGS)
    OUTFILENAME = joinpath(@__DIR__, "Morbit.so");
else
    OUTFILENAME = ARGS[1];
end

using Pkg
using PackageCompiler

Pkg.activate(joinpath( @__DIR__, ".." ) )

create_sysimage(
	[:Morbit];
	sysimage_path = OUTFILENAME,
	precompile_execution_file = joinpath(@__DIR__,"precompile_morbit.jl")
	)

