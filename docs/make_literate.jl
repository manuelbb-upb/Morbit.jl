
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using Literate

function replace_comments( content )
    content = replace(content, r"^(\h*)#~(.*)$"m => s"\1## \2")
    return content
end    

example_dir = joinpath(@__DIR__, "..", "examples")
src_dir = joinpath(@__DIR__, "..", "src")

#%%
Literate.markdown(
    joinpath( example_dir, "example_two_parabolas.jl"), 
    joinpath( @__DIR__, "src" );    
    documenter = true,
    preprocess = replace_comments
    )

Literate.markdown(
    joinpath( example_dir, "example_zdt.jl"), 
    joinpath( @__DIR__, "src" );    
    documenter = true,
    preprocess = replace_comments
    )

#%%
Literate.markdown(
    joinpath( src_dir, "custom_logging.jl"), 
    joinpath( @__DIR__, "src" );    
    documenter = false,
    execute = false,
)
Pkg.activate(current_env)