
using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using Literate

example_dir = joinpath(@__DIR__, "..", "examples")
src_dir = joinpath(@__DIR__, "..", "src")

function replace_comments( content )
    content = replace(content, r"^(\h*)#~(.*)$"m => s"\1## \2")
    return content
end

#%%
Literate.markdown(
    joinpath( example_dir, "constraints.jl"), 
    joinpath( @__DIR__, "src" );    
    preprocess = replace_comments,
)
Literate.markdown(
    joinpath( example_dir, "example_two_parabolas.jl"), 
    joinpath( @__DIR__, "src" );    
    preprocess = replace_comments,
#    codefence = "````julia" => "````",  # disables execution; REMOVE when it works again
    )
#=
Literate.markdown(
    joinpath( example_dir, "example_zdt.jl"), 
    joinpath( @__DIR__, "src" );   
    codefence = "````julia" => "````", # disables execution; REMOVE when it works again
    preprocess = replace_comments
    )
=#
Literate.markdown(
    joinpath( src_dir, "models", "TaylorModel.jl"), 
    joinpath( @__DIR__, "src" );    
    codefence = "````julia" => "````",
)

Literate.markdown(
    joinpath( src_dir, "models", "ExactModel.jl"), 
    joinpath( @__DIR__, "src" );    
    codefence = "````julia" => "````",
)

Literate.markdown(
    joinpath( src_dir, "models", "LagrangeModel.jl"), 
    joinpath( @__DIR__, "src" );    
    codefence = "````julia" => "````",
)

Literate.markdown(
    joinpath( src_dir, "models", "RbfModel.jl"), 
    joinpath( @__DIR__, "src" );    
    codefence = "````julia" => "````",
)

Literate.markdown(
    joinpath( src_dir, "custom_logging.jl"), 
    joinpath( @__DIR__, "src" );  
    codefence = "````julia" => "````",
)
#%%
Pkg.activate(current_env)