# include("make_literate.jl")

using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using Morbit
using Documenter

DocMeta.setdocmeta!(Morbit, :DocTestSetup, :(using Morbit); recursive=true)

makedocs(;
    modules=[Morbit],
    authors="Manuel Berkemeier",
    repo="https://github.com/manuelbb-upb/Morbit.jl/blob/{commit}{path}#{line}",
    sitename="Morbit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://manuelbb-upb.github.io/Morbit.jl",
        assets=String["assets/iframeResizer.min.js"],
    ),
    pages=[
        "Home" => "index.md",
        "Debug Info" => "custom_logging.md",
        "Examples" => [
            "Two Parabolas" => "example_two_parabolas.md",
            "ZDT3" => "example_zdt.md"
        ],
        "Models" => [
            "RbfModels" => "RbfModel.md",
        ],
        "Random Notebooks" => [
            "Finite Differences" => "notebook_finite_differences.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/Morbit.jl",
)

Pkg.activate(current_env)