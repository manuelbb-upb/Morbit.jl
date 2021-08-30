include("make_literate.jl")

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
        #assets=["/custom_assets/iframeResizer.min.js",],
    ),
    pages=[
        "Home" => "index.md",
        "Examples" => [
            "Two Parabolas" => "example_two_parabolas.md",
            "ZDT3" => "example_zdt.md"
        ],
        "Models" => [
            "ExactModels" => "ExactModel.md",
            "RbfModels" => "RbfModel.md",
            "TaylorModels" => "TaylorModel.md",
            "LagrangeModels" => "LagrangeModel.md"
        ],
        "Random Notebooks" => [
            "Finite Differences" => "notebook_finite_differences.md",
            "Lagrange Interpolation" => "notebook_polynomial_interpolation.md"
        ],
        "Pretty Printing" => "custom_logging.md",
        "Internals" => "dev_man.md"
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/Morbit.jl",
    devbranch="stronger_typing" # TODO Make this "master" after merging
)

Pkg.activate(current_env)