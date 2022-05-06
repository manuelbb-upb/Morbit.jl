include("make_literate.jl")

using Pkg
current_env = Base.load_path()[1]
Pkg.activate(@__DIR__)

using Morbit
using Documenter

using PlutoStaticHTML

const NOTEBOOK_DIR = joinpath(@__DIR__, "src", "notebooks")

"""
    build_notebooks()
Run all Pluto notebooks (".jl" files) in `NOTEBOOK_DIR`.
"""
function build_my_notebooks()
    println("Building notebooks")
    hopts = HTMLOptions(; append_build_context=false)
    output_format = documenter_output
    bopts = BuildOptions(NOTEBOOK_DIR; output_format, previous_dir = NOTEBOOK_DIR)
    build_notebooks(bopts, 
        ["notebook_finite_differences.jl", "notebook_polynomial_interpolation.jl"], hopts
    )
    return nothing
end

if !("DISABLE_NOTEBOOK_BUILD" in keys(ENV))
    build_my_notebooks()
end

DocMeta.setdocmeta!(Morbit, :DocTestSetup, :(using Morbit); recursive=true)

makedocs(;
    modules=[Morbit],
    authors="Manuel Berkemeier",
    repo="https://github.com/manuelbb-upb/Morbit.jl/blob/{commit}{path}#{line}",
    sitename="Morbit.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://manuelbb-upb.github.io/Morbit.jl",
        mathengine = MathJax3()
        #assets=["/custom_assets/iframeResizer.min.js",],
    ),
    pages=[
        "Home" => "index.md",
        "Quick Start" => "quickstart.md",
        "Examples" => [
            "Two Parabolas" => "example_two_parabolas.md",
            "Constraints" => "constraints.md",
            "Composite Functions" => "composites.md",
            #"ZDT3" => "example_zdt.md"
        ],
        "Models" => [
            "ExactModels" => "ExactModel.md",
            "RbfModels" => "RbfModel.md",
            "TaylorModels" => "TaylorModel.md",
            "LagrangeModels" => "LagrangeModel.md"
        ],
        "Random Notebooks" => [
            "Finite Differences" => "notebooks/notebook_finite_differences.md",
            "Lagrange Interpolation" => "notebooks/notebook_polynomial_interpolation.md"
        ],
        "Pretty Printing" => "custom_logging.md",
        "Developer" => [
            "DocStrings" => "dev_man.md",
        ]
    ],
)

deploydocs(;
    repo="github.com/manuelbb-upb/Morbit.jl",
    devbranch="nonlinear_structure" # TODO Make this "master" after merging
)
	
Pkg.activate(current_env)