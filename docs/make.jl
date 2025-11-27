using PolynomialQTT
using Documenter

DocMeta.setdocmeta!(PolynomialQTT, :DocTestSetup, :(using PolynomialQTT); recursive=true)

makedocs(;
    modules=[PolynomialQTT],
    authors="Hiroshi Shinaoka <h.shinaoka@gmail.com> and contributors",
    sitename="PolynomialQTT.jl",
    format=Documenter.HTML(;
        canonical="https://github.com/tensor4all/PolynomialQTT.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=["Home" => "index.md",
    "Examples" => "examples.md",
    "API Reference" => "apireference.md",]
    
)

deploydocs(; repo="github.com/tensor4all/PolynomialQTT.jl.git", devbranch="main")