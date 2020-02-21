using Documenter, FixedPointSinCosApproximations

makedocs(;
    modules=[FixedPointSinCosApproximations],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/zsoerenm/FixedPointSinCosApproximations.jl/blob/{commit}{path}#L{line}",
    sitename="FixedPointSinCosApproximations.jl",
    authors="Soeren Zorn",
    assets=String[],
)

deploydocs(;
    repo="github.com/zsoerenm/FixedPointSinCosApproximations.jl",
)
