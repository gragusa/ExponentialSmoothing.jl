using Documenter, ExponentialSmoothing

makedocs(
    modules = [ExponentialSmoothing],
    format = :html,
    sitename = "ExponentialSmoothing.jl",
    pages = Any["index.md"]
)

deploydocs(
    repo = "github.com/gragusa/ExponentialSmoothing.jl.git",
    target = "build",
    julia = "1.0",
    deps = nothing,
    make = nothing,
)
