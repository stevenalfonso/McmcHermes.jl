using Documenter
using McmcHermes

makedocs(
    sitename = "McmcHermes",
    format = Documenter.HTML(),
    modules = [McmcHermes]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
=deploydocs(
    repo = "https://github.com/stevenalfonso/McmcHermes.jl.git"
)=#


