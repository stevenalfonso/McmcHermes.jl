# push!(LOAD_PATH,"../src/")

using Documenter
using McmcHermes

import Pkg; Pkg.add("Distributions")
import Pkg; Pkg.add("Plots")
import Pkg; Pkg.add("LaTeXStrings")
import Pkg; Pkg.add("DataFrames")
import Pkg; Pkg.add("ProgressMeter")
import Pkg; Pkg.add("PairPlots")
import Pkg; Pkg.add("CairoMakie")
#import Pkg; Pkg.add("OpenCV")

using Distributions, Plots, LaTeXStrings, DataFrames, ProgressMeter
using PairPlots, CairoMakie
#using OpenCV

makedocs(
    sitename="McmcHermes",
    format=Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true"),
    modules=[McmcHermes]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
deploydocs(;
    repo="github.com/stevenalfonso/McmcHermes.jl.git",
)#


