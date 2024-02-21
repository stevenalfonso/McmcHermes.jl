[![Runtests](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/Runtests.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/Runtests.yml)
[![Documentation](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/documentation.yml)
[![CI](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/CI.yml)

# McmcHermes.jl


*A documentation for the McmcHermes package.*


McmcHermes provides a simple but efficient way to generate [Markov Chain Monte-Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) algorithms in order to sample probability density distributions.


## Overview

The major functions in this module are:

run_mcmc: run multiple chains with a specific number of walkers.

get\_flat\_chain: get the stored chain of MCMC samples.

gelman\_rubin\_diagnostic: get the Gelman Rubin convergence diagnostic of the chains. 



## Installation

```julia
using Pkg
Pkg.add("McmcHermes")
```

## Example

### Run mcmc

First, let's generate some data:

```julia
using Distributions, LaTeXStrings, Plots

mu, sigma = 5, 2
l_b, u_b = 0, 10
d = Truncated(Normal(mu, sigma), l_b, u_b)
N = 10000
data = rand(d, N)

histogram(data, legend=false, size=(300,300), xlabel="data", show=true)
```

![data](./assets/hist.png)

Then, we define the Gaussian likelihood and an uniform prior:

```julia
function log_likelihood(X::Vector, parameters::Vector)
    mu, sigma = parameters[1], parameters[2]
    y = 1 ./ (sqrt(2 * pi) .* sigma) .* exp.( -0.5 * ((X .- mu)./sigma).^2 )
    return sum(log.(y))
end

function log_prior(parameters::Vector)
    mu, sigma = parameters[1], parameters[2]
    if 1.0 < mu < 10.0 && 0.0 < sigma < 4.0
        return 0.0
    end
    return -Inf
end

function log_probability(X::Vector, parameters::Vector)
    lp = log_prior(parameters)
    if !isfinite(lp)
        return -Inf
    end
    return lp + log_likelihood(X, parameters)
end
```

The parameters are $$\mu$$ and $$\sigma$$. To run mcmc and sample the posterior probability function we call the McmcHermes packages. Let's define the number of walkers and iterations

```julia
using McmcHermes

mu, sigma = 5, 2
initparams = Vector{Float64}([mu, sigma])

n_iter, n_walkers = 10000, 30
n_dim = 2
seed = rand(n_walkers, n_dim) * 1e-4 .+ transpose(initparams)

chains = McmcHermes.run_mcmc(log_probability, data, seed, n_iter, n_walkers, n_dim, a=0.01)
println(size(chains))
```
(10000, 30, 2)

Gelman-Rubin's diagnostic of chains can be obtained calling the gelman\_rubin\_diagnostic method.
```julia 
println("Gelman Rubin Diagnostic: ", McmcHermes.gelman_rubin_diagnostic(chains))
```
Gelman Rubin Diagnostic: 1.0231750328396079

Plot the chains

```julia
labels = Tuple([L"\mu", L"\sigma"])
x = 1:size(chains)[1]
p = []
for ind in 1:n_dim
    push!(p, plot(x, [chains[:,i,ind] for i in 1:size(chains)[2]], legend=false, lc=:black, lw=1, ylabel=labels[ind], alpha=0.2, xticks=true))
end

plot(p[1], p[2], layout=(2,1), xlabel="iterations", tickfontsize=5, xguidefontsize=8)
plot!(size=(600,200), xlims=(0, size(chains)[1]), show=true, lw=1)
```
![chains](./assets/chains.png)

Chains can also be plotted in a corner. To do so, first get the flat chain

```julia
flat_chains = McmcHermes.get_flat_chain(chains, burn_in=100, thin=10)

flat = DataFrame(flat_chains, :auto)
colnames = ["mu", "sigma"]
flat = rename!(flat, Symbol.(colnames))

using PairPlots, CairoMakie
pairplot(flat)
```
![corner](./assets/corner.png)

### Get samples from a distribution (New)


See here the [documentation](https://stevenalfonso.github.io/McmcHermes.jl/dev/).