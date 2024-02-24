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

mu, sigma = 10, 2
l_b, u_b = 0, 20
d = Truncated(Normal(mu, sigma), l_b, u_b)
N = 10000
data = rand(d, N)

histogram(data, legend=false, size=(300,300), xlabel="data", show=true)
```

![data](./docs/src/assets/hist.png)

Then, we define the Gaussian likelihood and an uniform prior:

```julia
function log_likelihood(X::Vector, parameters::Vector)
    mu, sigma = parameters[1], parameters[2]
    y = 1 ./ (sqrt(2 * pi) .* sigma) .* exp.( -0.5 * ((X .- mu)./sigma).^2 )
    return sum(log.(y))
end

function log_prior(parameters::Vector)
    mu, sigma = parameters[1], parameters[2]
    if 5.0 < mu < 15.0 && 0.0 < sigma < 5.0
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

The parameters are $\mu$ and $\sigma$. To run mcmc and sample the posterior probability function we call the McmcHermes packages. Let's define the number of walkers and iterations

```julia
using McmcHermes

mu, sigma = 10, 2
initparams = Vector{Float64}([mu, sigma])

n_iter, n_walkers = 5000, 30
n_dim = 2
seed = rand(n_walkers, n_dim) * 1e-4 .+ transpose(initparams)

chains = McmcHermes.run_mcmc(log_probability, data, seed, n_iter, n_walkers, n_dim, a=0.01)
println(size(chains))
```
(5000, 30, 2)

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
![chains](./docs/src/assets/chains.png)

Chains can also be plotted in a corner. To do so, first get the flat chain

```julia
flat_chains = McmcHermes.get_flat_chain(chains, burn_in=100, thin=10)

using PairPlots, CairoMakie

table = (; x=flat_chains[:,1], y=flat_chains[:,2],)
fig = pairplot(table, labels = Dict(:x => L"\mu", :y => L"\sigma"))
```
![corner](./docs/src/assets/corner.png)

### Get samples from a distribution (New)

If you want to draw samples from the following distribution

$$\frac{1}{\sqrt{2 \pi} \sigma_{1}} e^\left[- \frac{(x - \mu_{1})^{2}}{2 \sigma_{1}^{2}}\right]$$

You would do something like:

```julia
function pdf(X::Number, params::Vector)
    s1, s2, mu1, mu2 = params[1], params[2], params[3], params[4]
    return 1 / (sqrt(2 * pi) * s1) * exp( -0.5*((X - mu1)/s1)^2 ) + 1 / (sqrt(2 * pi) * s2) * exp( -0.5*((X - mu2)/s2)^2 )
end

function gaussian_function(X::Vector, params::Vector)
    x_values = collect(range(minimum(X), maximum(X), length=length(X)))
    s1, s2, mu1, mu2 = params[1], params[2], params[3], params[4]
    return 0.5 ./ (sqrt(2 * pi) .* s1) .* exp.(-0.5*((x_values .- mu1)./s1).^2) .+ 0.5 ./ (sqrt(2 * pi) .* s2) .* exp.(-0.5*((x_values .- mu2)./s2).^2)
end

params = [3, 1.5, -5, 5]
interval = [-20, 20]
sampling = McmcHermes.sampler(pdf, 10000, interval, params)

x_values = Vector{Float64}(range(interval[1], interval[2], 100))

histogram(sampling, xlabel=L"samples", ylabel=L"p(x)", xguidefontsize=12, color=:gray, yguidefontsize=12, normalize=:pdf, show=true, label="samples")
plot!(x_values, gaussian_function(x_values, params), lw=3, size=(500,400), label="Function", lc=:orange, show=true)
```
![samples](./docs/src/assets/samples.png)

See here the [documentation](https://stevenalfonso.github.io/McmcHermes.jl/dev/).