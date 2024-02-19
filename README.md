[![Runtests](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/Runtests.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/Runtests.yml)
[![Documentation](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/documentation.yml)
[![CI](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/CI.yml)

# McmcHermes.jl


*A documentation for the McmcHermes package.*


McmcHermes provides a simple but efficient way to generate [Markov Chain Monte-Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) algorithms in order to sample a probability density distribution.


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


See here the [documentation](https://stevenalfonso.github.io/McmcHermes.jl/dev/).