# McmcHermes.jl


*A documentation for the McmcHermes package.*


McmcHermes provides a simple but efficient way to generate [Markov Chain Monte-Carlo](https://en.wikipedia.org/wiki/Markov_chain_Monte_Carlo) algorithms in order to sample a probability density distribution.


## Overview

The major functions in this module are:

run_mcmc: run multiple chains with a specific number of walkers.

get\_flat\_chain: get the stored chain of MCMC samples.

get\_gelman\_rubin: get the Gelman Rubin convergence diagnostic of the chains. 



## Pkg Registry

```julia
using Pkg
Pkg.add("McmcHermes")
```

See here the [documentation](https://stevenalfonso.github.io/McmcHermes.jl/dev/).


*Develop by [Steven Alfonso](https://github.com/stevenalfonso).*