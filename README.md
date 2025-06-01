[![Runtests](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/Runtests.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/Runtests.yml)
[![Documentation](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/documentation.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/documentation.yml)
[![CI](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/stevenalfonso/McmcHermes.jl/actions/workflows/CI.yml)
![GitHub release (latest by date)](https://img.shields.io/github/v/release/stevenalfonso/McmcHermes.jl)
![GitHub Repo stars](https://img.shields.io/github/stars/stevenalfonso/McmcHermes.jl?style=social)
![GitHub forks](https://img.shields.io/github/forks/stevenalfonso/McmcHermes.jl?style=social)
![GitHub issues](https://img.shields.io/github/issues/stevenalfonso/McmcHermes.jl)
![GitHub last commit](https://img.shields.io/github/last-commit/stevenalfonso/McmcHermes.jl)
[![GitHub license](https://img.shields.io/github/license/stevenalfonso/McmcHermes.jl)](https://github.com/stevenalfonso/McmcHermes.jl/blob/master/LICENSE)


# McmcHermes.jl

<div align="center">
<img src="./docs/src/assets/logo.png" alt="logo" width="200"/>
</div>

McmcHermes is a pure-Julia implementation of [Metropolis Hasting Algorithm](https://en.wikipedia.org/wiki/Metropolis–Hastings_algorithm) under an MIT license. McmcHermes will help you if you want to estimate model parameters or sample a probability density distribution.


## Installation

```julia
using Pkg
Pkg.add("McmcHermes")
```

## Development Installation (via Git)

If you want to work with the development version or contribute to the package, you can install it by cloning the repository

```bash
git clone https://github.com/stevenalfonso/McmcHermes.jl.git
cd McmcHermes.jl
julia --project
```

Then, in the Julia REPL

```julia
using Pkg
Pkg.instantiate()
```

Some examples and basic usage are in the [documentation](https://stevenalfonso.github.io/McmcHermes.jl/dev/).