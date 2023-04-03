using McmcHermes
using Test

@testset "McmcHermes.jl" begin
    # Write your tests here.
    using Distributions

    mu, sigma = 10, 2
    l_b, u_b = 0, 20
    d = Truncated(Normal(mu, sigma), l_b, u_b)
    N = 1000
    data = rand(d, N)

    function log_likelihood(X::Vector, parameters::Vector)
        mu, sigma = parameters[1], parameters[2]
        y = 1 ./ (sqrt(2 * pi) .* sigma) .* exp.( -0.5 * ((X .- mu)./sigma).^2 )
        return sum(log.(y))
    end
    
    function log_prior(parameters::Vector)
        mu, sigma = parameters[1], parameters[2]
        if 5.0 < mu < 15.0 && 0.0 < sigma < 4.0
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

    mu, sigma = 10, 2
    initparams = Vector{Float64}([mu, sigma])
    
    n_iter, a = 10000, 0.01
    chain_tests_one = McmcHermes.one_mcmc(log_probability, data, initparams, n_iter, a)
    @test typeof(chain_tests_one) == Array{Float64, 3}
    println(size(chain_tests_one))
    
    n_iter, n_walkers = 100, 50
    n_dim, a = 2, 0.02
    chain_tests = McmcHermes.run_mcmc(log_probability, data, initparams, n_iter, n_walkers, n_dim, a=a)
    @test typeof(chain_tests) == Array{Float64, 3}
    println(size(chain_tests))

    g_r = McmcHermes.get_gelman_rubin(chain_tests)
    @test typeof(g_r) == Float64

    flat_chains = McmcHermes.get_flat_chain(chain_tests, burn_in=10, thin=10)
    @test typeof(flat_chains) == Matrix{Float64}
    println(size(flat_chains))

end