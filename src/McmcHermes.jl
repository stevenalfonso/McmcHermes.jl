module McmcHermes

using ProgressMeter
using Distributions

export one_mcmc, run_mcmc, get_flat_chain, get_gelman_rubin

"""

    one_mcmc(log_prob, data, parameters, n_iter, a)

Returns one chain with the dimension of (n_iter, 1, n_dim)

"""

function one_mcmc(log_prob::Function, data::Vector, parameters::Vector, n_iter::Int64, a::Float64)
    
    x = ones(( n_iter, length(parameters) ))
    x[1,:] = [parameters][1]
    space = zeros(length(parameters))

    for i in 2:n_iter
        
        for j in 1:length(parameters)
            space[j] = x[i-1,j] + a * rand(Uniform(-1, 1), 1)[1]
        end

        present = log_prob(data, parameters)
        future = log_prob(data, space)
        
        alpha = min( exp(future - present), 1.0)
        g = rand()

        if alpha > g

            for k in 1:length(parameters)
                x[i,k] = space[k]
            end

        else
            x[i,:] = x[i-1,:]     
        end
    end

    return x[:,:,:]
end


"""

    run_mcmc(log_prob, data, parameters, n_iter, n_walkers, n_dim, a)

Returns chains with the dimension of (n_iter, n_walkers, n_dim)

"""

function run_mcmc(log_prob::Function, data::Vector, parameters::Vector, n_iter::Int64, n_walkers::Int64, 
    n_dim::Int64; a::Float64=0.1)

    mcmc_chains = ones(( n_iter, n_dim, n_walkers ))
    seed = rand(n_walkers, n_dim) * 1e-2 .+ transpose(parameters)

    @showprogress "Running chains..." for walkers in 1:n_walkers
        one_chain = one_mcmc(log_prob, data, seed[walkers,:], n_iter, a)
        mcmc_chains[:,:,walkers] = one_chain
        sleep(0.1)
    end

    chains = permutedims(mcmc_chains, [1, 3, 2])

    return chains
end


"""

    get_flat_chain(array, burn_in, thin) 

Returns the stored chain of MCMC samples.

"""

function get_flat_chain(array::Array; burn_in::Int=1, thin::Int=1)
    
    l = ones(( size([i for i in 1:size(array)[1] if i%thin==0])[1], size(array)[2], size(array)[3] ))
    ind = 0
    
    for (index, value) in enumerate(1:size(array)[1])
        
        if mod(index, thin) == 0
            ind += 1
            l[ind,:,:] = array[index,:,:]
        end
    end
    
    l_thin = reshape( l, (size(l)[1] * size(l)[2], size(l)[3] ) )
    
    return l_thin[burn_in:end,:]
end


"""

    get_gelman_rubin(chains)

Get the Gelman Rubin convergence diagnostic of the chains.
Returns the Gelman-Rubin number.

"""

function get_gelman_rubin(chains)
    s_j = var(chains, dims=2)
    W = mean(s_j)
    theta = mean(chains, dims=2)
    theta_j = mean(theta, dims=1)   
    M, N = size(chains)[1], size(chains)[2]
    B = N / (M - 1) * sum((theta_j .- theta).^2)
    var_theta = (N - 1) / N * W + B / N
    R_hat = sqrt(var_theta / W)
    return R_hat
end

end # module McmcHermes
