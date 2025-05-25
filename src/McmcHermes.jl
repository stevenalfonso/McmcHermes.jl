module McmcHermes

using Distributions
using ProgressMeter

export one_mcmc, run_mcmc, get_flat_chain, gelman_rubin_diagnostic, sampler


function one_mcmc(log_prob::Function, 
    data::Vector, 
    parameters::Vector, 
    n_iter::Int64, 
    a::Float64)

    """

    one_mcmc(log_prob, data, parameters, n_iter, a)
    Returns one chain with dimension of (n_iter, 1, n_dim)

    """

    x = ones(( n_iter, length(parameters) ))
    x[1,:] = [parameters][1]
    new_parameters = zeros(length(parameters))

    for i in 2:n_iter

        # Propose new parameters
        for j in 1:length(parameters)
            new_parameters[j] = x[i-1,j] + a * randn(1)[1]
        end

        present = log_prob(data, parameters)
        future = log_prob(data, new_parameters)

        # Accept or reject the proposed parameters
        alpha = min( exp(future - present), 1.0) 

        if rand() < alpha
            for k in 1:length(parameters)
                x[i,k] = new_parameters[k]
            end
        else
            x[i,:] = x[i-1,:]   
        end

    end

    return x[:,:,:]
end



function run_mcmc(log_prob::Function, 
    data::Vector, 
    seed::Matrix, 
    n_iter::Int64, 
    n_walkers::Int64, 
    n_dim::Int64; 
    a::Float64=0.1)

    """

    run_mcmc(log_prob, data, seed, n_iter, n_walkers, n_dim, a)
    Returns chains with dimension of (n_iter, n_walkers, n_dim)

    """

    mcmc_chains = ones(( n_iter, n_dim, n_walkers ))

    Threads.@threads for walkers in ProgressBar(1:n_walkers)
        one_chain = one_mcmc(log_prob, data, seed[walkers,:], n_iter, a)
        mcmc_chains[:,:,walkers] = one_chain
    end

    # @showprogress "Running chains..." for walkers in 1:n_walkers
    #     one_chain = one_mcmc(log_prob, data, seed[walkers,:], n_iter, a)
    #     mcmc_chains[:,:,walkers] = one_chain
    # end

    # start_time = time_ns()

    # Threads.@threads for walkers in 1:n_walkers
    #     one_chain = one_mcmc(log_prob, data, seed[walkers,:], n_iter, a)
    #     mcmc_chains[:,:,walkers] = one_chain
    # end

    # elapsed_time = (time_ns() - start_time) / 1_000_000_000  # Convert to seconds

    # if elapsed_time < 60
    #     println("Time elapsed running mcmc: $(elapsed_time) seconds.")
    # elseif (elapsed_time >= 60) && (elapsed_time < 3600)
    #     println("Time elapsed running mcmc: $(elapsed_time / 60) minutes.")
    # else
    #     println("Time elapsed running mcmc: $(elapsed_time / 3600) hours.")
    # end

    chains = permutedims(mcmc_chains, [1, 3, 2])

    return chains

end


function get_flat_chain(array::Array; 
    burn_in::Int=1, 
    thin::Int=1)

    """

    get_flat_chain(array, burn_in, thin) 
    Returns the stored chain of MCMC samples.

    """
    
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


function gelman_rubin_diagnostic(chains)

    """

    gelman_rubin_diagnostic(chains)
    Get the Gelman Rubin convergence diagnostic of the chains.
    Returns the Gelman-Rubin number.

    """

    s_j = var(chains, dims=2) # within chain variance
    W = mean(s_j)

    theta = mean(chains, dims=2) # chain mean
    theta_j = mean(theta, dims=1) # grand mean
    M, N = size(chains)[1], size(chains)[2]
    B = N / (M - 1) * sum((theta_j .- theta).^2) # between chain variance
    
    var_theta = (N - 1) / N * W + B / N
    R_hat = sqrt(var_theta / W) # Gelma-Rubin statistic
    
    return R_hat
    
end



function sampler(pdf::Function, 
    n_samples::Number, 
    interval::Vector, 
    params::Vector)

    """
    sampler(pdf, n_samples, intervals, params)
    Get n_samples samples from a 1D Probability Density Distribution
    given some parameters. The interval works for the range of samples.

    """

    #states = []
    states = ones(n_samples)
    current = rand(Uniform(interval[1], interval[2]), 1)[1]

    @showprogress "Sampling..." for i in 1:n_samples
        
        #push!(states, current)
        states[i] = current
        movement = rand(Uniform(interval[1], interval[2]), 1)[1]

        current_prob = pdf(current, params)
        movement_prob = pdf(movement, params)
        
        alpha = min(movement_prob / current_prob, 1.0)
        
        if rand() < alpha
            current = movement
        end
    end
            
    return states

end


end # module McmcHermes
