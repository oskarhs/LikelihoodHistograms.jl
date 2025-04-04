# The dynamical programming algorithm of Kanazawa (1988)
function dynamic_algorithm(phi::Function, k_max::Int)
    cum_weight = Matrix{Float64}(undef, k_max, k_max)
    ancestor = zeros(Int64, k_max, k_max)
    weight = Matrix{Float64}(undef, k_max+1, k_max+1)

    function optimal_path!(ancestor, cum_weight, k)
        ancestor0 = Array{Int64}(undef, k_max-k+1)
        cum_weight0 = Array{Float64}(undef, k_max-k+1)

        @inbounds for i = k:k_max
            obj = cum_weight[(k-1):(i-1), k-1] .+ weight[k:i, i+1]
            ancestor0[i-k+1] = argmax(obj)
            cum_weight0[i-k+1] = obj[ancestor0[i-k+1]]
        end
        ancestor[k:k_max, k] = ancestor0 .+ (k-2)
        cum_weight[k:k_max, k] = cum_weight0
    end

    # Compute weights for each possible interval
    for i in 1:k_max
        for j in (i+1):(k_max+1)
            @inbounds weight[i, j] = phi(i, j)
        end
    end

    # Compute cumulative weights
    cum_weight[:,1] = weight[1,2:k_max+1]
    for k in 2:k_max
        optimal_path!(ancestor, cum_weight, k)
    end
    optimal = cum_weight[k_max,:] # Get weight function for each partition

    return optimal, ancestor
end

# Compute optimal partition based on the output of the DP algorithm
function compute_bounds(ancestor, grid, k)
    L = [size(ancestor)[1]]
    for i = k:-1:1
        pushfirst!(L, ancestor[L[1],i])
    end
    bounds = grid[L .+ 1]
    return bounds
end

# Φ corresponding to penB of Rozenholc et al. (2010)
function phi_penB(i, j, N_cum, grid)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) # Contribution of the given bin to log-likelihood
    return contrib
end

# Φ corresponding to Bayesian histogram with fixed concentration parameter a (not dep. on k)
function phi_bayes(i, j, N_cum, grid, a, prior_cdf)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i] # Note: p0 = len_bin on the interval 0-1
    a_int = a * (prior_cdf(grid[j]) - prior_cdf(grid[i]))
    contrib = loggamma(a_int + N_bin) - loggamma(a_int) - N_bin * log(len_bin)
    return contrib
end

# Φ corresponding to penR of Rozenholc et al. (2010)
function phi_penR(i, j, N_cum, grid, n)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) - 0.5 * N_bin / (n*len_bin)
    return contrib
end

# Φ corresponding to Kullback-Leibler LOOCV
function phi_KLCV(i, j, N_cum, grid, n; minlength=0.0)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = 0.0
    if len_bin > minlength
        if N_bin >= 2
            contrib = N_bin * log((N_bin-1) / len_bin)
        else
            contrib = -Inf64
        end
    else
        contrib = -Inf64
    end
    return contrib
end

# Φ corresponding to L2 LOOCV
function phi_L2CV(i, j, N_cum, grid, n; minlength=0.0)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    if len_bin > minlength
        contrib = (2.0*N_bin - (n+1)/n^2 * N_bin^2) / len_bin
    else
        contrib = -Inf64
    end
    return contrib
end