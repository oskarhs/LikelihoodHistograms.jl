using FHist, StatsBase, Plots, Distributions
using SpecialFunctions

function dynamic_algorithm(phi::Function, n::Int, k_max::Int)
    cum_weight = Matrix{Float64}(undef, n, k_max)
    ancestor = Matrix{Float64}(undef, n, k_max)
    weight = Matrix{Float64}(undef, n+1, n+1)

    function optimal_path!(ancestor, cum_weight, k)
        ancestor0 = Array{Int64}(undef, n-k+1)
        cum_weight0 = Array{Float64}(undef, n-k+1)

        for i = k:n
            obj = cum_weight[(k-1):(i-1), k-1] .+ weight[k:i, i+1]
            ancestor0[i-k+1] = argmax(obj)
            cum_weight0[i-k+1] = obj[ancestor0[i-k+1]]
        end
        ancestor[k:n, k] = ancestor0 .+ (k-2)
        cum_weight[k:n, k] = cum_weight0
    end

    # Compute weights for each possible interval
    for i in 1:n
        for j in (i+1):(n+1)
            weight[i, j] = phi(i, j)
        end
    end

    # Compute cumulative weights
    cum_weight[:,1] = weight[1,2:n+1]
    for k in 2:k_max
        optimal_path!(ancestor, cum_weight, k)
    end
    optimal = cum_weight[n,:] # Get weight function for each partition

    return optimal, ancestor
end


function compute_bounds(ancestor, grid, k)
    L = [size(ancestor)[1]]
    for i = k:-1:1
        pushfirst!(L, ancestor[L[1],i])
    end
    bounds = grid[L .+ 1]
    return bounds
end

function phi_penB(i, j, N_cum, grid)
    N_bin = N_cum[j] - N_cum[i]
    len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) # Contribution of the given bin to log-likelihood
    return contrib
end

function phi_bayes(i, j, N_cum, grid)
    N_bin = N_cum[j] - N_cum[i]
    len_bin = grid[j] - grid[i] # Note: p0 = len_bin on the interval 0-1
    contrib = loggamma(1.0*len_bin + N_bin) - loggamma(1.0*len_bin) - N_bin * log(len_bin)
    return contrib
end


function histogram_irregular(x::AbstractArray; rule::String="penB", maxbins::Int=-1, logprior=k->-log(k))
    rule = lowercase(rule)

    xmin = minimum(x)
    xmax = maximum(x)
    z = @. (x - xmin) / (xmax - xmin)
    y = sort(z)
    n = length(y)

    k_max = n

    # Calculate gridpoints (left-open grid, breaks at data points)
    grid = y[1:n-1] .- eps()
    push!(grid, 0.5*(y[n-1]+y[n]))
    push!(grid, y[n]+eps())
    unique!(grid) 

    N = Hist1D(y; binedges=grid).bincounts
    pushfirst!(N, 0)
    N_cum = cumsum(N)

    if rule == "penb"
        phi = (i, j) -> phi_penB(i,j, N_cum, grid)
    elseif rule == "bayes"
        phi = (i, j) -> phi_bayes(i,j, N_cum, grid)
    end

    optimal, ancestor = dynamic_algorithm(phi, n, k_max)
    psi = zeros(k_max)
    if rule == "penb"    
        for k = 1:k_max
            psi[k] = -logabsbinomial(k_max-1, k-1)[1] - k - log(k)^(2.5)
        end
    elseif rule == "bayes"
        for k = 1:k_max
            psi[k] = logprior(k) - logabsbinomial(k_max-1, k-1)[1] + loggamma(1.0) - loggamma(1.0 + n)
        end
    end
    k_opt = argmax(optimal + psi)

    bin_edges_norm = compute_bounds(ancestor, grid, k_opt)
    bin_edges =  xmin .+ (xmax - xmin) * bin_edges_norm
    println(k_opt)
    H = convert(Histogram, Hist1D(x; binedges=bin_edges))
    p0 = bin_edges_norm[2:end] - bin_edges_norm[1:end-1]
    if rule == "bayes"
        H.weights = (H.weights .+ p0) ./ ((n + 1.0)*(bin_edges[2:end] - bin_edges[1:end-1]))
    else
        H.weights = H.weights ./ (n * (bin_edges[2:end] - bin_edges[1:end-1]) )
    end 
    H.isdensity = true
    return H
end


function test()
    y = rand(Normal(), 2*10^3)
    H = histogram_irregular(y; rule="bayes")
    p = plot(H)
    t = LinRange(-3.0, 3.0, 1000)
    plot!(p, t, pdf.(Normal(), t))
    display(p)
end

test()