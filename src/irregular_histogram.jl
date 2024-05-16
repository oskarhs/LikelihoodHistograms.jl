using FHist, StatsBase, Plots, Distributions
using SpecialFunctions

function dynamic_algorithm(phi::Function, n::Int, k_max::Int)
    cum_weight = Matrix{Float64}(undef, n, k_max)
    ancestor = zeros(Int64, n, k_max)
    weight = Matrix{Float64}(undef, n+1, n+1)

    function optimal_path!(ancestor, cum_weight, k)
        ancestor0 = Array{Int64}(undef, n-k+1)
        cum_weight0 = Array{Float64}(undef, n-k+1)

        @inbounds for i = k:n
            obj = cum_weight[(k-1):(i-1), k-1] .+ weight[k:i, i+1]
            ancestor0[i-k+1] = argmax(obj)
            cum_weight0[i-k+1] = obj[ancestor0[i-k+1]]
        end
        #= @inbounds for i = k:n
            ancestor[i, k] = ancestor0[i-k+1] + (k-2)
            cum_weight[i, k] = cum_weight0[i-k+1]
        end =#
        ancestor[k:n, k] = ancestor0 .+ (k-2)
        cum_weight[k:n, k] = cum_weight0
    end

    # Compute weights for each possible interval
    for i in 1:n
        for j in (i+1):(n+1)
            @inbounds weight[i, j] = phi(i, j)
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
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) # Contribution of the given bin to log-likelihood
    return contrib
end

function phi_bayes(i, j, N_cum, grid)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i] # Note: p0 = len_bin on the interval 0-1
    contrib = loggamma(1.0*len_bin + N_bin) - loggamma(1.0*len_bin) - N_bin * log(len_bin)
    return contrib
end

function phi_penR(i, j, N_cum, grid, n)
    @inbounds N_bin = N_cum[j] - N_cum[i]
    @inbounds len_bin = grid[j] - grid[i]
    contrib = N_bin * log(N_bin / len_bin) - 0.5 * N_bin / (n*len_bin)
    return contrib
end

function histogram_irregular(x::AbstractArray; rule::String="penB", right::Bool=true,
                            maxbins::Int=-1, logprior=k->-log(k))
    rule = lowercase(rule)
    if !(rule in ["pena", "penb", "penr", "bayes"])
        rule = "penb" # Set penalty to default
    end 

    xmin = minimum(x)
    xmax = maximum(x)
    z = @. (x - xmin) / (xmax - xmin)
    y = sort(z)
    n = length(y)

    k_max = n

    # Calculate gridpoints (left-open grid, breaks at data points)
    grid = zeros(n+1)
    if right
        grid[1] = y[1] - eps()
        grid[2] = 0.5*(y[1]+y[2])
        grid[3:n+1] = y[2:n] .+ eps()
    else
        grid[1:n-1] = y[1:n-1] .- eps()
        grid[n] = 0.5 * (y[n] - y[n-1])
        grid[n+1] = y[n] + eps()
    end

    N = Hist1D(y; binedges=grid).bincounts
    pushfirst!(N, 0)
    N_cum = cumsum(N)

    if rule in ["pena", "penb"]
        phi = (i,j) -> phi_penB(i, j, N_cum, grid)
    elseif rule == "bayes"
        phi = (i,j) -> phi_bayes(i, j, N_cum, grid)
    elseif rule == "penr"
        phi = (i,j) -> phi_penR(i, j, N_cum, grid, n)
    end

    optimal, ancestor = dynamic_algorithm(phi, n, k_max)
    psi = zeros(k_max)
    if rule == "penb"    
        @inbounds for k = 1:k_max
            psi[k] = -logabsbinomial(k_max-1, k-1)[1] - k - log(k)^(2.5)
        end
    elseif rule == "bayes"
        @inbounds for k = 1:k_max
            psi[k] = logprior(k) - logabsbinomial(k_max-1, k-1)[1] + loggamma(1.0) - loggamma(1.0 + n)
        end
    elseif rule == "penr"
        @inbounds for k = 1:k_max
            psi[k] = -logabsbinomial(k_max-1, k-1)[1] - k - log(k)^(2.5)
        end
    elseif rule == "pena"
        @inbounds for k = 1:k_max
            psi[k] = -logabsbinomial(k_max-1, k-1)[1] - k - 2.0*log(k) -
                    2.0 * sqrt(1.0*0.5*(k-1)*(logabsbinomial(k_max-1, k-1)[1] + 1.0*log(k)))
        end
    end
    k_opt = argmax(optimal + psi)
    criterion_opt = optimal[k_opt] + psi[k_opt]
    println(k_opt)

    bin_edges_norm = compute_bounds(ancestor, grid, k_opt)
    bin_edges =  xmin .+ (xmax - xmin) * bin_edges_norm
    println(k_opt)
    H = convert(Histogram, Hist1D(x; binedges=bin_edges)) # replace this
    p0 = bin_edges_norm[2:end] - bin_edges_norm[1:end-1]
    if rule == "bayes"
        H.weights = (H.weights .+ p0) ./ ((n + 1.0)*(bin_edges[2:end] - bin_edges[1:end-1]))
    else
        H.weights = H.weights ./ (n * (bin_edges[2:end] - bin_edges[1:end-1]) )
    end 
    H.isdensity = true
    return H, criterion_opt
end


function test()
    x = rand(Rocket(), 2*10^3)
    H, criterion_opt = histogram_irregular(x; rule="pena")
    H1, crit = histogram_irregular(x; rule="bayes", logprior=k->0.0)
    #println(H)
    p = plot(H, alpha=0.5)
    plot!(p, H1, alpha=0.5)
    t = LinRange(-3.4, 3.4, 1000)
    plot!(p, t, pdf.(Rocket(), t))
    #histogram!(x, normalize=:pdf, alpha=0.5)
    display(p)
end

test()