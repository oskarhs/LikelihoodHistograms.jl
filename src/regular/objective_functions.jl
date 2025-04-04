import SpecialFunctions.loggamma

# Unnormalized log-posterior with Geometric(τ) prior on k, Dirichlet(ap_0) prior on p
function logposterior_k(N, k, a, n, logprior, prior_cdf)
    logpost = loggamma(a) - loggamma(a + n) + n*log(k) + logprior(k)
    @inbounds for j = 1:k
        a_int = a * (prior_cdf(j/k) - prior_cdf((j-1.0)/k))
        logpost = logpost + loggamma(a_int + N[j]) - loggamma(a_int)
    end
    return logpost
end

function compute_BIC(N, k, n)
    BIC = n*log(k) - 0.5 *k * log(n)
    @inbounds for i in eachindex(N)
        if N[i] > 0
            BIC = BIC + N[i] * log(N[i]/n)
        end
    end
    return BIC
end

# Corresponds to -0.5*AIC
function compute_AIC(N, k, n)
    AIC = n*log(k) - k
    @inbounds for i in eachindex(N)
        if N[i] > 0
            AIC = AIC + N[i] * log(N[i]/n)
        end
    end
    return AIC
end

# Optimization criterion due to Birge and Rozenholc
# Note that the sign is flipped relative to AIC to stay consistent with their original paper
function compute_BR(N, k, n)
    BR = n*log(k) - k - log(k)^(2.5)
    @inbounds for i in eachindex(N)
        if N[i] > 0
            BR = BR + N[i] * log(N[i]/n)
        end
    end
    return BR
end

# Objective (maximization) for regular histograms based on minimum description length (Hall and Hannan)
function compute_MDL(N, k, n)
    is_inf = false
    MDL = -(n-0.5*k)*log(n-0.5*k) + n*log(k) - 0.5*k*log(n)
    @inbounds for i in eachindex(N)
        if N[i] > 0
            MDL = MDL + (N[i]-0.5) * log(N[i]-0.5)
        else 
            is_inf = true
        end
    end
    return return ifselse(is_inf, -Inf, MDL)
end

# Objective (maximization) for regular histograms based on the stochastic complexity criterion ()
function compute_SC(N, k, n)
    SC = n*log(k) + loggamma(k) - loggamma(k+n)
    @inbounds for i in eachindex(N)
        if N[i] > 0
            SC = SC + loggamma(N[i]+1)
        end
    end
    return SC
end

# Objective (maximization) for regular histograms based on Kullback-Leibler Cross Validation
function compute_KLCV(N, k, n)
    is_inf = false
    KLCV = n*log(k)
    @inbounds for i in eachindex(N)
        if N[i] > 1
            KLCV = KLCV + N[i] * log(N[i]-1.0)
        else
            is_inf = true
        end
    end
    return ifelse(is_inf, -Inf, KLCV)
end

# Objective (maximization) for regular histograms based on the l2cv criterion
function compute_L2CV(N, k, n) 
    L2CV = -2.0*k
    @inbounds for i in eachindex(N)
        L2CV = L2CV + k*(n+1)/n^2*sum(N[i]^2)
    end
    return L2CV
end

# Objective (maximization) for regular histograms based on Normalized Maximum Likelihood
function compute_NML(N, k, n)
    NML = n*log(k) + 
          0.5*(k-1.0)*log(0.5*n) + log(sqrt(π)/gamma(0.5*k)) + 
          sqrt(2.0)*k*gamma(0.5*k)/(3.0*sqrt(n)*gamma(0.5*(k-1))) + 
          1.0/n*( ( 3.0 + k*(k-2.0)*(2.0*k+1.0) )/ 36.0 - k^2 * gamma(0.5*k)^2/(9.0*gamma(0.5*(k-1.0))^2) )
    @inbounds for i in eachindex(N)
        NML = NML + N[i]*log(N[i]/n)
    end
    return NML
end