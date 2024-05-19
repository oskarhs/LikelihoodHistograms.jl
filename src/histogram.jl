using Distributions, Random

include("irregular_histogram.jl")
include("regular_histogram.jl")

function histogram(x::AbstractArray{Real}; type::String="regular", rule::String="br", a::Real=1.0)
    if !(type in ["regular", "irregular"])
        type = "regular"
    end

    if type == "regular"
        h, criterion = histogram_regular(x; rule=rule)
    elseif type == "irregular"
        h, criterion = histogram_irregular(x; rule=rule, a=a)
    end
    return h, criterion
end
