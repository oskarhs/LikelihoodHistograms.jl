module LikelihoodHistograms

export histogram_regular, histogram_irregular

include(joinpath("regular", "regular_histogram.jl"))
include(joinpath("irregular", "irregular_histogram.jl"))

end
