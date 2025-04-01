module LikelihoodHistograms

export histogram_regular, histogram_irregular

include(joinpath("regular", "histogram_regular.jl"))
include(joinpath("irregular", "histogram_irregular.jl"))

end
