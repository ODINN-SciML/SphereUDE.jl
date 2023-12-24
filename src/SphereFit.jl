module SphereFit

# types
using Base: @kwdef
# utils 
# training 
using LinearAlgebra, Statistics
using Lux 
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays: ComponentVector

export SphereParameters, SphereData, cart2sph
export train_sphere
export Results

include("utils.jl")
include("types.jl")
include("train.jl")

end
