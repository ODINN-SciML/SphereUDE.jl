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

export SphereParameters, SphereData 
export train_sphere

include("utils.jl")
include("types.jl")
include("train.jl")

end
