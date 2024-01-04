__precompile__()
module SphereFit

# types
using Base: @kwdef
# utils 
# training 
using LinearAlgebra, Statistics
using Lux, Zygote
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using ComponentArrays: ComponentVector
using PyPlot, PyCall

# Debugging
using Infiltrator

# Do I need to repeat these here if I have them in their respective import files?
export SphereParameters, SphereData, cart2sph
export train
export Results
export plot_sphere

include("utils.jl")
include("types.jl")
include("train.jl")
include("plot.jl")

# Python libraries 
const mpl_colors = PyNULL()
const mpl_colormap = PyNULL()
const sns = PyNULL()
const ccrs = PyNULL()
const feature = PyNULL()

function __init__()
    copy!(mpl_colors, pyimport("matplotlib.colors"))
    copy!(mpl_colormap, pyimport("matplotlib.cm"))
    copy!(sns, pyimport("seaborn"))
    copy!(ccrs, pyimport("cartopy.crs"))
    copy!(feature, pyimport("cartopy.feature"))
end

end
