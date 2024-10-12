__precompile__()
module SphereUDE

# utils 
# training 
using LinearAlgebra, Statistics, Distributions
using FastGaussQuadrature
using Lux, Zygote, DiffEqFlux
using ChainRules: @ignore_derivatives
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using SciMLSensitivity, ForwardDiff
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using OptimizationPolyalgorithms, LineSearches
using ComponentArrays
using PyPlot, PyCall
using PrettyTables, Printf

# Debugging
using Infiltrator

include("types.jl")
include("utils.jl")
include("train.jl")
include("plot.jl")

# Python libraries 
const mpl_base::PyObject = isdefined(SphereUDE, :mpl_base) ? SphereUDE.mpl_base : PyNULL()
const mpl_colors::PyObject = isdefined(SphereUDE, :mpl_colors) ? SphereUDE.mpl_colors : PyNULL()
const mpl_colormap::PyObject = isdefined(SphereUDE, :mpl_colormap) ? SphereUDE.mpl_colormap : PyNULL()
const sns::PyObject = isdefined(SphereUDE, :sns) ? SphereUDE.sns : PyNULL()
const ccrs::PyObject = isdefined(SphereUDE, :ccrs) ? SphereUDE.ccrs : PyNULL()
const feature::PyObject = isdefined(SphereUDE, :feature) ? SphereUDE.feature : PyNULL()

include("setup/config.jl")

end
