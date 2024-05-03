__precompile__()
module SphereUDE

# types
using Base: @kwdef
# utils 
# training 
using LinearAlgebra, Statistics, Distributions
using FastGaussQuadrature
using Lux, Zygote, DiffEqFlux
using ChainRules: @ignore_derivatives
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL, OptimizationPolyalgorithms
using ComponentArrays
using PyPlot, PyCall
using PrettyTables

# Testing double-differentiation
# using BatchedRoutines

# Debugging
using Infiltrator

include("types.jl")
include("utils.jl")
include("train.jl")
include("plot.jl")

# Python libraries 
const mpl_colors::PyObject = isdefined(SphereUDE, :mpl_colors) ? SphereUDE.mpl_colors : PyNULL()
const mpl_colormap::PyObject = isdefined(SphereUDE, :mpl_colormap) ? SphereUDE.mpl_colormap : PyNULL()
const sns::PyObject = isdefined(SphereUDE, :sns) ? SphereUDE.sns : PyNULL()
const ccrs::PyObject = isdefined(SphereUDE, :ccrs) ? SphereUDE.ccrs : PyNULL()
const feature::PyObject = isdefined(SphereUDE, :feature) ? SphereUDE.feature : PyNULL()

include("setup/config.jl")

end
