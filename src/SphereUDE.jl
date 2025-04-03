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
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using ComponentArrays
# using PyPlot, PyCall
using PythonCall, CondaPkg
using PrettyTables, Printf

# Debugging
using Infiltrator

include("types.jl")
include("utils.jl")
include("losses.jl")
include("diffeq/forward.jl")
include("diffeq/inverse.jl")
include("train.jl")
include("plot.jl")


# We define empty objects for the Python packages
const mpl_base = Ref{Py}()
const mpl_colors = Ref{Py}()
const mpl_colormap = Ref{Py}()
const plt = Ref{Py}()
const sns = Ref{Py}()
const ccrs = Ref{Py}()
const feature = Ref{Py}()

include("setup/config.jl")

end
