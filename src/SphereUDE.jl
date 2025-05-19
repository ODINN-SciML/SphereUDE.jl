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
using SciMLBase: NoAD
using MLUtils: DataLoader
using StaticArrays

# Debugging
using Infiltrator

# Data Types
include("DiffEq/adjoint.jl")
include("Parameters/SphereParameters.jl")
include("Data/SphereData.jl")
include("Data/data_utils.jl")

# Differential Equation support
include("DiffEq/forward.jl")
include("DiffEq/inverse.jl")

# Training utils
include("Training/ML/NN_utils.jl")
include("Training/ML/numerical_utils.jl")
include("Training/ML/AD.jl")

# Losses and regularization
include("Training/Regularization/Regularization.jl")
include("Training/Regularization/reg_utils.jl")
include("Training/Regularization/reg_splines_utils.jl")
include("Training/Loss/losses.jl")

# Training
include("Training/train.jl")
include("Training/train_utils.jl")

# Results
include("Results/Results.jl")
include("Results/result_utils.jl")

# Plotting
include("Plot/plot.jl")


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
