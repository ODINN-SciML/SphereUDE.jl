__precompile__()
module SphereUDE

# utils
# training
using LinearAlgebra, Statistics, Distributions
using Random
using FastGaussQuadrature
using Lux, Zygote, DiffEqFlux
using ChainRules: @ignore_derivatives
import ChainRulesCore
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using SciMLSensitivity, ForwardDiff
using Optimization, OptimizationOptimisers, OptimizationOptimJL, LineSearches
using Optimisers
using ComponentArrays
using Plots
using PrettyTables, Printf
using SciMLBase: NoAD
using DiffEqBase: AbstractDEAlgorithm
using MLUtils: DataLoader
using StaticArrays

# Data Types
include("DiffEq/adjoint.jl")
include("Training/Regularization/Quadrature.jl")
include("Parameters/SphereParameters.jl")
include("Data/SphereData.jl")
include("Data/data_utils.jl")

# Regressor interface — must come before DiffEq which depends on AbstractRegressor
include("Training/ML/Regressor.jl")
include("Training/ML/NN_utils.jl")
include("Training/ML/NNRegressor.jl")
include("Training/ML/SplineRegressor.jl")

# Differential Equation support
include("DiffEq/forward.jl")
include("DiffEq/inverse.jl")

# Training utils
include("Training/ML/numerical_utils.jl")
include("Training/ML/AD.jl")

# Losses and regularization
include("Training/Regularization/quadrature_utils.jl")
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

# Uncertainty quantification
include("Training/Sampling/bootstrap.jl")

# Plotting
include("Plot/plot.jl")


end
