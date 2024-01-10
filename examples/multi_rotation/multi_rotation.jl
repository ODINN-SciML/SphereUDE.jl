using Pkg; Pkg.activate(".")
using Revise 

using LinearAlgebra, Statistics, Distributions 
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using SphereFit

##############################################################
###############  Simulation of Simple Example ################
##############################################################

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

# Total time simulation
tspan = [0, 150.0]
# Number of sample points
N_samples = 100
# Times where we sample points
times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Change point
τ₀ = 55.0
τ₁ = 98.5
# Angular momentum
L0 = ω₀    .* [1.0, 0.0, 0.0]
L1 = 0.5ω₀ .* [0.0, 1/sqrt(2), 1/sqrt(2)]
L2 = 0.7ω₀ .* [0.0, 1.0, 0.0]

# Solver tolerances 
reltol = 1e-7
abstol = 1e-7

function L_true(t::Float64; τ₀=τ₀, p=[L0, L1, L2])
    if t < τ₀
        return p[1]
    elseif t < τ₁
        return p[2]
    else
        return p[3]
    end
end

function true_rotation!(du, u, p, t)
    L = L_true(t; τ₀=τ₀, p=p)
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, [0.0, 0.0, -1.0], tspan, [L0, L1, L2])
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = X_noiseless + FisherNoise(kappa=200.) 

##############################################################
#######################  Training  ###########################
##############################################################

data   = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

regs = [Regularization(order=1, power=1.0, λ=0.1, diff_mode="Finite Differences"), 
        Regularization(order=0, power=2.0, λ=0.1, diff_mode="Finite Differences")]

params = SphereParameters(tmin=tspan[1], tmax=tspan[2], 
                          reg=regs,
                          u0=[0.0, 0.0, -1.0], ωmax=2ω₀, reltol=reltol, abstol=abstol,
                          niter_ADAM=4000, niter_LBFGS=1000)

results = train(data, params, rng, nothing)

##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, -20., 150., "examples/multi_rotation/plot_sphere.pdf")
plot_L(data, results, saveas="examples/multi_rotation/plot_L.pdf")