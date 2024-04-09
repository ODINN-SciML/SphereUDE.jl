using Pkg; Pkg.activate(".")
using Revise 

using LinearAlgebra, Statistics, Distributions 
using OrdinaryDiffEq
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using SphereUDE

##############################################################
###############  Simulation of Simple Example ################
##############################################################

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 616)

# Total time simulation
tspan = [0, 100.0]
# Number of sample points
N_samples = 300
# Times where we sample points
times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0  
# Angular velocity s
ω₀ = Δω₀ * π / 180.0

# Solver tolerances 
reltol = 1e-7
abstol = 1e-7

function L_true(t::Float64)
    τ = π * (t / tspan[2])
    ω₀ * [sin(τ), cos(τ), 0]
end

function true_rotation!(du, u, p, t)
    L = L_true(t)
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, [0.0, 0.0, 1.0], tspan)
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = X_noiseless #+ FisherNoise(kappa=10000.) 

##############################################################
#######################  Training  ###########################
##############################################################

data   = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

regs = [Regularization(order=1, power=1.0, λ=0.001, diff_mode="Finite Differences"), 
        Regularization(order=0, power=2.0, λ=0.1, diff_mode="Finite Differences")]

params = SphereParameters(tmin=tspan[1], tmax=tspan[2], 
                          reg=regs, 
                          u0=[0.0, 0.0, 1.0], ωmax=10*ω₀, reltol=reltol, abstol=abstol,
                          niter_ADAM=100, niter_LBFGS=60)

results = train(data, params, rng, nothing)

##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, 0., 0., saveas="examples/curl/plot_sphere.pdf", title="Double rotation")
plot_L(data, results, saveas="examples/curl/plot_L.pdf", title="Double rotation")