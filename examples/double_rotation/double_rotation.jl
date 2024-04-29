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
Random.seed!(rng, 666)

# Total time simulation
tspan = [0, 160.0]
# Number of sample points
N_samples = 50
# Times where we sample points
times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Change point
τ₀ = 65.0
# Angular momentum
L0 = ω₀    .* [1.0, 0.0, 0.0]
L1 = 0.5ω₀ .* [0.0, 1/sqrt(2), 1/sqrt(2)]

# Solver tolerances 
reltol = 1e-7
abstol = 1e-7

function L_true(t::Float64; τ₀=τ₀, p=[L0, L1])
    if t < τ₀
        return p[1]
    else
        return p[2]
    end
end

function true_rotation!(du, u, p, t)
    L = L_true(t; τ₀=τ₀, p=p)
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, [0.0, 0.0, -1.0], tspan, [L0, L1])
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = X_noiseless + FisherNoise(kappa=50.) 

##############################################################
#######################  Training  ###########################
##############################################################

data   = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

regs = [Regularization(order=1, power=1.0, λ=1.0, diff_mode="CS"), 
        Regularization(order=0, power=2.0, λ=0.1, diff_mode=nothing)]
# regs = [Regularization(order=0, power=2.0, λ=0.1, diff_mode=nothing)] 
        # Regularization(order=1, power=1.1, λ=0.01, diff_mode="CS")]
# regs = nothing

params = SphereParameters(tmin=tspan[1], tmax=tspan[2], 
                          reg=regs, 
                          u0=[0.0, 0.0, -1.0], ωmax=ω₀, reltol=reltol, abstol=abstol,
                          niter_ADAM=1000, niter_LBFGS=800)

results = train(data, params, rng, nothing; train_initial_condition=false)


##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, -20., 150., saveas="examples/double_rotation/plot_sphere.pdf", title="Double rotation")
plot_L(data, results, saveas="examples/double_rotation/plot_L.pdf", title="Double rotation")