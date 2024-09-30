using Pkg; Pkg.activate(".")
using Revise 

using LinearAlgebra, Statistics, Distributions 
using SciMLSensitivity
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using SphereUDE

##############################################################
###############  Simulation of Simple Example ################
##############################################################

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

function run(;kappa=50., regs=regs, title="plot")

# Total time simulation
tspan = [0, 160.0]
# Number of sample points
N_samples = 100
# Times where we sample points
times_samples = sort(rand(sampler(Uniform(tspan[1], tspan[2])), N_samples))

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Change point
τ₀ = 70.0
# Angular momentum
L0 = ω₀    .* [1.0, 0.0, 0.0]
L1 = 0.6ω₀ .* [0.0, 1/sqrt(2), 1/sqrt(2)]

# Solver tolerances 
reltol = 1e-12
abstol = 1e-12

function L_true(t::Float64; τ₀=τ₀, p=[L0, L1])
    if t < τ₀
        return p[1]
    else
        return p[2]
    end
end

function true_rotation!(du, u, p, t)
    L = L_true(t; τ₀ = τ₀, p = p)
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, [0.0, 0.0, -1.0], tspan, [L0, L1])
true_sol  = solve(prob, Tsit5(), reltol = reltol, abstol = abstol, saveat = times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = X_noiseless + FisherNoise(kappa=kappa) 

##############################################################
#######################  Training  ###########################
##############################################################

data = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

params = SphereParameters(tmin = tspan[1], tmax = tspan[2], 
                          reg = regs, 
                          train_initial_condition = false,
                          multiple_shooting = false, 
                          u0 = [0.0, 0.0, -1.0], ωmax = ω₀, reltol = reltol, abstol = abstol,
                          niter_ADAM = 2000, niter_LBFGS = 2000, 
                          sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))) 

results = train(data, params, rng, nothing)

##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, -20., 125., saveas="examples/double_rotation/" * title * "_sphere.pdf", title="Double rotation") # , matplotlib_rcParams=Dict("font.size"=> 50))
plot_L(data, results, saveas="examples/double_rotation/" * title * "_L.pdf", title="Double rotation")

end # run

# Run different experiments

ϵ = 1e-5

λ₀ = 0.1
λ₁ = 0.01

run(; kappa = 50., 
      regs = [Regularization(order=1, power=1.0, λ=λ₁, diff_mode=FiniteDifferences(ϵ)),  
              Regularization(order=0, power=2.0, λ=λ₀)], 
      title = "plots/plot_50_lambda$(λ₁)")

λ₀ = 0.1
λ₁ = 0.1

run(; kappa = 200., 
      regs = [Regularization(order=1, power=1.0, λ=λ₁, diff_mode=FiniteDifferences(ϵ)),  
              Regularization(order=0, power=2.0, λ=λ₀)], 
      title = "plots/plot_200_lambda$(λ₁)")

λ₀ = 0.1
λ₁ = 0.1

run(; kappa = 1000., 
      regs = [Regularization(order=1, power=1.0, λ=λ₁, diff_mode=FiniteDifferences(ϵ)),  
              Regularization(order=0, power=2.0, λ=λ₀)], 
      title = "plots/plot_1000_lambda$(λ₁)")