using Pkg; Pkg.activate(".")
using Revise 

using LinearAlgebra, Statistics, Distributions 
using SciMLSensitivity
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using Optimization, OptimizationOptimisers, OptimizationOptimJL
using Lux 
using JLD2

using SphereUDE

##############################################################
###############  Simulation of Simple Example ################
##############################################################

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 333)
Random.seed!(rng, 333)

# Total time simulation
tspan = [0, 100.0]
# Number of sample points
N_samples = 300
# Times where we sample points
times_samples = collect(LinRange(tspan[1], tspan[2], N_samples))
times_samples = collect(LinRange(tspan[1], tspan[2], N_samples))

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 10.0  
# Angular velocity 
Δω₀ = 10.0  
# Angular velocity 
ω₀ = Δω₀ * π / 180.0
# Angular momentum
# Angular momentum

# Solver tolerances 
reltol = 1e-6
abstol = 1e-6
reltol = 1e-6
abstol = 1e-6

function L_true(t::Float64)
    lon = 0.0
    lat = -40.0 * (t - tspan[2]) / (tspan[2] - tspan[1]) - 40.0 * (t - tspan[1]) / (tspan[2] - tspan[1])
    return ω₀ * sph2cart([lat, lon], radians=false)
    lon = 0.0
    lat = -40.0 * (t - tspan[2]) / (tspan[2] - tspan[1]) - 40.0 * (t - tspan[1]) / (tspan[2] - tspan[1])
    return ω₀ * sph2cart([lat, lon], radians=false)
end

function true_rotation!(du, u, p, t)
    L = L_true(t)
    du .= cross(L, u)
end

x0 = [0.5, 0.0, 0.7]
x0 /= norm(x0)

prob = ODEProblem(true_rotation!, x0, tspan)
true_sol  = solve(prob, Tsit5(), reltol = reltol, abstol = abstol, saveat = times_samples)
x0 = [0.5, 0.0, 0.7]
x0 /= norm(x0)

prob = ODEProblem(true_rotation!, x0, tspan)
true_sol  = solve(prob, Tsit5(), reltol = reltol, abstol = abstol, saveat = times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = X_noiseless #+ FisherNoise(kappa=1000.0) 
X_true = X_noiseless #+ FisherNoise(kappa=1000.0) 

##############################################################
#######################  Training  ###########################
##############################################################

data = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)
data = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

params = SphereParameters(tmin = tspan[1], tmax = tspan[2], 
                          reg = nothing, #[Regularization(order=1, power=2.0, λ=1e0, diff_mode=LuxNestedAD())], 
                          u0 = [0.0, 0.0, -1.0],
                          train_initial_condition = true,
                          ωmax = ω₀, reltol = reltol, abstol = abstol,
                          niter_ADAM = 2000, niter_LBFGS = 8000, 
                          pretrain = false, 
                          sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))) 

init_bias(rng, in_dims) = LinRange(tspan[1], tspan[2], in_dims)
init_weight(rng, out_dims, in_dims) = 0.1 * ones(out_dims, in_dims)

U = Lux.Chain(
    Lux.Dense(1, 200, rbf, init_bias=init_bias, init_weight=init_weight, use_bias=true),
    Lux.Dense(200,10, gelu),
    Lux.Dense(10, 3, Base.Fix2(sigmoid_cap, params.ωmax), use_bias=false)
)

results = train(data, params, rng, nothing, U)
results_dict = convert2dict(data, results)

JLD2.@save "examples/curl/results/results_dict.jld2" results_dict

##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, 0.0, 0.0, saveas="examples/curl/_sphere.pdf", title="Curl") # , matplotlib_rcParams=Dict("font.size"=> 50))
plot_L(data, results, saveas="examples/curl/_L.pdf", title="Curl")

plot_sphere(data, results, 0.0, 0.0, saveas="examples/curl/_sphere.pdf", title="Curl") # , matplotlib_rcParams=Dict("font.size"=> 50))
plot_L(data, results, saveas="examples/curl/_L.pdf", title="Curl")
