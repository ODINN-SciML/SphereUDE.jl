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

# Example directory
example_dir = "examples/template"
# Solver tolerances 
reltol = 1e-7
abstol = 1e-7
# Fisher concentration parameter on observations (small = more dispersion)
κ = 200 
# Maximum angular velocity (in rad)
ωmax = 1.0
# Initial position
u0 = [0., 0., -1.]
# Total time simulation
tspan = [0, 130.0]
# Times where we sample points
times_samples = ...

"""
Angular momentum as a function of time and other parameters
"""
function L_true(t::Float64; ...)
    ...
end

function true_rotation!(du, u, p, t)
    L = L_true(t; ...)
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, u0, tspan, p)
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = mapslices(x -> rand(sampler(VonMisesFisher(x/norm(x), κ)), 1), X_noiseless, dims=1)

##############################################################
#######################  Training  ###########################
##############################################################

data   = SphereData(times=times_samples, directions=X_true, kappas=nothing, L=L_true)

params = SphereParameters(tmin=tspan[1], tmax=tspan[2], 
                          reg=[(1,1.0,1.0)],
                          u0=u0, ωmax=ωmax, reltol=reltol, abstol=abstol,
                          niter_ADAM=1000, niter_LBFGS=300)

results = train(data, params, rng, nothing)

##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, -20., 150., example_dir * "/plot_module.pdf")
plot_L(data, results, saveas=example_dir * "/plot_L.pdf")