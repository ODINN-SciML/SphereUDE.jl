import Pkg
Pkg.activate(dirname(Base.current_project()))

using SphereUDE
using BenchmarkTools
using Logging
Logging.disable_logging(Logging.Info)
BenchmarkTools.DEFAULT_PARAMETERS.seconds = 2 * 60

using Distributions, Statistics, LinearAlgebra
using SciMLSensitivity

println("# Performance benchmark")

using Random
rng = Random.default_rng()
Random.seed!(666)

### Create a simple simulation

# Total time simulation
tspan = [0, 160.0]
# Number of sample points
N_samples = 100
# Times where we sample points
random_times = rand(sampler(Uniform(tspan[1], tspan[2])), N_samples)
times_samples = sort(random_times)

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0
# Angular velocity
ω₀ = Δω₀ * π / 180.0

# Create simple example
X = zeros(3, N_samples)
X[3, :] .= 1
X[1, :] = LinRange(0,1,N_samples)
X = X ./ norm.(eachcol(X))' 

data = SphereData(times=times_samples, directions=X, kappas=nothing, L=nothing)

### Types of regularization to benchmark
regularization_types = [
    nothing,
    [Regularization(order=0, power=1.0, λ=0.1, diff_mode=nothing)],
    [Regularization(order=1, power=1.0, λ=0.1, diff_mode=FiniteDifferences(1e-6))],
    [Regularization(order=1, power=1.0, λ=0.1, diff_mode=LuxNestedAD())],
    ]

# BenchmarkTools evaluates things at global scope
params_benchmark = []
for regs in regularization_types
    params = SphereParameters(
                tmin = tspan[1], tmax = tspan[2],
                reg = regs,
                train_initial_condition = false,
                multiple_shooting = false,
                u0 = [0.0, 0.0, -1.0], ωmax = ω₀, reltol = 1e-12, abstol = 1e-12,
                niter_ADAM = 11, niter_LBFGS = 0, verbose=false,
                sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true))
            )
    push!(params_benchmark, params)
end

for params in params_benchmark
    println("## Benchmark of $(params.reg)")
    println("> Training for a total of $(params.niter_ADAM+params.niter_LBFGS) epochs")
    trial = @benchmark train(data, $params, $rng, nothing, nothing)
    display(trial)
    println("")
end