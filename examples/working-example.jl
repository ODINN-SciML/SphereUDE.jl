using Pkg; Pkg.activate(".")

# SciML Tools
using OrdinaryDiffEq, SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL
# using ModelingToolkit, DataDrivenDiffEq,DataDrivenSparse

# Standard Libraries
using LinearAlgebra, Statistics

# External Libraries
using ComponentArrays, Lux, Zygote #, StableRNGs

# Set a random seed for reproducible behaviour
# rng = StableRNG(1111)
using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

# using TimerOutputs
# const to = TimerOutput()


function lotka!(du, u, p, t)
    α, β, γ, δ = p
    du[1] = α * u[1] - β * u[2] * u[1]
    du[2] = γ * u[1] * u[2] - δ * u[2]
end

# Define the experimental parameter
tspan = (0.0, 5.0)
u0 = 5.0f0 * rand(rng, 2)
p_ = [1.3, 0.9, 0.8, 1.8]
prob = ODEProblem(lotka!, u0, tspan, p_)
solution = solve(prob, Vern7(), abstol = 1e-12, reltol = 1e-12, saveat = 0.25)

# Add noise in terms of the mean
X = Array(solution)
t = solution.t

x̄ = mean(X, dims = 2)
noise_magnitude = 5e-3
Xₙ = X .+ (noise_magnitude * x̄) .* randn(rng, eltype(X), size(X))

# plot(solution, alpha = 0.75, color = :black, label = ["True Data" nothing])
# scatter!(t, transpose(Xₙ), color = :red, label = ["Noisy Data" nothing])

rbf(x) = exp.(-(x .^ 2))

# Multilayer FeedForward
const U = Lux.Chain(Lux.Dense(2, 5, rbf), Lux.Dense(5, 5, rbf), Lux.Dense(5, 5, rbf),
              Lux.Dense(5, 2))
# Get the initial parameters and state variables of the model
p, st = Lux.setup(rng, U)
const _st = st

# Define the hybrid model
function ude_dynamics!(du, u, p, t, p_true)
    û = U(u, p, _st)[1] # Network prediction
    du[1] = p_true[1] * u[1] + û[1]
    du[2] = -p_true[4] * u[2] + û[2]
end

# Closure with the known parameter
nn_dynamics!(du, u, p, t) = ude_dynamics!(du, u, p, t, p_)
# Define the problem
prob_nn = ODEProblem(nn_dynamics!, Xₙ[:, 1], tspan, p)

function predict(θ, X = Xₙ[:, 1], T = t)
    _prob = remake(prob_nn, u0 = X, tspan = (T[1], T[end]), p = θ)
    Array(solve(_prob, Vern7(), saveat = T,
                abstol = 1e-6, reltol = 1e-6,
                sensealg=QuadratureAdjoint(autojacvec=ReverseDiffVJP(true))))
end

# function loss(θ)
#     # Empirical error
#     X̂ = predict(θ)
#     l_emp = mean(abs2, Xₙ .- X̂)
#     return l_emp
# end

"""
Works but it is clearly a little bit slow! 
"""
# function loss(θ)
#     # Empirical error
#     X̂ = predict(θ)
#     l_emp = mean(abs2, Xₙ .- X̂)
#     # regularization 
#     steps_reg = collect(0.0:0.1:10.0)
#     Ux = map(x -> (first ∘ U)([x, 1.0], θ, st), steps_reg)
#     dUdx = diff(Ux) ./ diff(steps_reg)
#     l_reg = sum(norm.(dUdx).^2.0)    
#     return l_emp + l_reg
# end

"""
Need to solve problem with mutating arrays!
"""
function loss(θ)
    # Empirical error
    X̂ = predict(θ)
    l_emp = mean(abs2, Xₙ .- X̂)
    # regularization 
    steps_reg = collect(0.0:0.1:10.0)
    dUdx = map(x -> Zygote.jacobian(first ∘ U, [x, 1.0], θ, st)[1], steps_reg)
    norm_dUdx = norm.(dUdx).^2.0
    l_reg = sum(norm_dUdx) 
    return l_emp + l_reg
end

losses = Float64[]

callback = function (p, l)
    push!(losses, l)
    if length(losses) % 100 == 0
        println("Current loss after $(length(losses)) iterations: $(losses[end])")
    end
    return false
end

adtype = Optimization.AutoZygote()
optf = Optimization.OptimizationFunction((x, p) -> loss(x), adtype)
optprob = Optimization.OptimizationProblem(optf, ComponentVector{Float64}(p))

res1 = Optimization.solve(optprob, ADAM(), callback = callback, maxiters = 500)
println("Training loss after $(length(losses)) iterations: $(losses[end])")