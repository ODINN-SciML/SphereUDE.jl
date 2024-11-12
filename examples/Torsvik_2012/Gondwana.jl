using Pkg; Pkg.activate(".")
using Revise 
using Lux

using LinearAlgebra, Statistics, Distributions 
using SciMLSensitivity
# using OrdinaryDiffEqCore, OrdinaryDiffEqTsit5
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using SphereUDE

# Random seed
using Random
rng = Random.default_rng()
Random.seed!(rng, 613)

using DataFrames, CSV
using Serialization, JLD2

df = CSV.read("./examples/Torsvik_2012/Torsvik-etal-2012_dataset.csv", DataFrame, delim=",")

# Filter the plates that were once part of the supercontinent Gondwana

Gondwana = ["Amazonia", "Parana", "Colorado", "Southern_Africa", 
            "East_Antarctica", "Madagascar", "Patagonia", "Northeast_Africa",
            "Northwest_Africa", "Somalia", "Arabia", "East_Gondwana"]
 
df = filter(row -> row.Plate ∈ Gondwana, df)
df.Times = df.Age .+= rand(sampler(Normal(0,0.1)), nrow(df))  # Needs to fix this! 

df = sort(df, :Times)
times = df.Times

# Fill missing values
df.RLat .= coalesce.(df.RLat, df.Lat)
df.RLon .= coalesce.(df.RLon, df.Lon)

X = sph2cart(Matrix(df[:,["RLat","RLon"]])'; radians=false)

# Retrieve uncertanties from poles and convert α95 into κ
kappas = (140.0 ./ df.a95).^2

data = SphereData(times=times, directions=X, kappas=kappas, L=nothing)

# Training

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.5   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0

tspan = [times[begin], times[end]]

params = SphereParameters(tmin = tspan[1], tmax = tspan[2], 
                          reg = [Regularization(order=1, power=2.0, λ=1e5, diff_mode=FiniteDifferences(1e-4))], 
                          # reg = nothing, 
                          pretrain = false, 
                          u0 = [0.0, 0.0, -1.0], ωmax = ω₀, 
                          reltol = 1e-6, abstol = 1e-6,
                          niter_ADAM = 5000, niter_LBFGS = 5000, 
                          sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))) 
# params = SphereParameters(tmin = tspan[1], tmax = tspan[2], 
#                           reg = [Regularization(order=1, power=2.0, λ=1.0, diff_mode=FiniteDifferences(1e-4))], 
#                           pretrain = false, 
#                           u0 = [0.0, 0.0, -1.0], ωmax = ω₀, 
#                           reltol = 1e-6, abstol = 1e-6,
#                           niter_ADAM = 2000, niter_LBFGS = 2000, 
#                           sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), 
#                           hyperparameter_balance = false) 


init_bias(rng, in_dims) = LinRange(tspan[1], tspan[2], in_dims)
init_weight(rng, out_dims, in_dims) = 0.1 * ones(out_dims, in_dims)

# Customized neural network to similate weighted moving window in L
U = Lux.Chain(
    Lux.Dense(1, 200, rbf, init_bias=init_bias, init_weight=init_weight, use_bias=true),
    Lux.Dense(200,10, gelu),
    Lux.Dense(10, 3, Base.Fix2(sigmoid_cap, params.ωmax), use_bias=false)
)
    
results = train(data, params, rng, nothing, U)
results_dict = convert2dict(data, results)


JLD2.@save "examples/Torsvik_2012/results/results_dict.jld2" results_dict

plot_sphere(data, results, -30., 0., saveas="examples/Torsvik_2012/plots/plot_sphere.pdf", title="Double rotation")
plot_sphere(data, results, -30., 0., saveas="examples/Torsvik_2012/plots/plot_sphere.png", title="Double rotation")
plot_L(data, results, saveas="examples/Torsvik_2012/plots/plot_L.pdf", title="Double rotation")
