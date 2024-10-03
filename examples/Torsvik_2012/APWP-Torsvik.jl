using Pkg; Pkg.activate(".")
using Revise 

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

df = CSV.read("./examples/Torsvik_2012/Torsvik-etal-2012_dataset.csv", DataFrame, delim=",")

# Filter the plates that were once part of the supercontinent Gondwana

Gondwana = ["Amazonia", "Parana", "Colorado", "Southern_Africa", 
            "East_Antarctica", "Madagascar", "Patagonia", "Northeast_Africa",
            "Northwest_Africa", "Somalia", "Arabia"]
 
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
                        #   reg = [Regularization(order=1, power=2.0, λ=1e2, diff_mode=ComplexStepDifferentiation())], 
                          reg = nothing, 
                          pretrain = false, 
                          u0 = [0.0, 0.0, -1.0], ωmax = ω₀, 
                          reltol = 1e-8, abstol = 1e-8,
                          niter_ADAM = 3000, niter_LBFGS = 8000, 
                          sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))) 


                          
results = train(data, params, rng, nothing)

plot_sphere(data, results, -30., 0., saveas="examples/Torsvik_2012/plots/plot_sphere.pdf", title="Double rotation") # , matplotlib_rcParams=Dict("font.size"=> 50))
plot_L(data, results, saveas="examples/Torsvik_2012/plots/plot_L.pdf", title="Double rotation")
