using Pkg; Pkg.activate(".")
using Revise 

using SphereUDE
using DataFrames, CSV
using Random
rng = Random.default_rng()
Random.seed!(rng, 666)


df = CSV.read("examples/APWP_Jupp1987/Jupp-etal-1987_dataset.csv", DataFrame; header=true)

times = df.Time
times .+= rand(sampler(Normal(0,0.1)), length(times))  # Needs to fix this! 
X = sph2cart(Matrix(df[:,["Lat","Lon"]])'; radians=false)

data = SphereData(times=times, directions=X, kappas=nothing, L=nothing)

# Expected maximum angular deviation in one unit of time (degrees)
Δω₀ = 1.0   
# Angular velocity 
ω₀ = Δω₀ * π / 180.0

# regs = [Regularization(order=1, power=1.0, λ=0.001, diff_mode="Finite Differences"), 
#         Regularization(order=0, power=2.0, λ=0.1, diff_mode="Finite Differences")]
# regs = [Regularization(order=1, power=2.0, λ=0.1, diff_mode="Finite Differences")]
regs = nothing

params = SphereParameters(tmin=0., tmax=520., 
                          reg=regs, 
                          u0=[0.0, 0.0, 1.0], ωmax=ω₀, reltol=1e-7, abstol=1e-7,
                          niter_ADAM=1000, niter_LBFGS=400)

results = train(data, params, rng, nothing)

##############################################################
######################  PyCall Plots #########################
##############################################################

plot_sphere(data, results, mean(df.Lat), mean(df.Lon), saveas="examples/APWP_Jupp1987/plot_sphere.pdf", title="Double rotation")
plot_L(data, results, saveas="examples/APWP_Jupp1987/plot_L.pdf", title="Double rotation")