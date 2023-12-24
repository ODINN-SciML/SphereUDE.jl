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
Random.seed!(rng, 000666)
# Fisher concentration parameter on observations (small = more dispersion)
κ = 200 

# Total time simulation
tspan = [0, 130.0]
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
L1 = 0.5ω₀ .* [0.0, sqrt(2), sqrt(2)]

# Solver tolerances 
reltol = 1e-7
abstol = 1e-7

function true_rotation!(du, u, p, t)
    if t < τ₀
        L = p[1]
    else 
        L = p[2]
    end
    du .= cross(L, u)
end

prob = ODEProblem(true_rotation!, [0.0, 0.0, -1.0], tspan, [L0, L1])
true_sol  = solve(prob, Tsit5(), reltol=reltol, abstol=abstol, saveat=times_samples)

# Add Fisher noise to true solution 
X_noiseless = Array(true_sol)
X_true = mapslices(x -> rand(sampler(VonMisesFisher(x/norm(x), κ)), 1), X_noiseless, dims=1)

##############################################################
#######################  Training  ###########################
##############################################################

data   = SphereData(times=times_samples, directions=X_true, kappas=nothing)
params = SphereParameters(tmin=tspan[1], tmax=tspan[2], 
                          u0=[0.0, 0.0, -1.0], ωmax=2*ω₀, reltol=reltol, abstol=abstol,
                          niter_ADAM=1000, niter_LBFGS=300)

results = train_sphere(data, params, rng, nothing)


##############################################################
######################  PyCall Plots #########################
##############################################################

using PyPlot, PyCall

X_true_sph = cart2sph(X_true, radians=false)

mpl_colors = pyimport("matplotlib.colors")
mpl_colormap = pyimport("matplotlib.cm")
sns = pyimport("seaborn")
ccrs = pyimport("cartopy.crs")
feature = pyimport("cartopy.feature")

plt.figure(figsize=(10,10))
ax = plt.axes(projection=ccrs.Orthographic(central_latitude=-20, central_longitude=150))

# ax.coastlines()
ax.gridlines()
ax.set_global()

cmap = mpl_colormap.get_cmap("viridis")
# norm = mpl_colors.Normalize(results.fit_times[1], results.fit_times[end])

sns.scatterplot(ax=ax, x = X_true_sph[1,:], y=X_true_sph[2, :], 
                hue = times_samples, s=50,
                palette="viridis",
                transform = ccrs.PlateCarree());

X_fit_sph = cart2sph(results.fit_directions, radians=false)

for i in 1:(length(results.fit_times)-1)
    plt.plot([X_fit_sph[1,i], X_fit_sph[1,i+1]], 
             [X_fit_sph[2,i], X_fit_sph[2,i+1]],
              linewidth=2, color="black",#cmap(norm(results.fit_times[i])),
              transform = ccrs.Geodetic())
end

plt.savefig("examples/plot.pdf", format="pdf")