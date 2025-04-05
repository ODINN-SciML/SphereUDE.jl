using LinearAlgebra, Statistics, Distributions
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

##############################################################
###############  Simulation of Simple Rotation ###############
##############################################################

function test_single_rotation(; repeat_times = false)

    # Total time simulation
    tspan = [0, 160.0]
    # Number of sample points
    N_samples = 10
    # Times where we sample points
    random_times = rand(sampler(Uniform(tspan[1], tspan[2])), N_samples)
    if repeat_times
        # We repeat a few ages
        random_times[end] = random_times[1]
        random_times[end-1] = random_times[1]
        random_times[end-2] = random_times[2]
    end
    times_samples = sort(random_times)

    # Expected maximum angular deviation in one unit of time (degrees)
    Δω₀ = 1.0
    # Angular velocity
    ω₀ = Δω₀ * π / 180.0

    # Create simple example
    X = zeros(3, N_samples)
    X[3, :] .= 1
    X[1, :] = LinRange(0, 1, N_samples)
    X = X ./ norm.(eachcol(X))'

    ##############################################################
    #######################  Training  ###########################
    ##############################################################

    data = SphereData(times = times_samples, directions = X, kappas = nothing, L = nothing)

    regs = [
        Regularization(
            order = 1,
            power = 1.0,
            λ = 0.1,
            diff_mode = FiniteDifferences(1e-6),
        ),
        Regularization(
            order = 0,
            power = 2.0,
            λ = 0.001,
            diff_mode = nothing,
        ),
    ]

    params = SphereParameters(
        tmin = tspan[1],
        tmax = tspan[2],
        reg = regs,
        train_initial_condition = false,
        multiple_shooting = false,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ω₀,
        reltol = 1e-12,
        abstol = 1e-12,
        niter_ADAM = 201,
        niter_LBFGS = 201,
        verbose_step = 50,
        sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
    )

    results = train(data, params, rng, nothing, nothing)
    @test true
end
