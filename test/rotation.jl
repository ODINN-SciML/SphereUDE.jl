using LinearAlgebra, Statistics, Distributions
using SciMLSensitivity
using Optimization, OptimizationOptimisers, OptimizationOptimJL

using Random
rng = Random.default_rng()
Random.seed!(rng, 666)

##############################################################
###############  Simulation of Simple Rotation ###############
##############################################################

function test_single_rotation(;
    repeat_times = false,
    use_regularization = true,
    sensealg = sensealg,
    regressor_builder = nothing,   # (params, rng) → AbstractRegressor, or nothing for default NN
)

    # Total time simulation
    tspan = [0, 160.0]
    # Number of sample points
    N_samples = 100
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
    Δω₀ = 3.0
    # Angular velocity
    ω₀ = Δω₀ * π / 180.0

    # Create simple example
    X = zeros(3, N_samples)
    X[3, :] .= 1
    X[1, :] = LinRange(0, 3, N_samples)
    X .+= rand(size(X)...)
    X = X ./ norm.(eachcol(X))'

    ##############################################################
    #######################  Training  ###########################
    ##############################################################

    data = SphereData(times = times_samples, directions = X, kappas = nothing, L = nothing)

    if use_regularization
        regs = [
            Regularization(order = 1, power = 1.0, λ = 1e3, diff_mode = FiniteDiff(1e-6)),
            Regularization(order = 0, power = 2.0, λ = 1e-6, diff_mode = nothing),
        ]
    else
        regs = nothing
    end

    # DummyAdjoint returns a random gradient, so LBFGS is meaningless
    niter_LBFGS = typeof(sensealg) <: SphereUDE.DummyAdjoint ? 0 : 51

    params = SphereParameters(
        tmin = tspan[1],
        tmax = tspan[2],
        reg = regs,
        train_initial_condition = false,
        multiple_shooting = false,
        pretrain = false,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ω₀,
        reltol = 1e-12,
        abstol = 1e-12,
        niter_ADAM = 101,
        niter_LBFGS = niter_LBFGS,
        verbose_step = 50,
        sensealg = sensealg,
    )

    regressor = isnothing(regressor_builder) ? nothing : regressor_builder(params, rng)
    regressor_type = isnothing(regressor) ? "default NNRegressor" : typeof(regressor)
    @info "Testing inversion | regressor=$regressor_type | sensealg=$(typeof(sensealg)) | regularization=$use_regularization | repeat_times=$repeat_times"
    results = train(data, params, rng, nothing, regressor)

    @test true
    if !(typeof(sensealg) <: SphereUDE.DummyAdjoint)
        @test results.losses[end] < 0.70 * results.losses[begin]
    end
end
