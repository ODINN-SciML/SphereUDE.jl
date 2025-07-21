function test_grad_finite_diff(
    sensealg::ADJ;
    thres = [0., 0., 0.],
    finite_difference_order = 5,
    repeat_times = false
) where {ADJ<:AbstractAdjointMethod}

    rng = Random.default_rng()
    Random.seed!(rng, 613)

    thres_ratio = thres[1]
    thres_angle = thres[2]
    thres_relerr = thres[3]

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
    Δω₀ = 2.0
    # Angular velocity
    ω₀ = Δω₀ * π / 180.0

    # Create simple example
    X = zeros(3, N_samples)
    X[3, :] .= 1
    X[1, :] = LinRange(0, 1, N_samples)
    X = X ./ norm.(eachcol(X))'

    data = SphereData(times = times_samples, directions = X, kappas = nothing, L = nothing)

    params = SphereParameters(
        tmin = tspan[1],
        tmax = tspan[2],
        reg = nothing,
        train_initial_condition = false,
        multiple_shooting = false,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ω₀,
        reltol = 1e-12,
        abstol = 1e-12,
        niter_ADAM = 201,
        niter_LBFGS = 201,
        sensealg = sensealg,
        verbose = false
    )

    U = get_default_NN(params, rng, nothing)
    θ, st = Lux.setup(rng, U)
    β = SphereUDE.ComponentVector{Float64}(θ = θ)

    f_loss(β) = loss(β, data, params, U, st)
    loss_function(_β) = (first ∘ f_loss)(_β)
    loss_grad!(_dβ, _β) = SphereUDE.rotation_grad!(_dβ, _β, data, params, U, st, params.sensealg)

    dβ = zero(β)
    loss_grad!(dβ, β)

    dβ_FD, = FiniteDifferences.grad(
        FiniteDifferences.central_fdm(finite_difference_order, 1),
        _β -> loss_function(_β),
        β
    )

    ratio_FD, angle_FD, relerr_FD = stats_err_arrays(dβ, dβ_FD)

    printVecScientific("ratio  = ", [ratio_FD], thres_ratio)
    printVecScientific("angle  = ", [angle_FD], thres_angle)
    printVecScientific("relerr = ", [relerr_FD], thres_relerr)

    if typeof(sensealg) <: SphereUDE.DummyAdjoint
        @test true
    else
        @test abs(ratio_FD) < thres_ratio
        @test abs(angle_FD) < thres_angle
        @test abs(relerr_FD) < thres_relerr
    end
end

function stats_err_arrays(a::T, b::T) where T
    ratio = sqrt(sum(a.^2)) / sqrt(sum(b.^2)) - 1
    angle = sum(a.*b) / (sqrt(sum(a.^2)) * sqrt(sum(b.^2))) - 1
    relerr = sqrt(sum((a - b).^2)) / sqrt(sum((a).^2))
    return ratio, angle, relerr
end

printVecScientific(v) = join([@sprintf("%9.2e", e) for e in v], " ")
function printVecScientific(baseVarName, v, thres=nothing)
    print(baseVarName)
    for e in v
        if isnothing(thres)
            print(@sprintf("%9.2e", e))
        else
            if abs(e)<=thres
                printstyled(@sprintf("%9.2e", e); color=:green)
            else
                printstyled(@sprintf("%9.2e", e); color=:red)
            end
        end
        print(" ")
    end
    if !isnothing(thres)
        printstyled("< $(thres)"; color=:blue)
    end
    println("")
end