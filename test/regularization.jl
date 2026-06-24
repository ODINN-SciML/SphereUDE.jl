"""
Test that the analytical gradient of `regularization` w.r.t. θ (via the rrule)
matches a finite-difference estimate, for both order=0 and order=1.
"""
function test_spline_regularization_gradient(thres = 1e-4)

    rng = Random.default_rng()
    Random.seed!(rng, 42)

    tmin, tmax = 0.0, 10.0
    ωmax = 0.1

    params = SphereParameters(
        tmin = tmin,
        tmax = tmax,
        reg  = nothing,
        u0   = [0.0, 0.0, -1.0],
        ωmax = ωmax,
    )

    regressor = get_default_splines(params, rng)
    θ         = init_params(regressor, rng)

    for (order, power, label) in [
            (0, 2.0, "order=0 power=2"),
            (1, 2.0, "order=1 power=2"),
            (1, 1.0, "order=1 power=1 (L1)"),
        ]

        reg = Regularization(order = order, power = power, λ = 1.0, diff_mode = nothing)

        # Analytical gradient via rrule
        _, pb   = ChainRulesCore.rrule(SphereUDE.regularization, θ, regressor, reg, params)
        grad_analytical = pb(1.0)[2]

        # Finite-difference gradient
        loss_fn(θ_) = SphereUDE.regularization(θ_, regressor, reg, params)
        grad_fd = FiniteDifferences.grad(FiniteDifferences.central_fdm(5, 1), loss_fn, θ)[1]

        @info "Spline reg gradient | $label | max err = $(maximum(abs.(grad_analytical .- grad_fd)))"
        @test isapprox(collect(grad_analytical), collect(grad_fd), rtol = thres)
    end
end


function test_quadrature(thres = 1e-4)

    rng = Random.default_rng()
    Random.seed!(rng, 613)

    reg_AD =
        [Regularization(order = 1, power = 2.0, λ = 10.0^6.0, diff_mode = LuxNestedAD())]
    reg_FD = [
        Regularization(
            order = 1,
            power = 2.0,
            λ = 10.0^6.0,
            diff_mode = FiniteDiff(ϵ = 1e-8),
        ),
    ]

    tmin, tmax = 0.0, 1.0
    ωmax = 0.01

    params_AD = SphereParameters(
        tmin = tmin,
        tmax = tmax,
        reg = reg_AD,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ωmax,
    )
    params_FD = SphereParameters(
        tmin = tmin,
        tmax = tmax,
        reg = reg_FD,
        u0 = [0.0, 0.0, -1.0],
        ωmax = ωmax,
    )

    n_fourier_features = 5
    U = Lux.Chain(
        # Scale function to bring input to [-1.0, 1.0]
        Lux.WrappedFunction(t -> 2.0 .* (t .- tmin) ./ (tmax .- tmin) .- 1.0),
        Lux.WrappedFunction(x -> fourier_feature(x; n = n_fourier_features)),
        Lux.Dense(2 * n_fourier_features, 10, tanh),
        Lux.Dense(10, 3, tanh),
        # Output function to scale output to have norm less than ωmax
        Lux.WrappedFunction(x -> scale_norm(ωmax * x; scale = ωmax)),
    )

    regressor, θ₀ = NNRegressor(U, rng)
    β = SphereUDE.ComponentArray{Float64}(θ₀)

    l_AD = regularization(β, regressor, only(reg_AD), params_AD)
    l_FD = regularization(β, regressor, only(reg_FD), params_FD)

    @test isapprox(l_AD, l_FD, rtol = thres)
end
