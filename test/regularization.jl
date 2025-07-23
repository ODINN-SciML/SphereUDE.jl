function test_quadrature(thres = 1e-4)

    rng = Random.default_rng()
    Random.seed!(rng, 613)

    reg_AD = [Regularization(order = 1, power = 2.0, λ = 10.0^6.0, diff_mode = LuxNestedAD())]
    reg_FD = [Regularization(order = 1, power = 2.0, λ = 10.0^6.0, diff_mode = FiniteDiff(ϵ=1e-8))]

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

    θ, st = Lux.setup(rng, U)
    β = SphereUDE.ComponentArray{Float64}(θ)

    l_AD = regularization(β, U, st, only(reg_AD), params_AD)
    l_FD = regularization(β, U, st, only(reg_FD), params_FD)

    @test isapprox(l_AD, l_FD, rtol = thres)
end
