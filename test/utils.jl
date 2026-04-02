
function test_coordinate()
    X₀ = [1 0 0 1/√2; 0 0 1 1/√2; 0 1 0 0]
    Y₀ = [0.0 90.0 0.0 0.0; 0.0 0.0 90.0 45.0]
    @test all(isapprox.(Y₀, cart2sph(X₀, radians = false), atol = 1e-6))
    @test all(isapprox.(Y₀ * π / 180.0, cart2sph(X₀, radians = true), atol = 1e-6))
    @test all(isapprox.(X₀, sph2cart(Y₀, radians = false), atol = 1e-6))
    @test all(isapprox.(X₀, sph2cart(Y₀ * π / 180.0, radians = true), atol = 1e-6))
end

function test_complex_activation()
    pure_real = 1.0
    pure_complex = 1.0 + 1.0 * im
    @test isapprox(Lux.sigmoid(pure_real), 0.7310585786300049, atol = 1e-6)
    @test isapprox(imag(SphereUDE.sigmoid(pure_complex)), 0.2019482276580129, atol = 1e-6)
end

function test_scale_norm()
    # Single vector: output norm must be ≤ scale, direction preserved
    scale = 0.05
    x = [1.0, 2.0, 3.0]
    y = scale_norm(x; scale = scale)
    @test norm(y) ≤ scale + 1e-10
    @test isapprox(y ./ norm(y), x ./ norm(x), atol = 1e-10)

    # Near-zero input should not produce NaN
    x_small = [1e-10, 0.0, 0.0]
    @test all(isfinite.(scale_norm(x_small; scale = scale)))

    # Batched: each column must satisfy the same properties
    X = randn(3, 50)
    Y = scale_norm(X; scale = scale)
    @test size(Y) == size(X)
    col_norms = vec(sqrt.(sum(abs2, Y; dims = 1)))
    @test all(col_norms .≤ scale + 1e-10)

    # Batched result must match column-wise single application
    Y_ref = reduce(hcat, [scale_norm(X[:, i]; scale = scale) for i in axes(X, 2)])
    @test isapprox(Y, Y_ref, atol = 1e-10)
end

function test_fourier_feature()
    n = 4

    # Single vector: output length must be 2n
    v = [0.5]
    f = fourier_feature(v; n = n)
    @test length(f) == 2 * n

    # Known values at v=0: sin terms = 0, cos terms = 1
    f0 = fourier_feature([0.0]; n = n)
    @test isapprox(f0[1:n], zeros(n), atol = 1e-10)
    @test isapprox(f0[n+1:end], ones(n), atol = 1e-10)

    # Batched output shape must be (2n, batch_size)
    X = reshape(collect(range(0.0, 1.0; length = 20)), 1, 20)
    F = fourier_feature(X; n = n)
    @test size(F) == (2 * n, 20)

    # Batched result must match column-wise single application
    F_ref = reduce(hcat, [fourier_feature([X[1, i]]; n = n) for i in axes(X, 2)])
    @test isapprox(F, F_ref, atol = 1e-10)
end

function test_integration()
    quad1 = GaussQuadrature(n_nodes = 100)
    quad2 = RandomGaussQuadrature(n_nodes_min = 100, n_nodes_max = 120)
    @test isapprox(numerical_integral(x -> 1, 0.0, 1.0, quad1), 1.0, rtol = 1e-6)
    @test isapprox(numerical_integral(x -> x, -1.0, 1.0, quad2), 0.0, atol = 1e-3)
    @test isapprox(numerical_integral(x -> x^2, -1.0, 1.0, quad1), 2 / 3.0, rtol = 1e-6)
end
