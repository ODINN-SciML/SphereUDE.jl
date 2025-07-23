
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

function test_integration()
    quad1 = GaussQuadrature(n_nodes = 100)
    quad2 = RandomGaussQuadrature(n_nodes_min = 100, n_nodes_max = 120)
    @test isapprox(numerical_integral(x -> 1, 0.0, 1.0, quad1), 1.0, rtol = 1e-6)
    @test isapprox(numerical_integral(x -> x, -1.0, 1.0, quad2), 0.0, atol = 1e-3)
    @test isapprox(numerical_integral(x -> x^2, -1.0, 1.0, quad1), 2 / 3.0, rtol = 1e-6)
end
