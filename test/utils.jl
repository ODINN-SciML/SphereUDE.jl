
function test_cart2sph()
    X₀ = [1 0 0 √2; 0 1 0 √2; 0 0 1 0]
    Y₀ = [0.0 90.0 0.0 45.0; 0.0 0.0 90.0 0.0]
    @test all(isapprox.(Y₀, cart2sph(X₀, radians=false), atol=1e-6))
end