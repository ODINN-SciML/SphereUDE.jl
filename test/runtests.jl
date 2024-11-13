using SphereUDE
using Test
using Lux

include("constructors.jl")
include("utils.jl")
include("rotation.jl")
include("python.jl")

@testset "Constructors" begin
    test_reg_constructor()
    test_param_constructor()
end

@testset "Utils" begin 
    test_coordinate()
    test_complex_activation()
    test_integration()
end

@testset "Python Integration" begin
    test_matplotplib()
    test_pmagpy()
end

@testset "Inversion" begin
    test_single_rotation()
end
