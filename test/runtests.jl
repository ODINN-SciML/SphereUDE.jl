using SphereUDE
using Test

include("constructors.jl")
include("utils.jl")


@testset "Constructors" begin
    test_reg_constructor()
    test_param_constructor()
end

@testset "Utils" begin 
    test_cart2sph()
end
