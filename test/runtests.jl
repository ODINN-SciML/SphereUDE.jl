using Revise

using SphereUDE
using Test
using Lux
using FiniteDifferences
using Printf

include("constructors.jl")
include("utils.jl")
include("rotation.jl")
include("python.jl")
include("gradient.jl")

@testset "Run all tests" begin

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
        "Dummy grandient" test_single_rotation(sensealg = SphereUDE.DummyAdjoint())
        "SciMLSensitivity gradient (without regularization)" test_single_rotation(sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)), use_regularization = false)
        "SciMLSensitivity gradient (with regularization)" test_single_rotation(sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
        "Custom Backsolve gradient" test_single_rotation(sensealg = SphereBackSolveAdjoint())
    end

    @testset "Custom Adjoint method" test_grad_finite_diff(SphereBackSolveAdjoint(); thres = [4e-3, 1e-6, 4e-3])

    @testset "Inversion with repeat times" begin
        test_single_rotation(repeat_times = true, sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
    end

end
