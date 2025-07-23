using Revise

using SphereUDE
using Test
using Lux
using FiniteDifferences
using Printf

import Random
Random.seed!(666)

include("constructors.jl")
include("utils.jl")
include("rotation.jl")
include("python.jl")
include("gradient.jl")
include("regularization.jl")

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
        # test_pmagpy()
    end

    @testset "Regularization with AD vs FD" begin
        test_quadrature()
    end

    @testset "Custom Adjoint method" test_grad_finite_diff(SphereBackSolveAdjoint(); thres = [4e-3, 1e-6, 4e-3])

    @testset "Inversion" begin
        @testset "Dummy grandient" test_single_rotation(sensealg = SphereUDE.DummyAdjoint())
        @testset "SciMLSensitivity gradient (without regularization)" begin
            @testset "Interpolating" test_single_rotation(sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)), use_regularization = false)
            # @testset "Quadrature" test_single_rotation(sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)), use_regularization = false)
            @testset "Gauss" test_single_rotation(sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true)), use_regularization = false)
            # @testset "Backsolve" test_single_rotation(sensealg = BacksolveAdjoint(autojacvec = ReverseDiffVJP(true)), use_regularization = false)
        end
        @testset "SciMLSensitivity gradient (with regularization)" test_single_rotation(sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)))
        @testset "Custom Backsolve gradient (without regularization)" test_single_rotation(sensealg = SphereBackSolveAdjoint(), use_regularization = false)
    end

    @testset "Inversion with repeat times" begin
        @testset test_single_rotation(
            repeat_times = true,
            sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
        )
    end

end
