using Revise

using SphereUDE
using Test
using Lux
using FiniteDifferences
using Printf
import ChainRulesCore

import Random
Random.seed!(666)

include("constructors.jl")
include("data.jl")
include("bootstrap.jl")
include("utils.jl")
include("rotation.jl")
include("gradient.jl")
include("regularization.jl")

# Set environment variable SPHERE_FAST_TESTS=true to run a reduced subset.
# From the shell:
#   SPHERE_FAST_TESTS=true julia --project test/runtests.jl
# From an active Julia session (then run ]test in Pkg mode):
#   ENV["SPHERE_FAST_TESTS"] = "true"
const FAST_TESTS = get(ENV, "SPHERE_FAST_TESTS", "false") == "true"

@testset "Run all tests" begin

    @testset "Constructors" begin
        test_reg_constructor()
        test_param_constructor()
        test_regressor_interface()
    end

    @testset "Data" begin
        test_resample_data()
    end

    @testset "Uncertainty quantification" begin
        test_sample_uq()
        test_train_multistart_parallel()
        test_plot_sphere_ensemble()
    end

    if !FAST_TESTS
        @testset "Utils" begin
            test_coordinate()
            test_complex_activation()
            test_integration()
            test_scale_norm()
            test_fourier_feature()
        end
    end

    @testset "Regularization with AD vs FD" begin
        test_quadrature()
    end

    @testset "SplineRegressor regularization gradient (rrule vs FD)" begin
        test_spline_regularization_gradient()
    end

    @testset "Custom Adjoint method" test_grad_finite_diff(
        SphereBackSolveAdjoint(
            reltol = 1e-12,
            abstol = 1e-12,
        );
        thres = [4e-3, 4e-5, 9e-3],
    )

    @testset "Inversion" begin
        if FAST_TESTS
            @testset "SciMLSensitivity (with regularization)" test_single_rotation(
                sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
            )
            @testset "Custom Backsolve (without regularization)" test_single_rotation(
                sensealg = SphereBackSolveAdjoint(),
                use_regularization = false,
            )
        else
            @testset "Dummy gradient" test_single_rotation(sensealg = SphereUDE.DummyAdjoint())
            @testset "SciMLSensitivity gradient (without regularization)" begin
                @testset "Interpolating" test_single_rotation(
                    sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
                    use_regularization = false,
                )
                @testset "Quadrature" test_single_rotation(
                    sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
                    use_regularization = false
                )
                @testset "Gauss" test_single_rotation(
                    sensealg = GaussAdjoint(autojacvec = ReverseDiffVJP(true)),
                    use_regularization = false,
                )
            end
            @testset "SciMLSensitivity gradient (with regularization)" test_single_rotation(
                sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
            )
            @testset "Custom Backsolve gradient (without regularization)" test_single_rotation(
                sensealg = SphereBackSolveAdjoint(),
                use_regularization = false,
            )
        end
    end

    if !FAST_TESTS
        @testset "Inversion with repeat times" begin
            @testset test_single_rotation(
                repeat_times = true,
                sensealg = QuadratureAdjoint(autojacvec = ReverseDiffVJP(true)),
            )
        end
    end

    @testset "Inversion with SplineRegressor" begin
        @testset "SciMLSensitivity (without regularization)" test_single_rotation(
            sensealg           = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
            use_regularization = false,
            regressor_builder  = (params, rng) -> get_default_splines(params, rng),
        )
        @testset "SciMLSensitivity (with regularization)" test_single_rotation(
            sensealg           = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true)),
            use_regularization = true,
            regressor_builder  = (params, rng) -> get_default_splines(params, rng),
        )
        @testset "Custom Backsolve (without regularization)" test_single_rotation(
            sensealg           = SphereBackSolveAdjoint(),
            use_regularization = false,
            regressor_builder  = (params, rng) -> get_default_splines(params, rng),
        )
        @testset "Custom Backsolve (with regularization)" test_single_rotation(
            sensealg           = SphereBackSolveAdjoint(),
            use_regularization = true,
            regressor_builder  = (params, rng) -> get_default_splines(params, rng),
        )
    end

end
