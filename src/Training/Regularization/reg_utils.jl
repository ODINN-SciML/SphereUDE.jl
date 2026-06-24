export regularization
export isL1reg

# Adapt each regressor to the smodel([t]) calling convention used throughout
_make_smodel(regressor::NNRegressor,        θ) = StatefulLuxLayer{true}(regressor.model, θ, regressor.st)
_make_smodel(regressor::AbstractRegressor,  θ) = t_vec -> regressor(only(t_vec), θ)

"""
Regularization
"""
function regularization(
    θ::ComponentVector,
    regressor::AbstractRegressor,
    reg::Regularization,
    params::AP,
) where {AP<:AbstractParameters}

    smodel = _make_smodel(regressor, θ)

    # Define number of elements for quadrature
    nodes, weights = extract_nodes_weights(params.tmin, params.tmax, params.quadrature)
    n_quadrature = length(nodes)

    l_ = 0.0
    if reg.order == 0

        l_ += numerical_integral(
            t -> norm(smodel([t]))^reg.power,
            params.tmin,
            params.tmax,
            params.quadrature,
        )

    elseif reg.order == 1

        if typeof(reg.diff_mode) <: LuxNestedAD

            regressor isa NNRegressor || throw(ArgumentError(
                "LuxNestedAD regularization requires a NNRegressor (uses batched_jacobian on a Lux model). " *
                "Use FiniteDiff or ComplexStepDifferentiation instead."
            ))

            if reg.diff_mode.method == "ForwardDiff"
                # This is giving now the wrong results!!!
                Jac = batched_jacobian(
                    smodel,
                    AutoForwardDiff(),
                    reshape(nodes, 1, n_quadrature),
                )
            elseif reg.diff_mode.method == "Zygote"
                # This can also be done with Zygote in reverse mode
                Jac = batched_jacobian(
                    smodel,
                    AutoZygote(),
                    reshape(nodes, 1, n_quadrature),
                )
            else
                throw("Method for AD backend no implemented.")
            end

            # Compute the final agregation to the loss
            l_AD = sum([
                weights[j] * norm(Jac[:, 1, j])^reg.power for j = 1:n_quadrature
            ])
            l_ += l_AD

            # Test every a few iterations that AD is working properly
            @ignore_derivatives begin
                if rand(Bernoulli(0.1))
                    l_FD = numerical_integral(
                        t -> norm(central_fdm(τ -> smodel([τ]), t, 1e-8))^reg.power,
                        params.tmin,
                        params.tmax,
                        params.quadrature,
                    )
                    if !isapprox(l_AD, l_FD, rtol = 1e-1)
                        @warn """
                        Nested AD is giving significant different results than Finite
                        Differences.
                        Regularization with AD: $(l_AD) vs $(l_FD) using Finite Differences.
                        This can be produced by errors in the batched AD caused by
                        Lux.WrappedFunction and how this is done in batched_jacobian().
                        See https://discourse.julialang.org/t/using-lux-wrappedfunction-for-pre-post-processing-in-lux-model/129254
                        """
                    end
                end
            end

        elseif typeof(reg.diff_mode) <: FiniteDiff

            l_ += numerical_integral(
                t -> norm(central_fdm(τ -> smodel([τ]), t, reg.diff_mode.ϵ))^reg.power,
                params.tmin,
                params.tmax,
                params.quadrature,
            )

        elseif typeof(reg.diff_mode) <: ComplexStepDifferentiation

            l_ += numerical_integral(
                t ->
                    norm(
                        complex_step_differentiation(τ -> smodel([τ]), t, reg.diff_mode.ϵ),
                    )^reg.power,
                params.tmin,
                params.tmax,
                params.quadrature,
            )

        else
            throw("Method not implemented.")
        end

    else
        throw("Method not implemented.")
    end

    return reg.λ * l_
end


"""
Exact regularization for SplineRegressor.
Always uses the analytical B-spline time derivative — `diff_mode` is ignored.
"""
function regularization(
    θ::ComponentVector,
    regressor::SplineRegressor,
    reg::Regularization,
    params::AP,
) where {AP<:AbstractParameters}

    if reg.order == 0

        return reg.λ * numerical_integral(
            t -> norm(regressor(t, θ))^reg.power,
            params.tmin, params.tmax, params.quadrature,
        )

    elseif reg.order == 1

        return reg.λ * numerical_integral(
            t -> norm(time_derivative(regressor, t, θ))^reg.power,
            params.tmin, params.tmax, params.quadrature,
        )

    else
        throw("Regularization order not implemented for SplineRegressor.")
    end
end


"""
Custom rrule for regularization with SplineRegressor.

Provides an exact, AD-free gradient ∂loss/∂θ using the analytical VJP helpers
`_vjp_L` (order=0) and `_vjp_dLdt` (order=1), avoiding any AD trace through
the B-spline recursion.
"""
function ChainRulesCore.rrule(
    ::typeof(regularization),
    θ::ComponentVector,
    regressor::SplineRegressor,
    reg::Regularization,
    params::AP,
) where {AP<:AbstractParameters}

    val = regularization(θ, regressor, reg, params)

    function pullback(Δ)
        nodes, weights = extract_nodes_weights(params.tmin, params.tmax, params.quadrature)
        n_q = length(nodes)
        p   = reg.power

        grad_θ = if reg.order == 0

            mapreduce(+, 1:n_q) do j
                v  = regressor(nodes[j], θ)
                nv = norm(v)
                w  = nv < 1e-30 ? zero(nv) : weights[j] * p * nv^(p - 2)
                w .* _vjp_L(regressor, nodes[j], θ, v)
            end

        elseif reg.order == 1

            mapreduce(+, 1:n_q) do j
                v  = time_derivative(regressor, nodes[j], θ)
                nv = norm(v)
                w  = nv < 1e-30 ? zero(nv) : weights[j] * p * nv^(p - 2)
                w .* _vjp_dLdt(regressor, nodes[j], θ, v)
            end

        else
            throw("Regularization order not implemented for SplineRegressor.")
        end

        return (
            ChainRulesCore.NoTangent(),
            reg.λ * Δ .* grad_θ,
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
            ChainRulesCore.NoTangent(),
        )
    end

    return val, pullback
end


"""

Function to check for the presence of L1 regularization in the loss function.
"""
function isL1reg(regs::Union{Vector{R},Nothing}) where {R<:AbstractRegularization}
    if isnothing(regs)
        return false
    end
    for reg in regs
        if reg.power == 1
            return true
        end
    end
    return false
end
