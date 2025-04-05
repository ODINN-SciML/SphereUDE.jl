export regularization
export isL1reg

"""
Regularization
"""
function regularization(
    θ::ComponentVector,
    U::Chain,
    st::NamedTuple,
    reg::Regularization,
    params::AP
) where {AP<:AbstractParameters}

    # Define Statefull Lux NN
    smodel = StatefulLuxLayer{true}(U, θ, st)

    l_ = 0.0
    if reg.order == 0

        # l_ += quadrature(t -> norm(predict_L(t, U, θ, st))^reg.power, params.tmin, params.tmax, params.n_quadrature)
        l_ += quadrature(
            t -> norm(smodel([t]))^reg.power,
            params.tmin,
            params.tmax,
            params.n_quadrature
        )

    elseif reg.order == 1

        if typeof(reg.diff_mode) <: LuxNestedAD
            # Automatic Differentiation 
            nodes, weights = quadrature(params.tmin, params.tmax, params.n_quadrature)

            if reg.diff_mode.method == "ForwardDiff"
                # Jac = ForwardDiff.jacobian(smodel, reshape(nodes, 1, params.n_quadrature))
                Jac = batched_jacobian(
                    smodel,
                    AutoForwardDiff(),
                    reshape(nodes, 1, params.n_quadrature),
                )
            elseif reg.diff_mode.method == "Zygote"
                # This can also be done with Zygote in reverse mode
                # Jac = Zygote.jacobian(smodel, reshape(nodes, 1, params.n_quadrature))[1]
                Jac = batched_jacobian(
                    smodel,
                    AutoZygote(),
                    reshape(nodes, 1, params.n_quadrature),
                )
            else
                throw("Method for AD backend no implemented.")
            end

            # Compute the final agregation to the loss
            l_ += sum([
                weights[j] * norm(Jac[:,1,j])^reg.power
                for j = 1:params.n_quadrature
            ])

            # Test every a few iterations that AD is working properly
            ignore() do
                if rand(Bernoulli(0.001))
                    l_AD = sum([
                        weights[j] * norm(Jac[:,1,j])^reg.power
                        for j = 1:params.n_quadrature
                    ])
                    l_FD = quadrature(
                        t -> norm(central_fdm(τ -> smodel([τ]), t, 1e-5))^reg.power,
                        params.tmin,
                        params.tmax,
                        params.n_quadrature
                    )
                    if abs(l_AD - l_FD) > 1e-2 * abs(l_FD) 
                        @warn "[SphereUDE] Nested AD is giving significant different results than Finite Differences."
                        @printf "[SphereUDE] Regularization with AD: %.9f vs %.9f using Finite Differences" l_AD l_FD
                    end
                end
            end

        elseif typeof(reg.diff_mode) <: FiniteDifferences
            # Finite differences 
            # l_ += quadrature(t -> norm(central_fdm(τ -> predict_L(τ, U, θ, st), t, reg.diff_mode.ϵ))^reg.power, params.tmin, params.tmax, params.n_quadrature)
            l_ += quadrature(
                t -> norm(central_fdm(τ -> smodel([τ]), t, reg.diff_mode.ϵ))^reg.power,
                params.tmin,
                params.tmax,
                params.n_quadrature
            )

        elseif typeof(reg.diff_mode) <: ComplexStepDifferentiation
            # Complex step differentiation
            # l_ += quadrature(t -> norm(complex_step_differentiation(τ -> predict_L(τ, U, θ, st), t, reg.diff_mode.ϵ))^reg.power, params.tmin, params.tmax, params.n_quadrature) 
            l_ += quadrature(
                t -> norm(complex_step_differentiation(τ -> smodel([τ]), t, reg.diff_mode.ϵ))^reg.power,
                params.tmin,
                params.tmax,
                params.n_quadrature
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
