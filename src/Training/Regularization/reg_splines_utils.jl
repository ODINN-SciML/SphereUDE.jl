export cubic_regularization

"""
Cubic regularization from Jupp (1987)
"""
function cubic_regularization(
    β::ComponentVector,
    U::Chain,
    st::NamedTuple,
    reg::CubicSplinesRegularization,
    params::AP
) where {AP<:AbstractParameters}

    smodel = StatefulLuxLayer{true}(U, β.θ, st)

    # Create prediction of solution time series in integration points
    nodes, weights = quadrature(params.tmin, params.tmax, params.n_quadrature)
    u_ = predict(β, params, nodes)

    if typeof(reg.diff_mode) <: LuxNestedAD
        # Automatic Differentiation

        if reg.diff_mode.method == "ForwardDiff"
            Jac = batched_jacobian(
                smodel,
                AutoForwardDiff(),
                reshape(nodes, 1, params.n_quadrature)
            )
        elseif reg.diff_mode.method == "Zygote"
            Jac = batched_jacobian(
                smodel,
                AutoZygote(),
                reshape(nodes, 1, params.n_quadrature)
            )
        else
            throw("Method for AD backend no implemented.")
        end

        L_cross_u = [
            cross(Jac[:, 1, j], u_[:, j])
            for j = 1:params.n_quadrature
        ]

    elseif typeof(reg.diff_mode) <: FiniteDifferences
        L_ = [
            central_fdm(τ -> smodel([τ]), t, reg.diff_mode.ϵ)
            for t = nodes
        ]
        L_cross_u = [
            cross(L_[j], u_[:, j])
            for j = 1:params.n_quadrature
        ]

    elseif typeof(reg.diff_mode) <: ComplexStepDifferentiation
        L_ = [
            complex_step_differentiation(τ -> smodel([τ]), t, reg.diff_mode.ϵ)
            for t = nodes
        ]
        L_cross_u = [
            cross(L_[j], u_[:, j])
            for j = 1:params.n_quadrature
        ]
    else
        throw("Method not implemented.")
    end

    return reg.λ * sum([weights[j] * norm(L_cross_u[j])^2.0 for j = 1:params.n_quadrature])
end