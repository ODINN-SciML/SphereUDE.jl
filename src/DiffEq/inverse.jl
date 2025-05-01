# Implementation of inverse models


"""
Manual gradient computation using backsolve adjoint
"""
function rotation_grad!(
    dβ::ComponentVector,
    β::ComponentVector,
    params::AP
    ) where {AP<:AbstractParameters}

    # Compute final solution of forward model
    t₀, t₀ = params.tmin, params.tmax

    u_final = predict(β, params, [t₁])
    λ_final = [0.0, 0.0, 0.0]
    dLdβ_final = 0.0

    U₀ = [u_final; λ_final; dLdβ_final]

    function rotation_reverse(dU, U, p, t)

    end

    dθ .= (rand() - 0.5) .* θ
end
