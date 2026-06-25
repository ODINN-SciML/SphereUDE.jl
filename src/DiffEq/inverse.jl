# Implementation of inverse models


"""
Manual gradient computation using backsolve adjoint
"""
function rotation_grad!(
    dβ::ComponentVector,
    β::ComponentVector,
    data::AD,
    params::AP,
    regressor::AbstractRegressor,
    sensealg::SphereBackSolveAdjoint,
    ) where {AD<:AbstractData, AP<:AbstractParameters}

    # Compute final solution of forward model
    t₁ = params.tmax

    _u = predict(β, params, [t₁], regressor)
    if !(_u isa AbstractArray) || !all(isfinite, _u)
        @error "[SphereUDE] Forward solve failed (non-finite solution) — cannot compute backsolve adjoint gradient. Returning zero gradient. Try a different random seed or smaller ωmax."
        dβ .= 0
        return
    end
    u_final = vec(_u)
    λ_final = [0.0, 0.0, 0.0]

    v₁ = [u_final; λ_final]

    # Definition of callback to introduce contrubution of loss function
    t_inverse = .-reverse(data.times)
    stop_condition(_, t, integrator) = t ∈ t_inverse
    function effect!(integrator)
        idxs = findall(t -> -t == integrator.t, data.times)
        if isnothing(data.kappas)
            κ = ones(length(idxs))
        else
            κ = data.kappas[idxs]
        end
        y = data.directions[:, idxs]
        if typeof(integrator.u) <: SVector
            integrator.u = vcat(
                integrator.u[1:3],
                SVector{3,Float64}(integrator.u[4:6] .- y * κ / (params.tmax - params.tmin))
                )
        else
            integrator.u[4:6] .+= .- y * κ / (params.tmax - params.tmin)
        end
    end
    cb_adjoint_loss = DiscreteCallback(stop_condition, effect!)
    @assert !params.weighted

    # Define ODE Problem with time in reverse
    if params.out_of_place

        v₁ = SVector{6,Float64}(v₁)

        rotation_adjoint(v, p, τ) = rotation_reverse(v, p, τ, regressor)

        rotation_adjoint_rev = ODEProblem{false}(
            rotation_adjoint,
            v₁,
            (- params.tmax, - params.tmin),
            β.θ
            )

    else

        rotation_adjoint!(dv, v, p, τ) = rotation_reverse!(dv, v, p, τ, regressor)

        rotation_adjoint_rev = ODEProblem{true}(
            rotation_adjoint!,
            v₁,
            (- params.tmax, - params.tmin),
            β.θ
            )
    end

    # Where the adjoint needs to be evaluated
    nodes, weights = extract_nodes_weights(params.tmin, params.tmax, params.quadrature)
    n_quadrature = length(nodes)
    t_nodes_rev = .- reverse(nodes)

    # Solve reverse adjoint PDE with dense output
    sol_adjoint = solve(
        rotation_adjoint_rev,
        callback = cb_adjoint_loss,
        saveat = t_nodes_rev,
        dense = false,
        save_everystep = false,
        tstops = t_inverse,
        sensealg.solver,
        # dtmax = sensealg.dtmax,
        reltol = sensealg.reltol,
        abstol = sensealg.abstol
        )

    if norm(params.u0 .- sol_adjoint(-params.tmin)[1:3]) > 1e-3
        @warn "Solution if backsolve adjoint differs from forward solve: ( u0 = $(params.u0) ) ≠ ( sol_adjoint = $(sol_adjoint(-params.tmin)[1:3]) )"
    end

    # Now we compute the contribution to the loss function.
    dLdθ = zeros(size(β.θ))
    for i in 1:n_quadrature
        t = nodes[i]
        τ = -t
        ∇θ = jacobian_params(regressor, t, β.θ)
        dLdθ .+= weights[i] * mapslices(
            x -> dot(sol_adjoint(τ)[4:6], cross(x, sol_adjoint(τ)[1:3])),
            ∇θ;
            dims = 1
        )[:]
    end

    dLdθ_cv = Vector2ComponentVector(dLdθ, β.θ)
    dβ.θ .= dLdθ_cv

    # Add regularization gradient (missing from the adjoint ODE above).
    # Zygote is used here: for SplineRegressor it automatically invokes the
    # analytical rrule; for NNRegressor it traces normally.
    if !isnothing(params.reg)
        for reg in params.reg
            if typeof(reg) <: Regularization
                grad_reg, = Zygote.gradient(θ -> regularization(θ, regressor, reg, params), β.θ)
                dβ.θ .+= grad_reg
            end
        end
    end
end

### Utils

"""
Dynamics for backpropagating adjoint (in-place)
"""
function rotation_reverse!(dv, v, p, τ, regressor::AbstractRegressor)
    t = -τ
    L = predict_L(t, regressor, p)
    dv[1:3] .= cross(-L, v[1:3])
    dv[4:6] .= cross(-L, v[4:6])
end

"""
Dynamics for backpropagating adjoint (out-of-place)
"""
function rotation_reverse(v, p, τ, regressor::AbstractRegressor)
    t = -τ
    L = predict_L(t, regressor, p)
    return SVector{6,Float64}([cross(-L, v[1:3]); cross(-L, v[4:6])])
end


"""
Dummy grad
"""
function rotation_grad!(
    dβ::ComponentVector,
    β::ComponentVector,
    data::AD,
    params::AP,
    regressor::AbstractRegressor,
    sensealg::SphereUDE.DummyAdjoint
    ) where {AD<:AbstractData,AP<:AbstractParameters}

    dβ .= (rand() - 0.5) .* β
end


function Vector2ComponentVector(v::Vector, cv_template::ComponentVector)
    cv = zero(cv_template)
    for i in 1:length(v)
        cv[i] = v[i]
    end
    return cv
end

function ComponentVector2Vector(cv::ComponentVector)
    return collect(cv)
end
