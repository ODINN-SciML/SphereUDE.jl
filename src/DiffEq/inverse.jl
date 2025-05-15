# Implementation of inverse models


"""
Manual gradient computation using backsolve adjoint
"""
function rotation_grad!(
    dβ::ComponentVector,
    β::ComponentVector,
    data::AD,
    params::AP,
    U::Chain,
    st::NamedTuple,
    sensealg::SphereBackSolveAdjoint,
    ) where {AD<:AbstractData, AP<:AbstractParameters}

    # Compute final solution of forward model
    t₀, t₁ = params.tmin, params.tmax

    u_final = predict(β, params, [t₁], U, st)
    λ_final = [0.0, 0.0, 0.0]
    # dLdβ_final = 0.0

    v₁ = [u_final; λ_final]

    """
    Dynamics for backpropagating adjoint
    """
    function rotation_reverse(dv, v, p, τ)
        t = -τ
        L = predict_L(t, U, β.θ, st)
        dv[1:3] .= cross(-L, v[1:3])
        dv[4:6] .= cross(-L, v[4:6])
    end

    # Definition of callback to introduce contrubution of loss function
    t_inverse = .-reverse(data.times)
    stop_condition(U, t, integrator) = t ∈ t_inverse
    @assert !params.weighted
    function effect!(integrator)
        idx = only(findall(t -> -t == integrator.t, data.times))
        if isnothing(data.kappas)
            κ = 1.0
        else
            κ = data.kappas[idx]
        end
        y = data.directions[:, idx]
        # TODO: This will not work with repeated times
        integrator.u[4:6] .+= .- κ * y / (params.tmax - params.tmin)
    end
    cb_adjoint_loss = DiscreteCallback(stop_condition, effect!)

    # Define ODE Problem with time in reverse
    rotation_adjoint_rev = ODEProblem(
        rotation_reverse,
        v₁,
        (- params.tmax, - params.tmin)
        )

    # Solve reverse adjoint PDE with dense output
    sol_adjoint = solve(
        rotation_adjoint_rev,
        callback = cb_adjoint_loss,
        # saveat=t_nodes_rev, # dont use this!
        dense = true,
        save_everystep = true,
        tstops = t_inverse,
        sensealg.solver,
        # dtmax = sensealg.dtmax,
        reltol = sensealg.reltol,
        abstol = sensealg.abstol
        )
        
    # TODO: solve this warning!
    @assert norm(params.u0 .- sol_adjoint.u[end][1:3]) < 1e-5

    # Now we compute the contribution to the loss function.
    nodes, weigths = quadrature(params.tmin, params.tmax, sensealg.n_quadrature)

    # dLdβ = zero(β)
    dLdθ = zeros(size(β.θ))
    for i in 1:sensealg.n_quadrature
        t = nodes[i]
        τ = -t
        ∇θ, = Zygote.jacobian(
            _θ -> StatefulLuxLayer{true}(U, _θ, st)([t]),
            β.θ
        )
        dLdθ .+= weigths[i] * mapslices(
            x -> dot(sol_adjoint(τ)[4:6], cross(x, sol_adjoint(τ)[1:3])),
            ∇θ;
            dims = 1
        )[:]
    end


    dLdθ_cv = Vector2ComponentVector(dLdθ, β.θ)
    dβ.θ .= dLdθ_cv
end


"""
Dummy grad
"""
function rotation_grad!(
    dβ::ComponentVector,
    β::ComponentVector,
    data::AD,
    params::AP,
    U::Chain,
    st::NamedTuple,
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