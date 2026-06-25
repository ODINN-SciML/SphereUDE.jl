export loss
export loss_empirical
export loss_multiple_shooting

"""
General Loss Function
"""
function loss(
    β::ComponentVector,
    data::AD,
    params::AP,
    regressor::AbstractRegressor,
) where {AD<:AbstractData,AP<:AbstractParameters}

    # Record the value of each individual loss to the total loss function for hyperparameter selection.
    loss_dict = Dict()

    l_emp = loss_empirical(β, data, params, regressor)

    loss_dict["Empirical"] = l_emp

    if params.hyperparameter_balance
        # @ignore_derivatives l_emp /= l_emp
        l_emp = log(l_emp)
    end

    # Regularization
    l_reg = 0.0
    if !isnothing(params.reg)
        for reg in params.reg
            if typeof(reg) <: Regularization
                reg₀ = regularization(β.θ, regressor, reg, params)
                loss_dict["Regularization (order=$(reg.order), power=$(reg.power))"] = reg₀
            elseif typeof(reg) <: CubicSplinesRegularization
                reg₀ = cubic_regularization(β, regressor, reg, params)
                loss_dict["Regularization with cubic splines"] = reg₀
            else
                throw("Regularization not implemented.")
            end
            # Add contribution to regularization
            if params.hyperparameter_balance
                reg₀ = log(reg₀)
                # @ignore_derivatives reg₀ /= reg₀
            end
            l_reg += reg₀
        end
    end
    return l_emp + l_reg, loss_dict
end

"""
Empirical loss function
"""
function loss_empirical(
    β::ComponentVector,
    data::AD,
    params::AP,
    regressor::AbstractRegressor,
) where {AD<:AbstractData,AP<:AbstractParameters}

    # Size of dataset
    @assert size(data.directions)[1] == 3
    N = size(data.directions)[2]

    # Predict trajectory on times associated to dataset
    if data.repeat_times
        times = data.times_unique
        u_unique = predict(β, params, times, regressor)
        u_ = u_unique[:, data.times_unique_inverse]
    else
        times = data.times
        u_ = predict(β, params, times, regressor)
    end

    if params.weighted
        if data.repeat_times
            @error "Repeat times not implemented for weighted sum."
        end
        weights = quadrature(
            params.tmin,
            params.tmax,
            times;
            method = :Linear
            ) ./ (params.tmax - params.tmin)
    else
        weights = 1 / (N * (params.tmax - params.tmin))
    end

    # Per-point residual (1 - cosθ_i), flattened to a length-N vector. `vec`
    # is essential here: `sum(..., dims=1)` keeps a (1,N) shape, and
    # broadcasting that directly against a length-N Vector (e.g. data.kappas
    # below) would silently compute an N×N outer product instead of pairing
    # each kappa_i with its own point's residual.
    residual = vec(1.0 .- sum(u_ .* data.directions, dims = 1))

    # Empirical error
    if isnothing(data.kappas)
        return sum(weights .* residual)
    else
        return sum(weights .* data.kappas .* residual)
    end

end

"""
Empirical Prediction function
"""
function predict(
    β::ComponentVector,
    params::AP,
    T::Vector,
    regressor::AbstractRegressor,
    ) where {AP<:AbstractParameters}

    # Closure of the ODE update for solve
    if params.out_of_place
        ude_rotation_closure(u, p, t) = ude_rotation(u, p, t, regressor)
        prob_nn = ODEProblem(
            ude_rotation_closure,
            params.u0,
            [params.tmin, params.tmax],
            β.θ
            )
    else
        ude_rotation_closure!(du, u, p, t) = ude_rotation!(du, u, p, t, regressor)
        prob_nn = ODEProblem(
            ude_rotation_closure!,
            params.u0,
            [params.tmin, params.tmax],
            β.θ
            )
    end

    if params.train_initial_condition
        _prob = remake(
            prob_nn,
            u0 = β.u0 / norm(β.u0), # We enforce the norm=1 condition again here
            tspan = (min(T[1], params.tmin), max(T[end], params.tmax)),
            p = β.θ,
        )
    else
        _prob = remake(
            prob_nn,
            u0 = params.u0,
            tspan = (min(T[1], params.tmin), max(T[end], params.tmax)),
            p = β.θ,
        )
    end

    # Force minimum step in case L(t) changes drastically due to bad behaviour of neural network
    sol = solve(
        _prob,
        params.solver,
        saveat = T,
        # Do not set save_everystep to any value. This needs to be free fro the sensitivity
        # method to work properly.
        # save_everystep = save_everystep,
        abstol = params.abstol,
        reltol = params.reltol,
        sensealg = params.sensealg,
        dtmin = 1e-6 * (params.tmax - params.tmin),
        force_dtmin = true,
    )

    if (typeof(params.sensealg) <: BacksolveAdjoint) & (!(params.tmax ≈ maximum(T)) | !(params.tmin ≈ minimum(T)))
        @warn "Backsolve adjoint requires to saveat initial and final time of the simulation"
    end

    # If numerical integration fails or bad choice of parameter, throw an error
    if sol.retcode != ReturnCode.Success
        error("""
        [SphereUDE] ODE solver failed with retcode $(sol.retcode).

        This is usually caused by one of the following:
          • A bad neural network initialization producing extreme angular velocities.
            → Try a different random seed or reduce ωmax in SphereParameters.
          • Tolerances (reltol/abstol) that are too tight for the chosen solver.
            → Try relaxing tolerances or switching to a stiffer solver.
          • Numerical instability in the ODE right-hand side during early training
            when NN weights are still far from a good solution.

        Parameters: tmin=$(params.tmin), tmax=$(params.tmax), solver=$(typeof(params.solver)), reltol=$(params.reltol), abstol=$(params.abstol)
        """)
    end

    return Array(sol)
end

"""
Loss function to be called for multiple shooting
This seems duplicated from before, so be careful with this
"""
function loss_multiple_shooting(data, pred)

    throw("[SphereUDE] Method to be implemented.")
    # # Empirical error
    # l_emp = 3.0 * mean(abs2.(pred .- data))

    # # Regularization
    # l_reg = 0.0
    # if !isnothing(params.reg)
    #     for reg in params.reg    
    #         reg₀ = regularization(β.θ, reg)
    #         l_reg += reg₀      
    #     end 
    # end

    # return l_emp + l_reg
    # end

    # # Define parameters for Multiple Shooting
    # group_size = 10
    # continuity_term = 100

    # ps = ComponentArray(θ) # are these necesary?
    # pd, pax = getdata(ps), getaxes(ps) 

    # function continuity_loss(u_pred, u_initial)
    # if !isapprox(norm(u_initial), 1.0, atol=1e-6) || !isapprox(norm(u_pred), 1.0, atol=1e-6)
    #     @warn "Directions during multiple shooting are not in the sphere. Small deviations from unit norm observed:" 
    #     @show norm(u_initial), norm(u_pred)
    # end
    # return sum(abs2, u_pred - u_initial)
    # end

    # function loss_multiple_shooting(β::ComponentVector)

    # ps = ComponentArray(β.θ, pax)

    # if params.train_initial_condition
    #     _prob = remake(prob_nn, u0=β.u0 / norm(β.u0), # We enforce the norm=1 condition again here
    #                 tspan=(min(data.times[1], params.tmin), max(data.times[end], params.tmax)), 
    #                 p = β.θ)
    # else
    #     _prob = remake(prob_nn, u0=params.u0, 
    #                 tspan=(min(data.times[1], params.tmin), max(data.times[end], params.tmax)), 
    #                 p = β.θ)
    # end

    # return multiple_shoot(β.θ, data.directions, data.times, _prob, _loss_multiple_shooting, continuity_loss, params.solver,
    #                         group_size; continuity_term, sensealg=params.sensealg)


end
