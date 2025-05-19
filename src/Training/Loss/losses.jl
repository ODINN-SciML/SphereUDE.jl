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
    U::Chain,
    st::NamedTuple,
) where {AD<:AbstractData,AP<:AbstractParameters}

    # Record the value of each individual loss to the total loss function for hyperparameter selection.
    loss_dict = Dict()

    l_emp = loss_empirical(β, data, params, U, st)

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
                reg₀ = regularization(β.θ, U, st, reg, params)
                loss_dict["Regularization (order=$(reg.order), power=$(reg.power))"] = reg₀
            elseif typeof(reg) <: CubicSplinesRegularization
                reg₀ = cubic_regularization(β, U, st, reg, params)
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
    U::Chain,
    st::NamedTuple,
) where {AD<:AbstractData,AP<:AbstractParameters}

    # Predict trajectory on times associated to dataset
    if data.repeat_times
        times = data.times_unique
        u_unique = predict(β, params, times, U, st)
        u_ = u_unique[:, data.times_unique_inverse]
    else
        times = data.times
        u_ = predict(β, params, times, U, st)
    end

    if params.weighted
        if data.repeat_times
            @error "Repeat times not implemented for weighted sum."
        end
        weigths = quadrature(
            params.tmin,
            params.tmax,
            times;
            method = :Linear
            ) ./ (params.tmax - params.tmin)
    else
        weigths = 1 / (params.tmax - params.tmin)
    end

    # Empirical error
    if isnothing(data.kappas)
        # @ignore_derivatives begin
        #     # TODO: I should ensure the output is directly norm one and avoid this test here
        #     @assert isapprox(2.0 * sum(weigths .* (1.0 .- sum(u_.* data.directions, dims = 1))), sum(weigths .* abs2.(u_ .- data.directions)), rtol = 1e-3)
        # end
        return sum(weigths .* (1.0 .- sum(u_ .* data.directions, dims = 1)))
    else
        return sum(weigths .* data.kappas .* (1.0 .- sum(u_ .* data.directions, dims = 1)))
        # @ignore_derivatives begin
        #     @assert isapprox(2.0 * sum(weigths .*  data.kappas .* (1.0 .- sum(u_.* data.directions, dims = 1))), sum(weigths .* data.kappas .* abs2.(u_ .- data.directions)), rtol = 1e-3)
        # end
    end

end

"""
Empirical Prediction function
"""
function predict(
    β::ComponentVector,
    params::AP,
    T::Vector,
    U::Chain,
    st::NamedTuple
    ) where {AP<:AbstractParameters}

    # Closure of the ODE update for solve
    if params.out_of_place
        ude_rotation_closure(u, p, t) = ude_rotation(u, p, t, U, st)
        prob_nn = ODEProblem(
            ude_rotation_closure,
            params.u0,
            [params.tmin, params.tmax],
            β.θ
            )
    else
        ude_rotation_closure!(du, u, p, t) = ude_rotation!(du, u, p, t, U, st)
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
        save_everystep = false,
        abstol = params.abstol,
        reltol = params.reltol,
        sensealg = params.sensealg,
        dtmin = 1e-6 * (params.tmax - params.tmin),
        force_dtmin = true,
    )

    if (typeof(params.sensealg) <: BacksolveAdjoint) & (!(params.tmax ≈ maximum(T)) | !(params.tmin ≈ minimum(T)))
        @warn "Backsolve adjoint requires to saveat initial and final time of the simulation"
    end

    # If numerical integration fails or bad choice of parameter, return infinity
    if sol.retcode != :Success
        @warn "[SphereUDE] Numerical solver not converging. This can be causes by numerical innestabilities around a bad choice of parameter. This can be due to just a bad initial condition of the neural network, so it is worth changing the randon number used for initialization. "
        return Inf
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
