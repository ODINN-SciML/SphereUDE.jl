export loss
export loss_empirical
export regularization
export cubic_regularization
export loss_multiple_shooting

"""
General Loss Function
"""
function loss(β::ComponentVector, 
              data::AD, 
              params::AP,
              U::Chain, 
              st::NamedTuple) where {AD <: AbstractData, AP <: AbstractParameters}
    
    # Record the value of each individual loss to the total loss function for hyperparameter selection.
    loss_dict = Dict()

    l_emp = loss_empirical(β, data, params)

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
function loss_empirical(β::ComponentVector, 
                        data::AD, 
                        params::AP) where {AD <: AbstractData, AP <: AbstractParameters}

    # Predict trajectory on times associated to dataset
    u_ = predict(β, params, data.times)

    # Empirical error
    if isnothing(data.kappas)
        # The 3 is needed since the mean is computen on a 3xN matrix
        return 3.0 * mean(abs2.(u_ .- data.directions))
        # l_emp = 1 - 3.0 * mean(u_ .* data.directions)
    else
        return mean(data.kappas .* sum(abs2.(u_ .- data.directions), dims=1))
        # l_emp = norm(data.kappas)^2 - 3.0 * mean(data.kappas .* u_ .* data.directions)
    end

end

"""
Empirical Prediction function
"""
function predict(β::ComponentVector,
                 params::AP,
                 T::Vector) where {AP <: AbstractParameters} 
    
    if params.train_initial_condition
        _prob = remake(prob_nn, u0=β.u0 / norm(β.u0), # We enforce the norm=1 condition again here
                       tspan=(min(T[1], params.tmin), max(T[end], params.tmax)), 
                       p = β.θ)
    else
        _prob = remake(prob_nn, u0=params.u0, 
                      tspan=(min(T[1], params.tmin), max(T[end], params.tmax)), 
                      p = β.θ)
    end

    # Force minimum step in case L(t) changes drastically due to bad behaviour of neural network
    sol = solve(_prob, params.solver, saveat=T,
                abstol=params.abstol, reltol=params.reltol,
                sensealg=params.sensealg, 
                dtmin=1e-4 * (params.tmax - params.tmin), force_dtmin=true) 
    
    # If numerical integration fails or bad choice of parameter, return infinity
    if sol.retcode != :Success
        @warn "[SphereUDE] Numerical solver not converging. This can be causes by numerical innestabilities around a bad choice of parameter. This can be due to just a bad initial condition of the neural network, so it is worth changing the randon number used for initialization. "
        return Inf
    end
    
    return Array(sol)
end

"""
Regularization
"""
function regularization(θ::ComponentVector, 
                        U::Chain,
                        st::NamedTuple,
                        reg::Regularization,
                        params::AP) where {AP <: AbstractParameters}
        
    # Define Statefull Lux NN
    smodel = StatefulLuxLayer{true}(U, θ, st)

    l_ = 0.0
    if reg.order==0
        
        # l_ += quadrature(t -> norm(predict_L(t, U, θ, st))^reg.power, params.tmin, params.tmax, params.n_quadrature)
        l_ += quadrature(t -> norm(smodel([t]))^reg.power, params.tmin, params.tmax, params.n_quadrature)

    elseif reg.order==1
        
        if typeof(reg.diff_mode) <: LuxNestedAD
            # Automatic Differentiation 
            nodes, weights = quadrature(params.tmin, params.tmax, params.n_quadrature)

            if reg.diff_mode.method == "ForwardDiff"
                # Jac = ForwardDiff.jacobian(smodel, reshape(nodes, 1, params.n_quadrature))
                Jac = batched_jacobian(smodel, AutoForwardDiff(), reshape(nodes, 1, params.n_quadrature))
            elseif reg.diff_mode.method == "Zygote"
                # This can also be done with Zygote in reverse mode
                # Jac = Zygote.jacobian(smodel, reshape(nodes, 1, params.n_quadrature))[1]
                Jac = batched_jacobian(smodel, AutoZygote(), reshape(nodes, 1, params.n_quadrature))
            else
                throw("Method for AD backend no implemented.")
            end

            # Compute the final agregation to the loss
            l_ += sum([weights[j] * norm(Jac[:,1,j])^reg.power for j in 1:params.n_quadrature])
        
            # Test every a few iterations that AD is working properly
            ignore() do
                if rand(Bernoulli(0.001))
                    l_AD = sum([weights[j] * norm(Jac[:,1,j])^reg.power for j in 1:params.n_quadrature])
                    l_FD = quadrature(t -> norm(central_fdm(τ -> smodel([τ]), t, 1e-5))^reg.power, params.tmin, params.tmax, params.n_quadrature)
                    if abs(l_AD - l_FD) > 1e-2 * abs(l_FD) 
                        @warn "[SphereUDE] Nested AD is giving significant different results than Finite Differences."
                        @printf "[SphereUDE] Regularization with AD: %.9f vs %.9f using Finite Differences" l_AD l_FD 
                    end
                end
            end

        elseif typeof(reg.diff_mode) <: FiniteDifferences
            # Finite differences 
            # l_ += quadrature(t -> norm(central_fdm(τ -> predict_L(τ, U, θ, st), t, reg.diff_mode.ϵ))^reg.power, params.tmin, params.tmax, params.n_quadrature)
            l_ += quadrature(t -> norm(central_fdm(τ -> smodel([τ]), t, reg.diff_mode.ϵ))^reg.power, params.tmin, params.tmax, params.n_quadrature)

        elseif typeof(reg.diff_mode) <: ComplexStepDifferentiation
            # Complex step differentiation
            # l_ += quadrature(t -> norm(complex_step_differentiation(τ -> predict_L(τ, U, θ, st), t, reg.diff_mode.ϵ))^reg.power, params.tmin, params.tmax, params.n_quadrature) 
            l_ += quadrature(t -> norm(complex_step_differentiation(τ -> smodel([τ]), t, reg.diff_mode.ϵ))^reg.power, params.tmin, params.tmax, params.n_quadrature) 

        else
            throw("Method not implemented.")
        end
    
    else
        throw("Method not implemented.")
    end

    return reg.λ * l_
end

"""
Cubic regularization from Jupp (1987)
"""
function cubic_regularization(β::ComponentVector, 
                              U::Chain,
                              st::NamedTuple,
                              reg::CubicSplinesRegularization,
                              params::AP) where {AP <: AbstractParameters}

    smodel = StatefulLuxLayer{true}(U, β.θ, st)

    # Create prediction of solution time series in integration points
    nodes, weights = quadrature(params.tmin, params.tmax, params.n_quadrature)
    u_ = predict(β, params, nodes)

    if typeof(reg.diff_mode) <: LuxNestedAD
        # Automatic Differentiation 
        
        if reg.diff_mode.method == "ForwardDiff"
            Jac = batched_jacobian(smodel, AutoForwardDiff(), reshape(nodes, 1, params.n_quadrature))
        elseif reg.diff_mode.method == "Zygote"
            Jac = batched_jacobian(smodel, AutoZygote(), reshape(nodes, 1, params.n_quadrature))
        else
            throw("Method for AD backend no implemented.")
        end

        L_cross_u = [cross(Jac[:,1,j], u_[:,j]) for j in 1:params.n_quadrature]
    
    elseif typeof(reg.diff_mode) <: FiniteDifferences     
        L_ = [central_fdm(τ -> smodel([τ]), t, reg.diff_mode.ϵ) for t in nodes]
        L_cross_u = [cross(L_[j], u_[:,j]) for j in 1:params.n_quadrature]
    
    elseif typeof(reg.diff_mode) <: ComplexStepDifferentiation
        L_ = [complex_step_differentiation(τ -> smodel([τ]), t, reg.diff_mode.ϵ) for t in nodes]
        L_cross_u = [cross(L_[j], u_[:,j]) for j in 1:params.n_quadrature]
    
    else
        throw("Method not implemented.")
    end

    return reg.λ * sum([weights[j] * norm(L_cross_u[j])^2.0 for j in 1:params.n_quadrature])
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