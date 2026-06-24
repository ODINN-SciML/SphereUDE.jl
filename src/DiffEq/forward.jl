export predict_L
export ude_rotation!
export callback_proj

"""
    predict_L(t, regressor, θ)

Predict angular velocity L at time t using the regressor and parameters θ.
"""
function predict_L(t, regressor::AbstractRegressor, θ)
    return regressor(t, θ)
end

"""
    ude_rotation!(du, u, p, t, regressor)

Sphere-constrained ODE (in-place).
"""
function ude_rotation!(du, u, p, t, regressor::AbstractRegressor)
    L = predict_L(t, regressor, p)
    du .= cross(L, u)
end

"""
    ude_rotation(u, p, t, regressor)

Sphere-constrained ODE (out-of-place).
"""
function ude_rotation(u, p, t, regressor::AbstractRegressor)
    L = predict_L(t, regressor, p)
    return SVector{3,Float64}(cross(L, u))
end

"""
    callback_proj(p, l, params)

Callback function to project solution in unit sphere.
"""
function callback_proj(p, l, params)
    if params.train_initial_condition
        p.u.u0 ./= norm(p.u.u0)
    end
    return false
end
