export predict_L
export ude_rotation!
export callback_proj

"""
    function predict_L(t, NN, θ, st)

Function to predict angular momentum. Used just for ODE part 
of the loss function
"""
function predict_L(t, NN, θ, st)
    smodel = StatefulLuxLayer{true}(NN, θ, st)
    return smodel([t])
end

"""
    function ude_rotation!(du, u, p, t)

Sphere-constrained ODE
"""
function ude_rotation!(du, u, p, t, U, st)
    # Angular momentum given by network prediction
    L = predict_L(t, U, p, st)
    du .= cross(L, u)
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
