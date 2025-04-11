export predict_L
export ude_rotation!
export callback_proj

"""
    function predict_L(t, NN, θ, st)

Function to predict angular momentum. Used just for ODE part 
of the loss function
"""
function predict_L(t::Real, NN::Chain, θ::AbstractArray, st::NamedTuple)
    smodel = StatefulLuxLayer{true}(NN, θ, st)
    return smodel([t])
end

"""
    function ude_rotation!(du, u, p, t)

Sphere-constrained ODE
"""
function ude_rotation!(du::AbstractVector, u::AbstractVector, p::AbstractArray, t::Real, U::Chain, st::NamedTuple) 
    # Angular momentum given by network prediction
    L = predict_L(t, U, p, st)
    du .= cross(L, u)
end

"""
    callback_proj(p, l, params)

Callback function to project solution in unit sphere.
"""
function callback_proj(p, l, params::AP) where {AP<:AbstractParameters}
    if params.train_initial_condition
        p.u.u0 ./= norm(p.u.u0)
    end
    return false
end
