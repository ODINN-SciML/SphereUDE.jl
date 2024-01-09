export sigmoid_cap, relu_cap
export cart2sph

# Normalization of the NN. Ideally we want to do this with L2 norm .

"""
    sigmoid_cap(x; ω₀)
"""
function sigmoid_cap(x; ω₀)
    min_value = - 2ω₀
    max_value = + 2ω₀
    return min_value + (max_value - min_value) / ( 1.0 + exp(-x) )
end

function relu_cap(x; ω₀)
    min_value = - 2ω₀
    max_value = + 2ω₀
    return min_value + (max_value - min_value) * max(0.0, min(x, 1.0))
end

"""
Convert cartesian coordinates to spherical
"""
function cart2sph(X::AbstractArray{<:Number}; radians=true)
    @assert size(X)[1] == 3 "Input array must have three rows."
    Y = mapslices(x -> [angle(x[1] + x[2]*im), asin(x[3])] , X, dims=1)
    if !radians
        Y *= 180. / π
    end
    return Y
end

