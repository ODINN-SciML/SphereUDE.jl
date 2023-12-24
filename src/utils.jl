export sigmoid_cap, relu_cap

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




