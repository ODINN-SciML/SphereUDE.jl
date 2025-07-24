export extract_nodes_weights

function extract_nodes_weights(
    t₀::F,
    t₁::F,
    quadrature::GaussQuadrature
) where {F<:AbstractFloat}

    # Extract nodes and weightds for numerical integration in [0,1]
    @ignore_derivatives nodes, weights = gausslegendre(quadrature.n_nodes)
    # Rescale nodes and weights to desired interval
    nodes = (t₀ + t₁) / 2 .+ nodes * (t₁ - t₀) / 2
    weights = (t₁ - t₀) / 2 * weights

    return nodes, weights
end

function extract_nodes_weights(
    t₀::F,
    t₁::F,
    quadrature::RandomGaussQuadrature
) where {F<:AbstractFloat}
    n_nodes = rand(quadrature.n_nodes_min:quadrature.n_nodes_max)
    return extract_nodes_weights(t₀, t₁, GaussQuadrature(n_nodes = n_nodes))
end

function extract_nodes_weights(
    t₀::F,
    t₁::F,
    quadrature::RandomQuadrature
) where {F<:AbstractFloat}

    nodes = rand(Uniform(t₀, t₁), quadrature.n_nodes)
    weights = repeat([(t₁ - t₀) / quadrature.n_nodes], quadrature.n_nodes)

    return nodes, weights
end

function extract_nodes_weights(
    t₀::F,
    t₁::F,
    quadrature::CustomQuadrature
) where {F<:AbstractFloat}

    nodes = quadrature.nodes
    n = length(nodes)

    @assert t₀ <= minimum(nodes) <= maximum(nodes) <= t₁

    if !isnothing(quadrature.weights)
        weights = quadrature.weights
    else
        if quadrature.interpolation_method == :Vandermonde
            """
            This works quite well for numerical integration but it has the problem that
            returns (potentially) negative weights, which are not suitable for a loss
            function.
            """
            # Build Vandermonde matrix
            V = [nodes[i]^j for j in 0:n-1, i in 1:n]
            # Right-hand side: exact integrals of monomials
            b = [(t₁^(k + 1) - t₀^(k + 1)) / (k + 1) for k in 0:n-1]
            # solve for the weights
            weights = V \ b
        elseif quadrature.interpolation_method == :Linear
            midpoints = (nodes[1:end-1] .+ nodes[2:end] ) ./ 2.0
            edges = [t₀; midpoints; t₁]
            weights = (edges[2:end] .- edges[1:end-1])
            @assert sum(weights) ≈ t₁ - t₀
        else
            @error "Method $(method) not implemented."
        end
    end

    return nodes, weights
end
