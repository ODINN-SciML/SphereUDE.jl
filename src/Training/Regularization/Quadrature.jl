export GaussQuadrature, RandomGaussQuadrature
export RandomQuadrature, CustomNodesQuadrature
export AbstractQuadrature

abstract type AbstractQuadrature end

@kwdef struct GaussQuadrature{I<:Int} <: AbstractQuadrature
    n_nodes::I
end

@kwdef struct RandomGaussQuadrature{I<:Int} <: AbstractQuadrature
    n_nodes_min::I
    n_nodes_max::I
end

@kwdef struct RandomQuadrature{I<:Int} <: AbstractQuadrature
    n_nodes::I
end

@kwdef struct CustomQuadrature{F<:AbstractFloat} <: AbstractQuadrature
    nodes::Vector{F}
    weights::Union{Nothing,Vector{F}} = nothing
    interpolation_method::Symbol = :Linear
end