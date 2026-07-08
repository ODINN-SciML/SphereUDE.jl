export Regularization, AbstractRegularization
export CubicSplinesRegularization
export RegularizationCV

abstract type AbstractRegularization end

"""
Regularization information
"""
@kwdef struct Regularization{F<:AbstractFloat,I<:Int} <: AbstractRegularization
    order::I        # Order of derivative
    power::F        # Power of the Euclidean norm 
    λ::F            # Regularization hyperparameter
    # AD differentiation mode used in regulatization
    diff_mode::Union{Nothing,AbstractDifferentiation} = nothing

    # Include this in the constructor
    # @assert (order == 0) || (!isnothing(diff_mode)) "Diffentiation methods needs to be provided for regularization with order larger than zero." 
end

# Display setup
function Base.show(io::IO, reg::Regularization)
    print(io)
    if isnothing(reg)
        print("No Regularization")
    else
        print(" ")
        print("Regularization of the form ")
        printstyled(@sprintf("%9.2e", reg.λ); color = :blue)
        print(" x ")
        if reg.order == 0
            print("∫|L(t)|^$(reg.power) dt")
        elseif reg.order == 1
            print("∫|dL(t)/dt|^$(reg.power) dt")
            print("  ")
            print("Differentiation method: $(reg.diff_mode)")
        else
            raise("Regularization not implemented.")
        end
    end
    println("")
end
# Vectorial form
# function Base.show(io::IO, regs::Vector{Regularization})
#     print(io)
#     for reg in regs
#         print(reg)
#     end
# end

@kwdef struct CubicSplinesRegularization{F<:AbstractFloat} <: AbstractRegularization
    λ::F
    diff_mode::Union{Nothing,AbstractDifferentiation} = nothing
end

"""
Same as [`Regularization`](@ref), except `λ` is a vector of candidate values
to cross-validate over instead of a single scalar. Used to mark a
regularization term whose strength should be selected by cross-validation
(see `Training/Validation/crossvalidation.jl`) rather than fixed by the caller.
"""
@kwdef struct RegularizationCV{F<:AbstractFloat,I<:Int} <: AbstractRegularization
    order::I        # Order of derivative
    power::F        # Power of the Euclidean norm
    λ::Vector{F}    # Candidate regularization hyperparameters to cross-validate over
    diff_mode::Union{Nothing,AbstractDifferentiation} = nothing
end
