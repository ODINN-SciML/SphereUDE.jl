export GPRegressor

include("GP_utils.jl")

"""
    
"""
struct GPRegressor{AK<:AbstractKernel, I<:Int, F<:AbstractFloat} <: AbstractRegressor
    kernel::AK
    n_basis::I
    ωmax::F
end

### Create defaul constructor