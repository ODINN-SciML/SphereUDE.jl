[![Build Status](https://github.com/ODINN-SciML/SphereUDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/SphereUDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ODINN-SciML/SphereUDE.jl/graph/badge.svg?token=UC0KFSUU3X)](https://codecov.io/gh/ODINN-SciML/SphereUDE.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/735154231.svg)](https://doi.org/10.5281/zenodo.15165465)

<img src="https://github.com/ODINN-SciML/SphereUDE.jl/blob/main/plots/logo.png?raw=true" width="350">

## ⚠️ New publication availbale available! 📖 ⚠️

The paper is associated to this package is now available at the [Journal of Geophysical Research: Machine Learning and Computation](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JH000626)

---

SphereUDE.jl is a Julia package for non-parametric regression of data supported in three-dimensional spheres. 
It implements a simple universal differential equation (UDE) that naturally constrains trajectories to lie on the surface of a sphere. 
This has an important application in Paleomagnetism, where the objective is to fit Apparent Polar Wander Paths (APWPs) to reconstruct continents' past motion. 
In addition to sphere regression, SphereUDE.jl implements a series of improvements over previous modeling methods, such as 
- Explicit sphere constraint that allows for universality of regression 
- Regularization on the path to incorporate physical priors

We are further working in new features such as
- Incorporation of temporal and spatial uncertainties
- Uncertainty quantification capabilities

## Installing SphereUDE

`SphereUDE.jl` is available through the Julia package manager. 
To install `SphereUDE` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.10) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add SphereUDE
```

## Usage

If you are interested in using `SphereUDE.jl`, we encourage you to take a look at our gallery of examples. 
Examples are included in the repository [ODINN-SciML/SphereUDE-examples](https://github.com/ODINN-SciML/SphereUDE-examples).

To train a model with new unobserved data, we need to define the _data_, _parameters_, and _regularization_ we want to use. 
Data objecs are defined with the `SphereData` construct, which takes  
```julia 
data = SphereData(times=times, directions=X, kappas=kappas, L=nothing)
```
where `times` correspond to an array of the sampled times where we observed the three-dimensional vectors in `directions`. 
We can further add an array `kappa` to specify uncertainty in the directions according to the Fisher distribution in the sphere. 

It is possible to add different types of regularizations at the same time by specifying an array of the type `Regularization`, which specifies the type of regularization being used and the `diff_mode` that specifies the underlying automatic differentiation machinery being used to compute the gradients. 
For example, we can add regularization using finite differences as 
```julia
reg = [Regularization(order=1, power=2.0, λ=1e5, diff_mode=LuxNestedAD())]
```
Finally, the parameters include the regularization together with other customizable training parameters:
```julia
params = SphereParameters(tmin = 0, tmax = 100, 
                          reg = reg,
                          pretrain = false, 
                          u0 = [0.0, 0.0, -1.0], ωmax = 2.0, 
                          reltol = 1e-6, abstol = 1e-6,
                          niter_ADAM = 5000, niter_LBFGS = 5000, 
                          sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))) 
```
Training is finally done with 
```julia
using Random
rng = Random.default_rng()
Random.seed!(rng, 613)

results = train(data, params, rng, nothing)
```
with `rng` a random seed used for the initial setup of the neural network. 

The architecture of the neural network can be customized and passed directy before training as follows: 
```julia
init_bias(rng, in_dims) = LinRange(tspan[1], tspan[2], in_dims)
init_weight(rng, out_dims, in_dims) = 0.1 * ones(out_dims, in_dims)

# Customized neural network to similate weighted moving window in L
U = Lux.Chain(
    Lux.Dense(1, 200, rbf, init_bias=init_bias, init_weight=init_weight, use_bias=true),
    Lux.Dense(200,10, gelu),
    Lux.Dense(10, 3, Base.Fix2(sigmoid_cap, params.ωmax), use_bias=false)
)

results = train(data, params, rng, nothing, U)
```
We can finally save and plot the data: 
```julia 
results_dict = convert2dict(data, results)

JLD2.@save "./results_dict.jld2" results_dict

plot_sphere(data, results, -30., 0., saveas="plot.png", title="Results")
```

:books: We are working in a more complete documentation. Feel free to reach out in the meantime if you have any questions! 

## SphereUDE integration with Python

To make plots using Matplotlib, Cartopy, and PMagPy, `SphereUDE.jl` uses `CondaPkg.jl` and `PythonCall.jl` to execute Python code directly from Julia.
This is done automatically once you install `SphereUDE.jl`, meaning that the first time you install `SphereUDE.jl` a conda environment linked to your Julia environmnet will be created and 
linked to your Julia session. 
From there, `SphereUDE.jl` will internally call the Python libraries without any further action from the user's side. 

## Contribute to the project! :wave:

We encourage you to contribute to this package. If you are interested in contributing, there are many ways in which you can help build this package:
- :collision: **Report bugs in the code.** You can report problems with the code by oppening issues under the `Issues` tab in this repository. Please explain the problem you encounter and try to give a complete description of it so we can follow up on that.
- :bulb: **Request new features and explanations.** If there is an important topic or example that you feel falls under the scope of this package and you would like us to include it, please request it! We are looking for new insights into what the community wants to learn.

## Contact 

If you have any questions or want to reach out, feel free to send us an email to `sapienza@stanford.edu`.
