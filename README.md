# SphereUDE

[![Build Status](https://github.com/facusapienza21/SphereUDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/facusapienza21/SphereUDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/facusapienza21/SphereUDE.jl/graph/badge.svg?token=UC0KFSUU3X)](https://codecov.io/gh/facusapienza21/SphereUDE.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SphereUDE.jl is a Julia package for non-parametric regression of data supported in three-dimensional spheres. 
It implements a simple universal differential equation (UDE) that naturally constrains trajectories to lie on the surface of a sphere. 
This has an important application in Paleomagnetism, where the objective is to fit Apparent Polar Wander Paths (APWPs) to reconstruct continents' past motion. 
In addition to sphere regression, SphereUDE.jl implements a series of improvements over previous modeling methods, such as 
- Explicit sphere constraint that allows for univesality of regression 
- Regularization on the path to incorporate physical priors 
- Incorporation of temporal and spatial uncertainties
- Uncertainty quantification capabilities

## Usage

To train a model with new unobserved data, we need to define the _data_, _parameters_, and _regularization_ we want to use. 
Data are defined as 
```julia 
data = SphereData(times=times_samples, 
                  directions=X_true, 
                  kappas=nothing, 
                  L=L_true)

```
where `times` correspond to an array of the sampled times where we observed the three-dimensional vectors in `directions`. 
We can further add an array `kappa` to specify uncertainty in the directions according to the Fisher distribution in the sphere. 

It is possible to add different types of regularizations at the same time by specifying an array of the type `Regularization`, which specifies the type of regularization being used and the `diff_mode` that specifies the underlying automatic differentiation machinery being used to compute the gradients. 
```julia
regs = [Regularization(order=1, power=1.0, λ=0.001, diff_mode="Finite Differences"), 
        Regularization(order=0, power=2.0, λ=0.1, diff_mode="Finite Differences")]

```
Finally, the parameters include the regularization together with other customizable training parameters:
```julia
params = SphereParameters(tmin=0.0, tmax=100.0, 
                          reg=regs, 
                          u0=[0.0, 0.0, -1.0], ωmax=1.0, reltol=1e-12, abstol=1e-12,
                          niter_ADAM=1000, niter_LBFGS=600)
```

Training is finally being done with 
```julia
results = train(data, params, rng, nothing)
```
with `rng` a random seed used for the initial setup of the neural network. 

Here there is a simple [example](https://github.com/facusapienza21/SphereUDE.jl/blob/main/examples/double_rotation/double_rotation.jl) for the reconstruction of two solid rotations using `SphereUDE.jl`. 

## Installing SphereUDE

To install `SphereUDE` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.10) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add SphereUDE
```

## SphereUDE initialization: integration with Python

To make plots using Matplotlib, Cartopy, and PMagPy, we install both [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) and execute Python code directly from Julia. To do this setup manually, you can follow the next steps. 

- Create a Python conda environment, based on [this conda environment file](https://raw.githubusercontent.com/facusapienza21/SphereUDE.jl/main/environment.yml), with all the required packages using `conda env create -f environment.yml`.
- Inside the Julia REPL, install both `PyCall.jl` and `PyPlot.jl` with `] add PyCall, Pyplot`.
- Specify the Python path of the new environment with `ENV["PYTHON"] = ...`, where you should complete the path of the Python installation that shows when you do `conda activate SphereUDE`, `which python`. Inside the Julia REPL, execute `Pkg.build("PyCall")` to re-build PyCall with the new Python path: 
```
julia> ENV["PYTHON"] = read(`which python`, String)[1:end-1] # trim backspace
julia> import Pkg; Pkg.build("PyCall")
julia> exit()
```

You are ready to use Python from your Julia session!
