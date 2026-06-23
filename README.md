[![Build Status](https://github.com/ODINN-SciML/SphereUDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/ODINN-SciML/SphereUDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/ODINN-SciML/SphereUDE.jl/graph/badge.svg?token=UC0KFSUU3X)](https://codecov.io/gh/ODINN-SciML/SphereUDE.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/735154231.svg)](https://doi.org/10.5281/zenodo.15165465)

<img src="https://github.com/ODINN-SciML/SphereUDE.jl/blob/main/plots/logo.png?raw=true" width="350">

## ⚠️ New publication available! 📖 ⚠️

The paper associated to this package is now available at the [Journal of Geophysical Research: Machine Learning and Computation](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2025JH000626)

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
Examples are included in the repository [ODINN-SciML/SphereUDE-examples](https://github.com/ODINN-SciML/SphereUDE-examples), including a full Gondwana paleomagnetic reconstruction.

To train a model with new unobserved data, we need to define the _data_, _parameters_, the _regressor_ that models the angular velocity, and (optionally) the _regularization_ we want to use. 

### Data

Data objects are defined with the `SphereData` construct, which takes  
```julia 
data = SphereData(times=times, directions=X, kappas=kappas, L=nothing)
```
where `times` correspond to an array of the sampled times where we observed the three-dimensional vectors in `directions`. 
We can further add an array `kappas` to specify uncertainty in the directions according to the Fisher distribution in the sphere, or pass `kappas=nothing` if unknown. 

### Regularization

It is possible to add different types of regularizations at the same time by specifying an array of the type `Regularization`, which specifies the order of the derivative being penalized, the power of the norm, the regularization strength `λ`, and (for neural-network regressors) the `diff_mode` used to differentiate the regressor. 
For example, we can add a first-order regularization using finite differences as 
```julia
reg = [Regularization(order=1, power=2.0, λ=1e5, diff_mode=FiniteDiff(1e-6))]
```

### Parameters

The parameters include the regularization together with other customizable training settings, such as the time span, the initial condition `u0`, the maximum angular velocity `ωmax`, the tolerances of the ODE solver, the number of iterations, and the sensitivity algorithm used to differentiate through the solve:
```julia
using SciMLSensitivity

params = SphereParameters(tmin = 0.0, tmax = 100.0, 
                          reg = reg,
                          pretrain = false, 
                          u0 = [0.0, 0.0, -1.0], ωmax = 2.0, 
                          reltol = 1e-6, abstol = 1e-6,
                          niter_ADAM = 5000, niter_LBFGS = 5000, 
                          sensealg = InterpolatingAdjoint(autojacvec = ReverseDiffVJP(true))) 
```
For larger problems, the custom `SphereBackSolveAdjoint(reltol=1e-8, abstol=1e-8)` sensitivity algorithm is typically faster than the generic `SciMLSensitivity` adjoints above.

### Regressor: how the angular velocity L(t) is modeled

SphereUDE represents the angular velocity `L(t)` with a `regressor`, any subtype of `AbstractRegressor`. Two are provided out of the box:
- `NNRegressor`, which wraps a `Lux.jl` neural network.
- `SplineRegressor`, which models `L(t)` with clamped B-splines and is typically cheaper and better-conditioned for this low-dimensional problem.

If you don't pass a `regressor` to `train`, a default `NNRegressor` is built for you. 
You can also build one explicitly, for instance customizing the neural network architecture: 
```julia
init_bias(rng, in_dims) = LinRange(params.tmin, params.tmax, in_dims)
init_weight(rng, out_dims, in_dims) = 0.1 * ones(out_dims, in_dims)

# Customized neural network to similate weighted moving window in L
U = Lux.Chain(
    Lux.Dense(1, 200, rbf, init_bias=init_bias, init_weight=init_weight, use_bias=true),
    Lux.Dense(200,10, gelu),
    Lux.Dense(10, 3, Base.Fix2(sigmoid_cap, params.ωmax), use_bias=false)
)

using Random
rng = Random.default_rng()
Random.seed!(rng, 613)

regressor, _ = NNRegressor(U, rng)
```
Or use a `SplineRegressor` instead, which only needs the time span, the number of control points, the spline degree, and `ωmax`:
```julia
regressor = SplineRegressor(
    tmin    = params.tmin,
    tmax    = params.tmax,
    n_basis = 50,
    degree  = 3,
    ωmax    = params.ωmax,
)
```

### Training

Training is finally done with 
```julia
results = train(data, params, rng, nothing, regressor)
```
with `rng` a random seed used for the initial parameter values of the regressor. 

Optimization can be sensitive to the random initialization, particularly for neural-network regressors. 
Passing `n_runs > 1` repeats the full training `n_runs` times from independent random initializations (drawn from `rng`) and keeps the run with the lowest final loss:
```julia
results = train(data, params, rng, nothing, regressor, n_runs = 5)
```

We can finally save and plot the data: 
```julia 
results_dict = convert2dict(data, results)

JLD2.@save "./results_dict.jld2" results_dict

plot_sphere(data, results, -30., 0., saveas="plot.png", title="Results")
plot_L(data, results, saveas="plot_L.png", title="Results")
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

## Citation

You can cite the software `SphereUDE.jl` as 
```
@software{SphereUDE.jl,
  author = {Sapienza, Facundo and Bolibar, Jordi},
  title = {SphereUDE.jl},
  month = jul,
  year = 2025,
  type = {software},
  publisher = {Zenodo},
  version = {v0.2.1},
  doi = {10.5281/zenodo.16423274},
  url = {https://doi.org/10.5281/zenodo.16423274},
  keywords = {software}
}
```
or cite our associated publication as
```
@article{sapienza2025spherical,
    title = {Spherical Path Regression Through Universal Differential Equations With Applications to Paleomagnetism}, 
    volume = {2}, 
    ISSN = {2993-5210}, 
    DOI = {10.1029/2025jh000626}, 
    number = {4},
    journal = {Journal of Geophysical Research: Machine Learning and Computation},
    author = {Sapienza, F. and Gallo, L. C. and Bolibar, J. and Pérez, F. and Taylor, J.},
    year = {2025},
}
```
