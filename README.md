# SphereUDE

[![Build Status](https://github.com/facusapienza21/SphereUDE.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/facusapienza21/SphereUDE.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/facusapienza21/SphereUDE.jl/graph/badge.svg?token=UC0KFSUU3X)](https://codecov.io/gh/facusapienza21/SphereUDE.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

SphereUDE.jl is a Julia package for non-parametric regression of data supported in the three-dimensional sphere. 
It implements a simple universal differential equation (UDE) that naturaly constrains path to lie in the sphere. 
This has important application in Paleomagnetism, where the objective is to fit Aparent Polar Wander Paths (APWPs) to recunstruct continents past motion. 
In addition to sphere regression, SphereUDE.jl implements a series of improvements over previous modelling methods, such as 
- Explicit sphere constrint that allows for univesality of regression 
- Regularization on the path to incorporate physical priors 
- Incorporation of temporal and spatial uncertanties
- Uncertanty quantification capabilities

## Installing SphereUDE

In order to install `SphereUDE` in a given environment, just do in the REPL:
```julia
julia> ] # enter Pkg mode
(@v1.9) pkg> activate MyEnvironment # or activate whatever path for the Julia environment
(MyEnvironment) pkg> add SphereUDE
```

## SphereUDE initialization: integration with Python

In order to make plots using Matplotlib, Cartopy, and PMagPy, we install both [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) and execute Python code directly from Julia. In order to do this setup manually, you can follow the next steps. 

- Create a Python conda environmnet with all the requiremend packages using `conda env create -f environment.yml`.
- Inside the Julia REPL, install both `PyCall.jl` and `PyPlot.jl` with `] add PyCall, Pyplot`.
- Specify the Python path of the new environment with `ENV["PYTHON"] = ...`, where you should complete the path of the Python installation that shows when you do `conda activate SphereUDE`, `which python`.
- Inside the Julia REPL, execute `Pkg.build("PyCall")` to re-build PyCall with the new Python path. 

You are ready to use Python from your Julia session!
