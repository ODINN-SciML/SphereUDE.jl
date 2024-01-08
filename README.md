# SphereFit

[![Build Status](https://github.com/facusapienza21/SphereFit.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/facusapienza21/SphereFit.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/facusapienza21/SphereFit.jl/graph/badge.svg?token=UC0KFSUU3X)](https://codecov.io/gh/facusapienza21/SphereFit.jl)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Python setup

In order to make plots using Matplotlib, Cartopy, and PMagPy, we install both [PyCall.jl](https://github.com/JuliaPy/PyCall.jl) and [PyPlot.jl](https://github.com/JuliaPy/PyPlot.jl) and execute Python code directly from Julia. In order to do this setup manually, you can follow the next steps. 

- Create a Python conda environmnet with all the requiremend packages using `conda env create -f environment.yml`.
- Inside the Julia REPL, install both `PyCall.jl` and `PyPlot.jl` with `] add PyCall, Pyplot`.
- Specify the Python path of the new environment with `ENV["PYTHON"] = ...`, where you should complete the path of the Python installation that shows when you do `conda activate SphereFit`, `which python`.
- Inside the Julia REPL, execute `Pkg.build("PyCall")` to re-build PyCall with the new Python path. 

You are ready to use Python from your Julia session!
