export mpl_base, mpl_colormap, mpl_colormap, sns, ccrs, feature

function __init__()

    try
        copy!(mpl_base, pyimport("matplotlib"))
        copy!(mpl_colors, pyimport("matplotlib.colors"))
        copy!(mpl_colormap, pyimport("matplotlib.cm"))
        copy!(sns, pyimport("seaborn"))
        copy!(ccrs, pyimport("cartopy.crs"))
        copy!(feature, pyimport("cartopy.feature"))
    catch e 
        @warn "It looks like you have not installed and/or activated the virtual Python environment. \n 
        Please follow the guidelines in: https://github.com/ODINN-SciML/SphereUDE.jl"
        @warn exception=(e, catch_backtrace())
    end

end