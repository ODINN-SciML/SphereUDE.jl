export mpl_base, mpl_colormap, mpl_colormap, sns, ccrs, feature

function __init__()

    try
        isassigned(mpl_base) ? nothing : mpl_base[] = pyimport("matplotlib")
        isassigned(mpl_colors) ? nothing : mpl_colors[] = pyimport("matplotlib.colors")
        isassigned(mpl_colormap) ? nothing : mpl_colormap[] = pyimport("matplotlib.cm")
        isassigned(sns) ? nothing : sns[] = pyimport("seaborn")
        isassigned(ccrs) ? nothing : ccrs[] = pyimport("cartopy.crs")
        isassigned(feature) ? nothing : feature[] = pyimport("cartopy.feature")
    catch e 
        @warn "It looks like you have not installed and/or activated the virtual Python environment. \n 
        Please follow the guidelines in: https://github.com/ODINN-SciML/SphereUDE.jl"
        @warn exception=(e, catch_backtrace())
    end

end