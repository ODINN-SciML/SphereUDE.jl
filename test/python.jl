using SphereUDE

function test_matplotplib()
    plt[].figure(figsize = (10, 10))
    ax = plt[].axes(
        projection = ccrs[].Orthographic(central_latitude = 0.0, central_longitude = 0.0),
    )
    @test true
end

function test_pmagpy()
    pmag = SphereUDE.pyimport("pmagpy.pmag")
    @test true
end