
function test_reg_constructor()

    reg = Regularization(order=1, power=2.0, λ=0.1, diff_mode=ComplexStepDifferentiation())

    @test reg.order == 1
    @test reg.power == 2.0 
    @test reg.λ == 0.1 
    @test typeof(reg.diff_mode) <: AbstractDifferentiation

end

function test_param_constructor()

    params = SphereParameters(tmin=0.0, tmax=100., 
                              u0=[0. ,0. ,1.], 
                              ωmax=1.0, 
                              reg=nothing,
                              train_initial_condition=false,
                              multiple_shooting=true,
                              niter_ADAM=1000, niter_LBFGS=300, 
                              reltol=1e6, abstol=1e-6)

    reg1 = Regularization(order=0, power=2.0, λ=0.1, diff_mode=FiniteDifferences())
    reg2 = Regularization(order=1, power=1.0, λ=0.1, diff_mode=LuxNestedAD())

    params2 = SphereParameters(tmin=0.0, tmax=100., 
                               u0=[0. ,0. ,1.], 
                               ωmax=1.0, 
                               reg=[reg1, reg2],
                               train_initial_condition=false,
                               multiple_shooting=true,
                               niter_ADAM=1000, niter_LBFGS=300, 
                               reltol=1e6, abstol=1e-6)

    @test params.niter_ADAM == 1000
    @test params.tmax == 100.0

    @test params2.reg[1].order == 0
    @test typeof(params2.reg[2].diff_mode) <: LuxNestedAD

end