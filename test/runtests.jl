using Flax
using Test
using PyCall

pythoninstalled = false

@testset "Python environment" begin
    py"""
    import jax
    import numpy as np
    import jax.numpy as jnp
    from flax import linen as nn
    def one():
        return np.asfarray(jnp.ones([1]))
    """
    @test all(py"one"() .≈ 1.0)
end

pythoninstalled = all(py"one"() .≈ 1.0)

if (pythoninstalled)
    println("PyCall operational 🚀")  
    @testset "Flax.jl" begin
        include("layers.jl")
    end
else
    println("PyCall or python installation is broken")
end

