using NNlib, PyCall, Flax, Flux

py"""
import jax
import numpy as np
import jax.numpy as jnp
from flax import linen as nn

def convtest(seed):
    rng = jax.random.PRNGKey(seed)
    conv = nn.Conv(features = 2, kernel_size=(2,2), padding=0)
    convparams = conv.init(rng, jnp.ones((1,4,4,1)))['params']
    x = jax.random.normal(rng, (1,4,4,1))
    y = conv.apply({'params':convparams}, x)
    return np.asfarray(convparams['kernel']), np.asfarray(convparams['bias']), \
     np.asfarray(x), np.asfarray(y)

def densetest(seed=0):
    rng = jax.random.PRNGKey(seed)
    dense = nn.Dense(features = 2)
    denseparams = dense.init(rng, jnp.ones((1,4)))['params']
    x = jax.random.normal(rng, (1,4))
    y = dense.apply({'params':denseparams}, x)
    return np.asfarray(denseparams['kernel']), np.asfarray(denseparams['bias']), \
    np.asfarray(x), np.asfarray(y)

def convtest2(seed):
    rng = jax.random.PRNGKey(seed)
    rng, init1, init0 = jax.random.split(rng, 3)
    x = jax.random.normal(rng, (1,4,4,1))
    
    conv0 = nn.Conv(features = 2, kernel_size=(2,2), padding=0)
    conv0params = conv0.init(init0, jnp.ones((1,4,4,1)))['params']
    y = conv0.apply({'params':conv0params}, x)

    y = nn.relu(y)

    conv1 = nn.Conv(features = 2, kernel_size=(2,2), padding=0)
    conv1params = conv1.init(init1, y)['params']
    y = conv1.apply({'params':conv1params}, y)

    return np.asfarray(conv0params['kernel']), np.asfarray(conv0params['bias']), \
      np.asfarray(conv1params['kernel']), np.asfarray(conv1params['bias']), np.asfarray(x), np.asfarray(y)

def convdensetest(seed):
    rng = jax.random.PRNGKey(seed)
    rng, init1, init2 = jax.random.split(rng, 3)
    x = jax.random.normal(rng, (1,4,4,1))

    conv0 = nn.Conv(features = 2, kernel_size=(2,2), padding=0)
    conv0params = conv0.init(init1, x)['params']
    y = conv0.apply({'params':conv0params}, x)

    y = nn.relu(y)

    y0 = jnp.copy(y)

    y = y.reshape(y.shape[0],-1,order='F')
    y1 = jnp.copy(y)

    dense1 = nn.Dense(features = 2)
    dense1params = dense1.init(init2, y)['params']
    y = dense1.apply({'params':dense1params}, y)

    return np.asfarray(conv0params['kernel']), np.asfarray(conv0params['bias']), \
      np.asfarray(dense1params['kernel']), np.asfarray(dense1params['bias']), np.asfarray(x), np.asfarray(y), \
      np.asfarray(y0), np.asfarray(y1)
"""

@testset "Single Conv" begin 
    W,b,x,y = py"convtest(0)"
    xx = Float32.(permutedims(x, (3,2,4,1)))
    fc = Flax.FConv(W,b)
    yy = permutedims(y, (3,2,4,1))
    @test isapprox(fc(xx), yy; atol=1e-6)
end

@testset "Conv => relu => Conv" begin
    W0,b0,W1,b1,x,y = py"convtest2"(0)
    xx = permutedims(x, (3,2,4,1))
    yy = permutedims(y, (3,2,4,1))
    m = Chain(
            Flax.FConv(W0, b0, relu),
            Flax.FConv(W1,b1)
        )
    @test isapprox(m(xx), yy, rtol=1e-5)
end

@testset "Single dense" begin 
    W,b,x,y = py"densetest(0)"
    xx = permutedims(x, (2,1))
    yy = permutedims(y, (2,1))
    WW = permutedims(W, (2,1))
    fc = Dense(WW,b)
    @test isapprox(fc(xx),yy,atol=1e-5)
end

######

@testset "Conv => Relu => Flatten => Dense" begin
    W0,b0,W1,b1,x,y ,y0,y1= py"convdensetest"(0)
    
    xx = permutedims(x, (3,2,4,1))
    yy = permutedims(y, (2,1))

    m = Chain(
        Flax.FConv(W0,b0,relu),
        Flax.Fflatten,
        Flax.FDense(W1,b1)
    )
    @test isapprox(m(xx), yy, rtol=1e-5)
end