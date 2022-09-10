
from flax import linen as nn
import jax
import numpy as np
import jax.numpy as jnp
import haiku as hk


seed = 3
rng = jax.random.PRNGKey(seed)
rng, init_key = jax.random.split(rng)
x = jax.random.normal(rng,[1,3,3,1])
convt = nn.ConvTranspose(1,(3,3))
ps = convt.init(init_key, x)['params']
y = convt.apply({'params':ps}, x)
y

def hkconvtfn(x: jnp.ndarray) -> jnp.ndarray:
  net = hk.Sequential([
    hk.Conv2DTranspose(1, (3,3))
  ])
  return net(x)

seed = 3
rng = jax.random.PRNGKey(seed)
rng, init_key = jax.random.split(rng)
x = jax.random.normal(rng,[1,3,3,1])
hkconvtfn_t = hk.without_apply_rng(hk.transform(hkconvtfn))
params = hkconvtfn_t.init(init_key, x)
hkconvtfn_t.apply(params, x)
w = params['conv2_d_transpose']['w']
stride = (1,1)
padding = "SAME"
j = jax.lax.conv_transpose(x, w, strides = stride, padding=padding, transpose_kernel=True)