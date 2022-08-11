import functools

import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj
import optax

MLP = functools.partial(nj.HaikuModule, hk.nets.MLP)
Linear = functools.partial(nj.FlaxModule, nn.Dense)
Adam = functools.partial(nj.OptaxModule, optax.adam)


class Module(nj.Module):

  def __init__(self, size):
    self.mlp = MLP([128, 128])
    self.out = Linear(size)
    self.opt = Adam(1e-3)

  def __call__(self, x):
    x = self.mlp(x)
    x = jax.nn.relu(x)
    x = self.out(x)
    return x

  def train(self, x, y):
    return self.opt(self.loss, [self.mlp, self.out], x, y)

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


def main():
  x = jnp.ones((16, 8), jnp.float32)
  y = jnp.ones((16, 4), jnp.float32)

  model = Module(4)
  call = jax.jit(nj.pure(model))
  train = jax.jit(nj.pure(model.train))
  loss = jax.jit(nj.pure(model.loss))

  state = {}
  main = jax.random.PRNGKey(0)
  for step in range(5):
    rng, main = jax.random.split(main)
    metrics, state = train(state, rng, x, y)
    print('Loss:', float(metrics['loss']))


if __name__ == '__main__':
  main()
