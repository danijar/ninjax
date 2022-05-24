import functools

import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj

MLP = functools.partial(nj.HaikuModule, hk.nets.MLP)


class MyModule(nj.Module):

  def __init__(self, size):
    self.mlp = MLP([128, 128, size])

  def __call__(self, x):
    return self.mlp(x)

  def train(self, x, y):
    self(x)  # Initialize weights.
    params = self.mlp.state()
    loss, grads = nj.grad(self.loss, params)(x, y)
    params = jax.tree_map(lambda p, g: p - 0.01 * p, params, grads)
    self.update(params)
    return {'loss': loss}

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


def main():
  x = jnp.ones((16, 8), jnp.float32)
  y = jnp.ones((16, 4), jnp.float32)

  model = MyModule(4)
  call = jax.jit(functools.partial(nj.run, model))
  train = jax.jit(functools.partial(nj.run, model.train))
  loss = jax.jit(functools.partial(nj.run, model.loss))
  state = {}
  main = jax.random.PRNGKey(0)

  for step in range(5):
    rng, main = jax.random.split(main)
    state, metrics = train(state, rng, x, y)
    print('Loss:', float(metrics['loss']))


if __name__ == '__main__':
  main()

