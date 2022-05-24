import functools

import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj


class MyModule(nj.Module):

  def __init__(self, size):
    def net(x):
      return hk.nets.MLP([128, 128, size])(x)
    self.mlp = hk.transform(net)

  def __call__(self, x):
    weights = self.get('mlp', self.mlp.init, nj.rng(), x)
    return self.mlp.apply(weights, None, x)

  def train(self, x, y):
    self(x)  # Initialize weights.
    params = self.state('mlp')
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
  main_rng = jax.random.PRNGKey(0)

  for step in range(5):
    rng, main_rng = jax.random.split(main_rng)
    state, metrics = train(state, rng, x, y)
    if step == 0:
      print(jax.tree_map(lambda x: x.shape, state))
    print('Loss:', float(metrics['loss']))
  rng, main_rng = jax.random.split(main_rng)
  state, out = call(state, rng, x)


if __name__ == '__main__':
  main()
