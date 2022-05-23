import functools

import jax
import jax.numpy as jnp
import ninjax as nj
import optax


class Optimizer(nj.Module):

  def __init__(self, opt):
    self.opt = opt

  def __call__(self, pattern, loss, *a, **k):
    metrics = {}
    params = nj.find(pattern + r'(/?.*)')
    loss, grads = nj.grad(params.keys(), loss, *a, **k)
    optstate = nj.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate)
    nj.state()['state'] = optstate
    nj.state().update(optax.apply_updates(params, updates))
    metrics['loss'] = loss.mean()
    metrics['grad_norm'] = optax.global_norm(grads)
    return metrics


class MyModule(nj.Module):

  def __init__(self, size):
    self.size = size
    self.opt = Optimizer(optax.adam(1e-2))

  def __call__(self, x):
    init = jax.nn.initializers.glorot_uniform()
    shape = (x.shape[-1], self.size)
    w = nj.get('w', init, nj.next_rng_key(), shape, jnp.float32)
    b = nj.get('b', jnp.zeros, (self.size), jnp.float32)
    return x @ w + b

  def train(self, x, y):
    self(x)  # Initialize weights.
    metrics = self.opt(self.path + '/(w|b)', self.loss, x, y)
    return metrics

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
