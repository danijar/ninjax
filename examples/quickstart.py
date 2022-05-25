import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj


class Model(nj.Module):

  def __init__(self, size, act=jax.nn.relu):
    self.size = size
    self.act = act
    self.h1 = nj.HaikuModule(hk.Linear, 128)
    self.h2 = nj.HaikuModule(hk.Linear, 128)
    self.h3 = nj.HaikuModule(hk.Linear, size)

  def __call__(self, x):
    x = self.act(self.h1(x))
    x = self.act(self.h2(x))
    x = self.h3(x)
    return x

  def train(self, x, y):
    self(x)  # Create weights needed for gradient.
    state = self.get_state()
    loss, grad = nj.grad(self.loss, state)(x, y)
    state = jax.tree_map(lambda p, g: p - 0.01 * g, state, grad)
    self.set_state(state)
    return loss

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


dataset = [(jnp.ones((64, 32)), jnp.ones((64, 8)))] * 10
model = Model(8)
main = jax.random.PRNGKey(0)
state = {}
for x, y in dataset:
  rng, main = jax.random.split(main)
  loss, state = nj.run(model.train, state, rng, x, y)
  print('Loss:', float(loss))
