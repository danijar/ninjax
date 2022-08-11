import functools

import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj

# Ninjax supports all Haiku and Flax modules and new libraries are easy to add.
Linear = functools.partial(nj.HaikuModule, hk.Linear)


class Model(nj.Module):

  def __init__(self, size, lr=0.01, act=jax.nn.relu):
    self.size = size
    self.lr = lr
    self.act = act
    # Define submodules upfront.
    self.h1 = Linear(128)
    self.h2 = Linear(128)

  def __call__(self, x):
    x = self.act(self.h1(x))
    x = self.act(self.h2(x))
    # Define submodules inline.
    x = self.get('h3', Linear, self.size, with_bias=False)(x)
    # Create state entries of array values.
    x += self.get('bias', jnp.array, 0.0)
    return x

  def train(self, x, y):
    # Compute gradient with respect to all parameters in this module.
    loss, params, grad = nj.grad(self.loss, self)(x, y)
    # Update the parameters with gradient descent.
    state = jax.tree_util.tree_map(lambda p, g: p - self.lr * g, params, grad)
    # Update multiple state entries of this module.
    self.putm(state)
    return loss

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


model = Model(8)
train = nj.pure(model.train)  # nj.jit(...), nj.pmap(...)
main = jax.random.PRNGKey(0)

# The complete state is stored in a flat dictionary. Ninjax automatically
# applies scopes to the string keys based on the module names.
state = {}

# Let's train on some example data.
dataset = [(jnp.ones((64, 32)), jnp.ones((64, 8)))] * 10
for x, y in dataset:
  rng, main = jax.random.split(main)

  # Variables are automatically initialized on the first call. This adds them
  # to the state dictionary.
  loss, state = train(state, rng, x, y)

  # To look at parameters, simply use the state dictionary.
  assert state['/Model/bias'].shape == ()
  print('Loss:', float(loss))
