import flax
import jax
import jax.numpy as jnp
import ninjax as nj
import optax

Linear = nj.FromFlax(flax.linen.Dense)
Adam = nj.FromOptax(optax.adam)


class MyModel(nj.Module):

  lr: float = 1e-3

  def __init__(self, size):
    self.size = size
    # Define submodules upfront
    self.h1 = Linear(128, name='h1')
    self.h2 = Linear(128, name='h2')
    self.opt = Adam(self.lr, name='opt')

  def predict(self, x):
    x = jax.nn.relu(self.h1(x))
    x = jax.nn.relu(self.h2(x))
    # Define submodules inline
    x = self.sub('h3', Linear, self.size, use_bias=False)(x)
    # Create state entries inline
    x += self.value('bias', jnp.zeros, self.size)
    # Update state entries inline
    self.write('bias', self.read('bias') + 0.1)
    return x

  def loss(self, x, y):
    return ((self.predict(x) - y) ** 2).mean()

  def train(self, x, y):
    # Take grads wrt. to submodules or state keys
    wrt = [self.h1, self.h2, f'{self.path}/h3', f'{self.path}/bias']
    loss, params, grads = nj.grad(self.loss, wrt)(x, y)
    # Update weights
    updates = self.opt.update(grads, params)
    params = optax.apply_updates(params, updates)
    nj.context().update(params)  # Store the new params
    return loss


# Create model and example data
model = MyModel(3, name='model')
x = jnp.ones((64, 32), jnp.float32)
y = jnp.ones((64, 3), jnp.float32)

# Populate initial state from one or more functions
state = {}
state = nj.init(model.train)(state, x, y, seed=0)
print(state['model/bias'])

# Purify for JAX transformations
train = jax.jit(nj.pure(model.train))

# Training loop
for x, y in [(x, y)] * 10:
  state, loss = train(state, x, y)
  print('Loss:', float(loss))

# Look at the parameters
print(state['model/bias'])
