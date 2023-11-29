import flax
import jax
import jax.numpy as jnp
import ninjax as nj
import optax

# Supports all Flax and Haiku modules
Linear = nj.FromFlax(flax.linen.Dense)


class MyModel(nj.Module):

  lr: float = 0.01

  def __init__(self, size):
    self.size = size
    self.opt = optax.adam(self.lr)
    # Define submodules upfront
    self.h1 = Linear(128, name='h1')
    self.h2 = Linear(128, name='h2')

  def predict(self, x):
    x = jax.nn.relu(self.h1(x))
    x = jax.nn.relu(self.h2(x))
    # Define submodules inline
    x = self.get('h3', Linear, self.size, use_bias=False)(x)
    # Create state entries inline
    x += self.get('bias', jnp.zeros, self.size)
    return x

  def train(self, x, y):
    # Gradient with respect to submodules or state entries
    keys = [self.h1, self.h2, f'{self.path}/h3', f'{self.path}/bias']
    loss, params, grads = nj.grad(self.loss, keys)(x, y)
    # Update weights
    optstate = self.get('optstate', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate)
    new_params = optax.apply_updates(params, updates)
    self.put(new_params)  # Store the new params
    return loss

  def loss(self, x, y):
    return ((self.predict(x) - y) ** 2).mean()


# Create model and example data
model = MyModel(3, lr=0.01, name='model')
x = jnp.ones((64, 32), jnp.float32)
y = jnp.ones((64, 3), jnp.float32)

# Populate initial state from one or more functions
state = {}
state = nj.init(model.train)(state, x, y, seed=0)
print(state['model/bias'])  # [0., 0., 0.]

# Purify and jit one or more functions
train = nj.pure(model.train)
train = jax.jit(train)

# Training loop
for x, y in [(x, y)] * 10:
  state, loss = train(state, x, y)
  print('Loss:', float(loss))

# Look at the parameters
print(state['model/bias'])  # [-1.2e-09  1.8e-08 -2.5e-09]
