[![PyPI](https://img.shields.io/pypi/v/ninjax.svg)](https://pypi.python.org/pypi/ninjax/#history)

# 🥷  Ninjax: Flexible Modules for JAX

Ninjax is a general and practical module system for [JAX][jax]. It gives users
full and transparent control over updating the state of each module, bringing
flexibility to JAX and enabling new use cases.

## Overview

Ninjax provides a simple and general `nj.Module` class.

- Modules can store state for things like model parameters, Adam momentum
  buffer, BatchNorm statistics, recurrent state, etc.
- Modules can read and write their state entries. For example, this allows
  modules to have train methods, because they can update their parameters from
  the inside.
- Any method can initialize, read, and write state entries. This avoids the
  need for a special `build()` method or `@compact` decorator used in Flax.
- Ninjax makes it easy to mix and match modules from different libraries, such
  as [Flax][flax] and [Haiku][flax].
- Instead of PyTrees, Ninjax state is a flat `dict` that maps
  string keys like `/net/layer1/weights` to `jnp.array`s. This makes it easy
  to iterate over, modify, and save or load state.
- Modules can specify typed hyperparameters using the [dataclass][dataclass]
  syntax.

[jax]: https://github.com/google/jax
[flax]: https://github.com/google/flax
[haiku]: https://github.com/deepmind/dm-haiku
[dataclass]: https://docs.python.org/3/library/dataclasses.html

## Installation

Ninjax is [a single file][file], so you can just copy it to your project
directory. Or you can install the package:

```
pip install ninjax
```

[file]: https://github.com/danijar/ninjax/blob/main/ninjax/ninjax.py

## Quickstart

```python3
import flax
import jax
import jax.numpy as jnp
import ninjax as nj
import optax

Linear = nj.FromFlax(flax.linen.Dense)


class MyModel(nj.Module):

  lr: float = 1e-3

  def __init__(self, size):
    self.size = size
    # Define submodules upfront
    self.h1 = Linear(128, name='h1')
    self.h2 = Linear(128, name='h2')
    self.opt = optax.adam(self.lr)

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
    state = self.sub('optstate', nj.Tree, self.opt.init, params)
    updates, new_state = self.opt.update(grads, state.read(), params)
    params = optax.apply_updates(params, updates)
    nj.context().update(params)  # Store the new params
    state.write(new_state)       # Store new optimizer state
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
```

## Questions

If you have a question, please [file an issue][issues].

[issues]: https://github.com/danijar/ninjax/issues
