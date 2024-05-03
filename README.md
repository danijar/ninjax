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
```

## Tutorial

### How can I create state entries?

Ninjax gives modules full control over reading and updating their state
entries. Use `self.get(name, ctor, *args, **kwargs)` to define state entries.
The first call creates the entry as `ctor(*args, **kwargs)`. Later calls return
the current value:

```python3
class Module(nj.Module):

  def compute(self, x):
    init = jax.nn.initializers.variance_scaling(1, 'fan_avg', 'uniform')
    weights = self.get('weights', init, nj.rng(), (64, 32))
    bias = self.get('bias', jnp.zeros, (32,), jnp.float32)
    print(self.getm())  # {'/path/to/module/weights': ..., '/path/to/module/bias': ...}
    return x @ weights + bias
```

### How can I update state entries?

To update the state entries of a module, use `self.put(name, value)` for
individual entries of `self.put(mapping)` to update multiple values:

```python3
class Module(nj.Module):

  def counting(self):
    counter = nj.get('counter', jnp.zeros, (), jnp.int32)
    self.put('counter', counter + 1)
    print(self.get('counter'))  # 1
    state = self.getm()
    state['counter'] += 1
    self.put(state)
    print(self.getm()['counter'])  # 2
    print(self.get('counter'))  # 2
```

### How can I use JIT compilation?

The `nj.pure()` function makes the state your JAX code uses explicit, so it can
be transformed freely:

```python3
model = Model()
train = jax.jit(nj.pure(model.train))
params = {}
result, params = train(param, rng, ...)
```

Calling the pure function will create any parameters that are not yet in the
`params` dictionary and return the new parameters alongside the function
output.

You can speed up the first function call (where parameters are created) by
using `nj.jit` instead of `jax.jit`. Internally, this avoids compiling two
versions of the function.

### How can I compute gradients?

You can use `jax.grad` as normal for computing gradients with respect to
explicit inputs of your function. To compute gradients with respect to Ninjax
state, use `nj.grad(fn, keys)`:

```python3
class Module(nj.Module):

  def train(self, x, y):
    params = self.getm('.*')
    loss, grads = nj.grad(self.loss, params.keys())(x, y)
    params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    self.putm(params)
```

The `self.getm(filter='.*')` method optionally accepts a regex pattern to select
only a subset of the state dictionary. It also returns only state entries of
the current module. To access the global state, use `nj.state()`.

### How can I define modules compactly?

You can use `self.get(name, ctor, *args, **kwargs)` inside methods of your
modules. When called for the first time, it creates a new state entry from the
constructor `ctor(*args, **kwargs)`. Later calls return the existing entry:

```python3
class Module(nj.Module):

  def __call__(self, x):
    x = jax.nn.relu(self.get('h1', Linear, 128)(x))
    x = jax.nn.relu(self.get('h2', Linear, 128)(x))
    x = self.get('h3', Linear, 32)(x)
    return x
```

### How can I use Haiku modules?

There is nothing special about using external libraries with Ninjax. Haiku
requires its modules to be passed through `hk.transform` and the initialized
via `transformed.init(rng, batch)`. For convenience, Ninjax provides
`nj.HaikuModule` to do this for you:

```python3
class Module(nj.Module):

  def __init__(self):
    self.mlp = nj.HaikuModule(hk.nets.MLP, [128, 128, 32])

  def __call__(self, x):
    return self.mlp(x)
```

You can also predefine a list of aliases for Haiku modules that you want to use
frequently:

```python3
Linear = functools.partial(nj.HaikuModule, hk.Linear)
Conv2D = functools.partial(nj.HaikuModule, hk.Conv2D)
MLP = functools.partial(nj.HaikuModule, hk.nets.MLP)
# ...
```

### How can I use Flax modules?

There is nothing special about using external libraries with Ninjax. Flax
requires its modules to be initialized via `params = model.init(rng, batch)`
and used via `model.apply(params, data)`. For convenience, Ninjax provides
`nj.FlaxModule` to do this for you:

```python3
class Module(nj.Module):

  def __init__(self):
    self.linear = nj.FlaxModule(nn.Dense, 128)

  def __call__(self, x):
    return self.linear(x)
```

You can also predefine a list of aliases for Flax modules that you want to use
frequently:

```python3
Dense = functools.partial(nj.FlaxModule, nn.Dense)
Conv = functools.partial(nj.FlaxModule, nn.Conv)
# ...
```

### How can I use Optax optimizers?

There is nothing special about using external libraries like Optax with Ninjax.
Optax requires its optimizers to be initialized, their state to be passed
through the optimizer call, and the resulting updates to be applied. For
convenience, Ninjax provides `nj.OptaxModule` to do this for you:

```python3
class Module(nj.Module):

  def __init__(self):
    self.mlp = MLP()
    self.opt = nj.OptaxModule(optax.adam, 1e-3)

  def train(self, x, y):
    self.mlp(x)  # Ensure paramters are created.
    metrics = self.opt(self.mlp.getm('.*'), self.loss, x, y)
    return metrics  # {'loss': ..., 'grad_norm': ...}

  def loss(self, x, y):
    return ((self.mlp(x) - y) ** 2).mean()
```

## Questions

If you have a question, please [file an issue][issues].

[issues]: https://github.com/danijar/ninjax/issues
