[![PyPI](https://img.shields.io/pypi/v/ninjax.svg)](https://pypi.python.org/pypi/ninjax/#history)

# 🥷  Ninjax

Ninjax is a lightweight module system for [JAX][jax] that gives modules full
control over their state (e.g. to have their own `train()` functions). It also
makes it easy to use modules from different libraries together, such as
[Flax][flax] and [Haiku][flax].

[jax]: https://github.com/google/jax
[flax]: https://github.com/google/flax
[haiku]: https://github.com/deepmind/dm-haiku

## Motivation

Existing neural network libraries for JAX provide modules, but their modules
only specify neural graphs and cannot easily implement their own training
logic. Orchestrating training logic all in one place, outside of the modules,
is fine for simple code bases. But it becomes a problem when there are mnay
modules with their own training logic and optimizers.

Ninjax solves this problem by giving each `nj.Module` full read and write
access to its state, while remaining functional via `nj.run()`. This means
modules can have `train()` functions to implement custom training logic, and
call each other's train functions. Ninjax is intended to be used with one or
more neural network libraries, such as [Haiku][haiku] and [Flax][flax].

## Installation

Ninjax is [a single file][file], so you can just copy it to your project
directory. Or you can install the package:

```
pip install ninjax
```

[file]: https://github.com/danijar/ninjax/blob/main/ninjax/ninjax.py

## Quickstart

```python3
import haiku as hk
import flax.linen as nn
import jax
import jax.numpy as jnp
import ninjax as nj


class Model(nj.Module):

  def __init__(self, size, act=jax.nn.relu):
    self.size = size
    self.act = act
    self.h1 = nj.HaikuModule(hk.Linear, 128)
    self.h2 = nj.HaikuModule(hk.Linear, 128)
    self.h3 = nj.FlaxModule(nn.Linear, size)

  def __call__(self, x):
    x = self.act(self.h1(x))
    x = self.act(self.h2(x))
    x = self.h3(x)
    return x

  def train(self, x, y):
    self(x)  # Ensure parameters are created.
    state = self.get_state()
    loss, grad = nj.grad(self.loss, state)(x, y)
    state = jax.tree_map(lambda p, g: p - 0.01 * g, state, grad)
    self.set_state(state)
    return loss

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


model = Model(8)
main = jax.random.PRNGKey(0)
state = {}
for x, y in dataset:
  rng, main = jax.random.split(main)
  state, loss = nj.run(model.train, state, rng, x, y)
  print('Loss:', float(loss))
```

## API

Ninjax uses a simple API that provides flexible access to state. The `nj.run()`
lets you call your modules from the outside. All other functions are to be used
in your modules or other code that is within the `nj.run()` call.

```python3
# Run a function or method that uses Ninjax state.
state, out = nj.run(state, rng, fn, *args, **kwargs)

# Inherit your modules from this class for automatic name scoping and helper
# functions for accesing the state that belongs to this module.
class nj.Module:
  @path                                 # Unique scope string for this module.
  def get(name, ctor, *args, **kwargs)  # Get or create state entry.
  def set(name, value)                  # Update state entry.
  def get_state(filter='.*')            # Get multiple state entries.
  def set_state(entries)                # Update multiple state entries.

# Return the mutable global state dictionary.
state = nj.state()

# Get a unique random number generator key.
rng = nj.rng()

# Compute the gradient with respect to global state entries, specified by key.
grad = nj.grad(fn, keys)(*args, **kwargs)

# Ninjax provides convenience wrappers for popular JAX libraries. These
# automate the manual initialization and state passing that these libraries
# require for you:

mlp = nj.HaikuModule(hk.nets.MLP, [128, 128, 32])
outputs = mlp(inputs)

opt = nj.OptaxModule(optax.adam, 1e-3)
opt(mlp.get_state(), loss, data)  # Train the MLP with a loss function.
```

## Tutorial

### How can I use JIT compilation?

The `nj.run()` function makes the state your JAX code uses explicit, so it can
be jitted and transformed freely:

```python3
model = Model()
train = jax.jit(functools.partial(nj.run, model.train))
train(state, rng, ...)
```

### How can I compute gradients?

You can use `jax.grad` as normal for computing gradients with respect to
explicit inputs of your function. To compute gradients with respect to Ninjax
state, use `nj.grad(fn, keys)`:

```python3
class Module(nj.Module):

  def train(self, x, y):
    params = self.get_state('.*')
    loss, grads = nj.grad(self.loss, params.keys())(x, y)
    params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    self.set_state(params)
```

The `self.get_state(filter='.*')` method optionally accepts a regex pattern to select
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
    metrics = self.opt(self.mlp.get_state('.*'), self.loss, x, y)
    return metrics  # {'loss': ..., 'grad_norm': ...}

  def loss(self, x, y):
    return ((self.mlp(x) - y) ** 2).mean()
```

## Limitations

Ninjax is still a young library. One current limitation is that LAX symbolic
control flow and computing gradients of gradients has not been tested and might
not work correctly. If you are interested in this functionality or encounter
any other issues, let me know.

## Questions

If you have a question, please [file an issue][issues].

[issues]: https://github.com/danijar/ninjax/issues
