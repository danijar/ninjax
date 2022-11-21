[![PyPI](https://img.shields.io/pypi/v/ninjax.svg)](https://pypi.python.org/pypi/ninjax/#history)

# 🥷  Ninjax: General Modules for JAX

Ninjax is a general module system for [JAX][jax]. It gives the user complete
and transparent control over updating the state of each module, bringing the
flexibility of PyTorch and TensorFlow to JAX. Moreover, Ninjax makes it easy to
mix and match modules from different libraries, such as [Flax][flax] and
[Haiku][flax].

[jax]: https://github.com/google/jax
[flax]: https://github.com/google/flax
[haiku]: https://github.com/deepmind/dm-haiku

## Motivation

Existing deep learning libraries for JAX provide modules, but those modules
only specify neural networks and cannot easily implement training logic.
Orchestrating training all in one place, outside of the modules, is fine for
simple code bases. But it becomes a problem when there are many modules with
their own training logic and optimizers.

Ninjax solves this problem by giving each `nj.Module` full read and write
access to its state. This means modules can have train functions to
implement custom training logic, and call each other's train functions. Ninjax
is intended to be used with one or more neural network libraries, such as
[Haiku][haiku] and [Flax][flax].

The main differences to existing deep learning libraries are:

- Ninjax does not need separate `apply()`/`init()` functions. Instead, the
  first function call creates variables automatically.
- Ninjax lets you access and update model parameters inside of impure
  functions, so modules can handle their own optimizers and update logic.
- Natural support for modules with multiple functions without need for
  Flax's `setup()` function or Haiku's `hk.multi_transform()`.
- Ninjax' flexible state handling makes it trivial to mix and match
  modules from other deep learning libraries in your models.

## Installation

Ninjax is [a single file][file], so you can just copy it to your project
directory. Or you can install the package:

```
pip install ninjax
```

[file]: https://github.com/danijar/ninjax/blob/main/ninjax/ninjax.py

## Quickstart

```python3
import functools

import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj

# Ninjax supports all Haiku and Flax modules and new libraries are easy to add.
Linear = functools.partial(nj.HaikuModule, hk.Linear)


class MyModel(nj.Module):

  def __init__(self, size, lr=0.01, act=jax.nn.relu):
    self.size = size
    self.lr = lr
    self.act = act
    # Define submodules upfront.
    self.h1 = Linear(128, name='h1')
    self.h2 = Linear(128, name='h2')

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


# The complete state is stored in a flat dictionary. Ninjax automatically
# applies scopes to the string keys based on the module names.
state = {}
model = MyModel(8, name='model')
train = nj.pure(model.train)  # nj.jit(...), nj.pmap(...)
main = jax.random.PRNGKey(0)

# Let's train on some example data.
dataset = [(jnp.ones((64, 32)), jnp.ones((64, 8)))] * 10
for x, y in dataset:
  rng, main = jax.random.split(main)
  # Variables are automatically initialized on the first call. This adds them
  # to the state dictionary.
  loss, state = train(state, rng, x, y)
  # To look at parameters, simply use the state dictionary.
  assert state['/model/bias'].shape == ()
  print('Loss:', float(loss))
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
individual entries of `self.putm(mapping)` to update multiple values:

```python3
class Module(nj.Module):

  def counting(self):
    counter = nj.get('counter', jnp.zeros, (), jnp.int32)
    self.put('counter', counter + 1)
    print(self.get('counter'))  # 1
    state = self.getm()
    state['counter'] += 1
    self.putm(state)
    print(self.getm()['counter'])  # 2
    print(self.get('counter'))  # 2
```

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
