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
    # Get and put variables.
    counter = nj.get('counter', jnp.zeros, (), jnp.int32)
    self.put('counter', counter + 1)
    return loss

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


model = Model(8)
main = jax.random.PRNGKey(0)
state = {}
for x, y in dataset:
  rng, main = jax.random.split(main)
  loss, state = nj.run(model.train, state, rng, x, y)
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
    print(self.get_state())  # {'/path/to/module/weights': ..., '/path/to/module/bias': ...}
    return x @ weights + bias
```

### How can I update state entries?

To update the state entries of a module, use `self.put(name, value)` for
individual entries of `self.set_state(mapping)` to update multiple values:

```python3
class Module(nj.Module):

  def counting(self):
    counter = nj.get('counter', jnp.zeros, (), jnp.int32)
    self.put('counter', counter + 1)
    print(self.get('counter'))  # 1
    state = self.get_state()
    state['counter'] += 1
    self.set_state(state)
    print(self.get_state()['counter'])  # 2
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
