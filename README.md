[![PyPI](https://img.shields.io/pypi/v/ninjax.svg)](https://pypi.python.org/pypi/ninjax/#history)

# 🥷  Ninjax

Ninjax brings the flexibility of PyTorch and TensorFlow 2 to [JAX][jax]. Ninjax
is a simple state manager for JAX that makes it easy to have nested components
that update their own state (e.g. have their own `train()` functions). It's
intended to be used together with a neural network library, such as
[Flax][flax] or [Haiku][flax].

[jax]: https://github.com/google/jax
[flax]: https://github.com/google/flax
[haiku]: https://github.com/deepmind/dm-haiku

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
    loss, grad = nj.grad(self.loss, [self.h1, self.h2, self.h3])(x, y)
    state = jax.tree_map(lambda p, g: p - 0.01 * g, state, grad)
    self.update(state)
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

## Guide

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
    params = self.state()
    loss, grads = nj.grad(self.loss, params.keys())(x, y)
    params = jax.tree_map(lambda p, g: p - 0.01 * g, params, grads)
    self.update(params)
```

The `self.state(filter)` method optionally accepts a regex pattern to select
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

Haiku requires its modules to be passed through `hk.transform` and the
initialized via `transformed.init(rng, batch)`. Ninjax provides
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

