import contextlib
import functools
import inspect
import re

import jax
import jax.numpy as jnp


FRAME = [None]
SCOPE = ['']


class Frame:

  def __init__(self, state, rng, reserve, creating):
    self.state = state
    self.rng = rng
    self.reserve = reserve
    self.creating = creating


def frame():
  global FRAME
  if FRAME[0] is None:
    raise RuntimeError('Run stateful functions with run().')
  return FRAME[0]


def reset():
  global FRAME
  ModuleMeta.COUNTERS.clear()
  FRAME[0] = None


def run(fn, state, rng, *args, nested=False, creating=True, **kwargs):
  """Run a function or method that uses the global state or the global RNG
  key. The new global state and function output are returned."""
  global FRAME
  if not isinstance(state, dict):
    raise ValueError('Must provide a dict as state.')
  if (FRAME[0] is not None) and (not nested):
    raise RuntimeError(
        'If you really want to nest run() calls, use nested=True.')
  before = FRAME[0]
  try:
    FRAME[0] = Frame(state, rng, [], creating)
    out = fn(*args, **kwargs)
    state = FRAME[0].state
    return out, state
  finally:
    FRAME[0] = before


def state():
  """Access the global state tree. You can modify the state tree inplace."""
  return frame().state


def rng(amount=None, reserve=16):
  """Split the global RNG key and return a local key."""
  frame_ = frame()
  if amount:
    keys = jax.random.split(frame_.rng, amount + 1)
    frame_.rng = keys[0]
    return keys[1:]
  else:
    if not frame_.reserve:
      keys = jax.random.split(frame_.rng, reserve)
      frame_.rng = keys[0]
      frame_.reserve = list(keys[1:])
    return frame_.reserve.pop(0)


@contextlib.contextmanager
def scope(scope, absolute=False):
  """Enter a relative or absolute name scope. Name scopes are used to make
  variable names unique."""
  global SCOPE
  if SCOPE[0] is None:
    raise RuntimeError('Run stateful functions with run().')
  previous = SCOPE[0]
  if absolute:
    SCOPE[0] = scope
  else:
    SCOPE[0] += '/' + scope
  yield SCOPE[0]
  SCOPE[0] = previous


class ModuleMeta(type):

  """Meta class that creates a unique path for each module instance and wraps
  the methods and properties of the module to enter the name scope."""

  COUNTERS = {}

  def __new__(mcs, name, bases, clsdict):
    """This runs once per user module class definition. It wraps the methods of
    the module class to automatically enter the name scope of the module."""
    method_names = []
    for key, value in clsdict.items():
      if key.startswith('__') and key != '__call__':
        continue
      elif isinstance(value, property):
        clsdict[key] = property(
            value.fget if not value.fget else _scope_method(value.fget),
            value.fset if not value.fset else _scope_method(value.fset),
            value.fdel if not value.fdel else _scope_method(value.fdel),
            doc=value.__doc__)
      elif inspect.isfunction(value):
        method_names.append(key)
    cls = super(ModuleMeta, mcs).__new__(mcs, name, bases, clsdict)
    for method_name in method_names:
      method = getattr(cls, method_name)
      method = _scope_method(method)
      setattr(cls, method_name, method)
    return cls

  def __call__(cls, *args, name=None, **kwargs):
    """This runs once per use module instance creation. It derives a unique
    name and path for the module instance."""
    obj = cls.__new__(cls)
    name = name or cls.__name__
    global SCOPE
    path = SCOPE[0] + '/' + name
    if path in cls.COUNTERS:
      cls.COUNTERS[path] += 1
      path += str(cls.COUNTERS[path])
    else:
      cls.COUNTERS[path] = 1
    obj._path = path
    obj._submodules = {}
    init = _scope_method(cls.__init__)
    init(obj, *args, **kwargs)
    return obj


def _scope_method(method):
  @functools.wraps(method)
  def wrapper(self, *args, **kwargs):
    with scope(self._path, absolute=True):
      return method(self, *args, **kwargs)
  return wrapper


class Module(object, metaclass=ModuleMeta):

  """Base class for users to inherit their modules from. Provides automatic
  name scoping via the meta class and helper functions for accessing state."""

  def __repr__(self):
    return f'{self.__class__.__name__}({self.path})'

  @property
  def path(self):
    """The unique name scope of this module instance as a string."""
    return self._path

  def get(self, name, *args, **kwargs):
    """Retrieve or create a state entry that belongs to this module."""
    state_ = state()
    path = self.path + '/' + name
    if name in self._submodules:
      return self._submodules[name]
    if path in state_:
      return state_[path]
    if not frame().creating:
      raise RuntimeError(
          'Cannot create new variables inside symbolic loops. Call the inner '
          'function at least once before the loop.')
    ctor, *args = args
    if 'name' in inspect.signature(ctor).parameters:
      kwargs['name'] = name
    value = ctor(*args, **kwargs)
    flat, _ = jax.tree_util.tree_flatten(value)
    if all(isinstance(x, jnp.ndarray) for x in flat):
      state_[path] = value
    else:
      self._submodules[name] = value
    return value

  def put(self, name, value):
    """Update or create a single state entry that belongs to this module."""
    self.set_state({self.path + '/' + name: value})
    return value

  def get_state(self, pattern=r'.*', allow_empty=False):
    """Read the state entries of this module, optionally filtered by regex."""
    state_ = state()
    pattern = re.compile(pattern)
    prefix = self.path + '/'
    results = {}
    for key, value in state_.items():
      if not key.startswith(prefix):
        continue
      if pattern.match(key[len(prefix):]):
        results[key] = value
    if not allow_empty and not results:
      raise KeyError(f'Pattern {pattern} matched no state keys.')
    return results

  def set_state(self, mapping):
    """Update or create multiple state entries that belong to this module."""
    prefix = self.path + '/'
    for key in mapping:
      if not key.startswith(prefix):
        raise KeyError(f'Key {key} does not belong to module {self.path}.')
    state().update(mapping)


class Variable(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.ctor = ctor
    self.args = args
    self.kwargs = kwargs

  def read(self):
    return self.get('value', self.ctor, *self.args, **self.kwargs)

  def write(self, value):
    return self.put('value', value)


def grad(fn, keys, has_aux=False):
  """Compute the value and gradient of a function with respect to the state
  entries of the provided keys."""
  state_ = state()
  before = state_.copy()
  def inner(x, *args, **kwargs):
    state_.update(x)
    out = fn(*args, **kwargs)
    return out
  grad_fn = jax.value_and_grad(inner, has_aux=has_aux)
  @functools.wraps(grad_fn)
  def wrapper(*args, **kwargs):
    x = {k: state_[k] for k in keys}
    out = grad_fn(x, *args, **kwargs)
    state_.update(before)
    return out
  return wrapper


def cond(pred, true_fn, false_fn, *operands):
  out, state = jax.lax.cond(
      pred,
      lambda state, rng1, rng2, *args: run(true_fn, state, rng1, *args, nested=True),
      lambda state, rng1, rng2, *args: run(false_fn, state, rng2, *args, nested=True),
      frame().state, *rng(2), *operands)
  frame().state = state
  return out


def scan(f, init, xs, length=None, reverse=False, unroll=1):

  # # Run once outside of LAX to initialize variables, but do not update
  # # variables that already exist to reduce side effects.
  # x = jax.tree_util.tree_map(lambda x: x[0], xs)
  # state = run(f, STATE[0].copy(), rng, init, x, nested=True)[1]
  # print('A', STATE[0])
  # for key, value in state.items():
  #   if key not in STATE[0]:
  #     STATE[0][key] = value
  # print('B', STATE[0])

  # We currently use creating=False to forbid creating new variables inside the
  # symbolic loop. That's because the

  def inner(carry, x):
    carry, state = carry
    x, rng = x
    (carry, y), state = run(
        f, state, rng, carry, x, nested=True, creating=False)
    return (carry, state), y
  rngs = rng(length or len(jax.tree_util.tree_flatten(xs)[0][0]))
  carry, ys = jax.lax.scan(
      inner, (init, frame().state), (xs, rngs), length, reverse, unroll)
  carry, state = carry
  frame().state = state
  return carry, ys


class HaikuModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    import haiku as hk
    def net(*args_, **kwargs_):
      return ctor(*args, **kwargs)(*args_, **kwargs_)
    self.transformed = hk.transform(net)

  def __call__(self, *args, **kwargs):
    state = self.get('state', self.transformed.init, rng(), *args, **kwargs)
    return self.transformed.apply(state, rng(), *args, **kwargs)


class FlaxModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.module = ctor(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    state = self.get('state', self.module.init, rng(), *args, **kwargs)
    return self.module.apply(state, *args, **kwargs)


class OptaxModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.opt = ctor(*args, **kwargs)

  def __call__(self, params, loss, *args, **kwargs):
    import optax
    loss, grads = grad(loss, params.keys())(*args, **kwargs)
    optstate = self.get('state', self.opt.init, params)
    updates, optstate = self.opt.update(grads, optstate)
    self.put('state', optstate)
    state().update(optax.apply_updates(params, updates))
    return {'loss': loss.mean(), 'grad_norm': optax.global_norm(grads)}
