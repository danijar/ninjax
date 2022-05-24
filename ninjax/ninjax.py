import contextlib
import functools
import inspect
import re

import jax


STATE = [None]
SCOPE = ['']
RNG = [None]


def run(fn, state, rng, *a, **k):
  """Run a function or method that uses the global state or the global RNG
  key. The new global state and function output are returned."""
  global STATE
  STATE[0] = state
  RNG[0] = rng
  out = fn(*a, **k)
  STATE[0] = None
  RNG[0] = None
  return state, out


def state():
  """Access the global state tree. You can modify the state tree inplace."""
  global STATE
  if STATE[0] is None:
    raise RuntimeError('Run stateful functions with run().')
  return STATE[0]


def rng():
  """Split the global RNG key and return a local key."""
  global RNG
  if RNG[0] is None:
    raise RuntimeError('Run functions that use the global RNG key with run().')
  RNG[0], rng = jax.random.split(RNG[0])
  return rng


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
    if name in cls.COUNTERS:
      cls.COUNTERS[name] += 1
      name += str(cls.COUNTERS[name])
    else:
      cls.COUNTERS[name] = 1
    obj.name = name
    with scope(name) as path:
      obj._path = path
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
    name = self.path + '/' + name
    if name not in state_:
      ctor, *args = args
      state_[name] = ctor(*args, **kwargs)
    return state_[name]

  def put(self, name, value):
    """Update or create a single state entry that belongs to this module."""
    self.set_state({self.path + '/' + name: value})

  def get_state(self, filter=r'.*', allow_empty=False):
    """Read the state entries of this module, optionally filtered by regex."""
    state_ = state()
    filter = re.compile(filter)
    prefix = self.path + '/'
    results = {}
    for key, value in state_.items():
      if not key.startswith(prefix):
        continue
      if filter.match(key[len(prefix):]):
        results[key] = value
    if not allow_empty and not results:
      raise KeyError(f'Filter {filter} matched no state keys.')
    return results

  def set_state(self, mapping):
    """Update or create multiple state entries that belong to this module."""
    prefix = self.path + '/'
    for key in mapping:
      if not key.startswith(prefix):
        raise KeyError(f'Key {key} does not belong to module {self.path}.')
    state().update(mapping)


def grad(fn, keys):
  """Compute the value and gradient of a function with respect to the state
  entries of the provided keys."""
  state_ = state()
  def inner(x, *args, **kwargs):
    state_.update(x)
    return fn(*args, **kwargs)
  grad_fn = jax.value_and_grad(inner)
  @functools.wraps(grad_fn)
  def wrapper(*args, **kwargs):
    x = {k: state_[k] for k in keys}
    return grad_fn(x, *args, **kwargs)
  return wrapper


class HaikuModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    import haiku as hk
    def net(*args_, **kwargs_):
      return ctor(*args, **kwargs)(*args_, **kwargs_)
    self.transformed = hk.transform(net)

  def __call__(self, *args, **kwargs):
    state = self.get('haiku', self.transformed.init, rng(), *args, **kwargs)
    return self.transformed.apply(state, rng(), *args, **kwargs)


class FlaxModule(Module):

  def __init__(self, ctor, *args, **kwargs):
    self.module = ctor(*args, **kwargs)

  def __call__(self, *args, **kwargs):
    state = self.get('flax', self.module.init, rng(), *args, **kwargs)
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
