import contextlib
import functools
import inspect
import re

import jax


STATE = [None]
SCOPE = ['']
RNG_KEY = [None]


###############################################################################
# State
###############################################################################


def run(fn, state, rng_key, *a, **k):
  """Run a function or method that uses the global state or the global RNG
  key. The new global state and function output are returned."""
  global STATE
  STATE[0] = state
  RNG_KEY[0] = rng_key
  out = fn(*a, **k)
  STATE[0] = None
  RNG_KEY[0] = None
  return state, out


def state():
  """Access the global state tree. You can modify the state tree inplace."""
  global STATE
  if STATE[0] is None:
    raise RuntimeError('Run stateful functions with run().')
  return STATE[0]


def get(name, ctor, *a, **k):
  """Get or create a new entry in the global state. Respects the current name
  scope."""
  global SCOPE
  if SCOPE[0] is None:
    raise RuntimeError('Run stateful functions with run().')
  name = SCOPE[0] + '/' + name
  s = state()
  if name not in s:
    s[name] = ctor(*a, **k)
  return s[name]


def find(pattern, allow_empty=False):
  """Retrieve subtrees from the global state by matching the top-level keys to
  a regex."""
  pattern = re.compile(pattern)
  results = {}
  for key, value in state().items():
    if pattern.match(key):
      results[key] = value
  if not allow_empty and not results:
    keys = ', '.join(state().keys())
    raise KeyError(f'Pattern {pattern} matched none of the state keys: {keys}')
  return results


def grad(names, fn, *a, **k):
  """Compute the value and gradient of a function with respect to the state
  entries of the provided keys."""
  s = state()
  x = {k: s[k] for k in names}
  def inner(x, *a, **k):
    state().update(x)
    return fn(*a, **k)
  return jax.value_and_grad(inner)(x, *a, **k)


###############################################################################
# Randomness
###############################################################################


def next_rng_key():
  """Split the global RNG key to obtain a local key."""
  # TODO: Should we split keys hierarchically based on the module scopes to
  # reduce dependencies between random functions?
  global RNG_KEY
  if RNG_KEY[0] is None:
    raise RuntimeError('Run functions that use the global RNG key with run().')
  RNG_KEY[0], rng_key = jax.random.split(RNG_KEY[0])
  return rng_key


###############################################################################
# Scopes
###############################################################################


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
      obj.path = path
    init = _scope_method(cls.__init__)
    init(obj, *args, **kwargs)
    return obj


def _scope_method(method):
  @functools.wraps(method)
  def wrapper(self, *args, **kwargs):
    with scope(self.path, absolute=True):
      return method(self, *args, **kwargs)
  return wrapper


class Module(object, metaclass=ModuleMeta):

  def __repr__(self):
    return f'{self.__class__.__name__}({self.path})'
