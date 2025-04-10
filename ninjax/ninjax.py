import contextlib
import functools
import inspect
import re
import threading
import traceback
import types

import jax
import jax.numpy as jnp

__version__ = '3.6.1'


def add_note(e, note):
  if hasattr(e, 'add_note'):
    e.add_note(note)
  else:
    print(note)


def hidestack(fn):
  @functools.wraps(fn)
  def hidewrapper(*args, **kwargs):
    try:
      return fn(*args, **kwargs)
    except Exception as e:
      tb = e.__traceback__
      filtered = None
      frames = list(traceback.walk_tb(tb))
      for i, (f, lineno) in enumerate(reversed(frames)):
        if f.f_code.co_filename == __file__:
          if i == 0:
            pass
          elif f.f_code.co_name == 'hidewrapper':
            continue
          elif jax.config.jax_traceback_filtering != 'off':
            continue
        filtered = types.TracebackType(filtered, f, f.f_lasti, lineno)
      e.with_traceback(filtered)
      raise e
  return hidewrapper


###############################################################################
# State
###############################################################################


# When running an impure function that accesses state, it will find the state
# in this global variable. The pure() wrapper populates this global variable
# with the provided state, calls the inner function, and then the takes the
# resulting state out of the global variable to return it back to the user.
# To allow multi-threaded programs to use impure functions in parallel, the
# context is a dictionary with a slot for each thread identifier.
CONTEXT = {}


class Context(dict):

  def __init__(
      self, entries, seed, create, modify, ignore, reserve, name):
    super().__init__(entries)
    self.create = create   # Allow creating new state entries.
    self.modify = modify   # Allow modifying existing state entries.
    self.ignore = ignore   # Ignore modifications to existing state entries.
    self.seed = seed
    self.reserve = reserve
    self.name = name
    self.accessed = set()  # Keys accessed for reading.
    self.created = set()   # Keys accessed for creating.
    self.modified = set()  # Keys accessed for modifying (even if ignored).

  def update(self, entries):
    for key, value in dict(entries).items():
      self[key] = value

  def __getitem__(self, key):
    self.accessed.add(key)
    try:
      return super().__getitem__(key)
    except KeyError:
      raise KeyError(
          f"Trying to access state key '{key}' that does not exist in context "
          f'create={self.create} modify={self.modify} ignore={self.ignore}.')

  def __setitem__(self, key, value):
    if key in self:
      self.modified.add(key)
    else:
      self.created.add(key)
    if self.ignore and key in self:
      return  # Do not overwrite existing entries.
    if not self.create and key not in self:
      raise RuntimeError(
          'Pass create=True to pure functions to allow them to create new '
          f'state entries or use nj.init(). You were trying to set {key} to '
          f'shape {value.shape}.')
    if not self.modify:
      existing = self[key]
      raise RuntimeError(
          'Cannot modify state entries here. (If you want to modify '
          'state inside of scan() pass modify=True.) ' +
          f'You were trying to change {key} from shape {existing.shape} '
          f'and dtype {existing.dtype} to shape {value.shape} and ' +
          f'dtype {value.dtype}.')
    super().__setitem__(key, value)


def pure(fun, nested=False):
  """Wrap an impure function that uses global state to explicitly pass the
  state in and out. The result is a pure function that is composable with JAX
  transformation. The pure function can be used as follows:
  ```
  state, out = fun(state, *args, **kwargs)
  ```
  Additional keyword arguments can be provided:
  - `seed`: Provide an integer or array of two integers to be able to use
    `nj.seed()` inside the impure function.
  - `create=False`: Boolean indicating whether the impure function will be
    allowed to create new state entries.
  - `modify=True`: Boolean indicating whether the impure function will be
    allowed to modify existing state entries.
  - `ignore=False`: Boolean indicating whether state modifications by the
    impure function will be ignored silently; useful for initialization.
  - `track=False`: Boolean indicating whether to return the sets of state
    keys that the impure function attempted to read, modify, and create.
  """
  @hidestack
  def purified(
      state, *args, seed=None, create=None, modify=None, ignore=None,
      track=False, **kwargs):
    if isinstance(seed, int) or (hasattr(seed, 'shape') and seed.shape == ()):
      seed = jnp.array([seed, seed], jnp.uint32)
    context = CONTEXT.get(threading.get_ident(), None)
    if context is not None:
      create = create if create is not None else context.create
      modify = modify if modify is not None else context.modify
      ignore = ignore if ignore is not None else context.ignore
      assert context.create or not create, 'Parent context disabled create.'
      assert context.modify or not modify, 'Parent context disabled modify.'
      assert not context.ignore or ignore, 'Parent context enabled ignore.'
    else:
      create = create if create is not None else False
      modify = modify if modify is not None else True
      ignore = ignore if ignore is not None else False
    if not isinstance(state, dict):
      raise ValueError('Must provide a dict as state.')
    name = getattr(fun, '__name__', str(fun))
    if context and (not nested):
      raise RuntimeError(
          f'You are trying to call pure {name}() inside pure '
          f'{context.name}(). Is that intentional? If you want to nest pure '
          f'functions, use pure(..., nested=True) for the inner function.')
    before = context
    try:
      context = Context(
          state.copy(), seed, create, modify, ignore, [], name)
      CONTEXT[threading.get_ident()] = context
      out = fun(*args, **kwargs)
      state = dict(context)
      if before:
        before.accessed |= context.accessed
        before.modified |= context.modified
        before.created |= context.created
      if track:
        return state, out, context.accessed, context.modified, context.created
      return state, out
    finally:
      CONTEXT[threading.get_ident()] = before
  purified._is_pure = True
  return purified


def context():
  """Access and modify the global context from within an impure function. For
  advanced users only. Prefer to use module methods to access and modify state
  and seed() to get the next random seed."""
  context = CONTEXT.get(threading.get_ident(), None)
  if context is None:
    raise RuntimeError('Wrap impure functions in pure() before running them.')
  return context


def init(fun, **jit_kwargs):
  """Creates an initializer for a pure or impure function, which when called
  with example inputs , quickly populates the initial state without performing
  the actual computation of the function."""
  if not getattr(fun, '_is_pure', False):
    fun = pure(fun)
  @hidestack
  def wrapper(*args, **kwargs):
    state, out = fun(*args, create=True, modify=True, ignore=True, **kwargs)
    del out
    return state
  return jax.jit(wrapper, **jit_kwargs)


@jax.named_scope('seed')
def seed(amount=None, optional=False, reserve=16):
  """Split the global random seed and return a new local seed."""
  ctx = context()
  if ctx.seed is None:
    if optional:
      return None if amount is None else [None] * amount
    raise ValueError(
        'You must provide a seed to the pure function to use nj.seed() '
        'inside the impure function.')
  if amount:
    keys = jax.random.split(ctx.seed, amount + 1)
    ctx.seed = keys[0]
    return keys[1:]
  else:
    if not ctx.reserve:
      keys = jax.random.split(ctx.seed, reserve)
      ctx.seed = keys[0]
      ctx.reserve = list(keys[1:])
    return ctx.reserve.pop(0)


def creating():
  """Indicates whether the program is currently allowed to create state
  entries. Can use used for initialization logic that should be excluded from
  compiled functions."""
  return context().create


###############################################################################
# Transformations
###############################################################################


@jax.named_scope('grad')
def grad(fun, targets, has_aux=False):
  """Compute the gradient of an impure function with respect to either function
  arguments or state entries. The transformed function returns a tuple
  containing the function output, the selected targets, their gradients, and if
  applicable auxiliary outputs of the function."""

  single = isinstance(targets, int)
  targets = targets if hasattr(targets, '__len__') else (targets,)

  ctx = context()

  if not has_aux:
    fun = lambda *args, _fun=fun, **kwargs: (_fun(*args, **kwargs), {})
  fun = pure(fun, nested=True)

  @hidestack
  def wrapper(*args, **kwargs):
    accessed, modified = _prerun(fun, *args, **kwargs)

    # If differentiating with respect to function inputs, offset the argument
    # numbers to account for the state entry inputs to the wrapper function.
    if all(isinstance(x, int) for x in targets):
      argnums = [2 + x for x in targets]
      x1 = {k: v for k, v in ctx.items() if k in accessed}
      x2 = {}
      selected = [args[i] for i in targets]

    # If differentiating with respect to state entries, take the gradient with
    # respect to the first input of the wrapper function.
    else:
      argnums = 0
      strs = []
      for target in targets:
        if isinstance(target, Module):
          prefix = target.path + '/'
          matches = {k: v for k, v in ctx.items() if k.startswith(prefix)}
        if isinstance(target, str):
          pattern = re.compile(f'^{target}(/.*|$)')
          matches = [k for k in ctx if pattern.match(k)]
        if not matches:
          existing = ', '.join(ctx.keys())
          raise KeyError(
              f"Gradient target '{target}' did not match any state entries. " +
              f'Existing state entries: {existing}')
        strs += matches
      existing = ctx.keys()
      assert all(key in existing for key in strs), (strs, existing)
      x1 = {k: v for k, v in ctx.items() if k in strs}
      x2 = {k: v for k, v in ctx.items() if k not in strs}
      if not x1:
        raise ValueError(
            'No inputs to differentiate with respect to. ' +
            f"User provided targets: '{targets}'")
      for key in x1.keys():
        if key not in accessed:
          raise RuntimeError(
              f"Trying to compute gradient with respect to key '{key}' "
              'but the differentiated function does not access it.\n'
              f'Accessed keys: {list(accessed)}\n'
              f'Gradient keys: {list(strs)}')
      x1 = {k: v for k, v in x1.items() if k in accessed}
      x2 = {k: v for k, v in x2.items() if k in accessed}
      selected = x1

    def forward(x1, x2, *args, **kwargs):
      before = {**x1, **x2}
      state, (y, aux) = fun(before, *args, create=False, **kwargs)
      changes = {k: v for k, v in state.items() if k in modified}
      return y, (changes, aux)

    backward = jax.value_and_grad(forward, argnums, has_aux=True)
    (y, (changes, aux)), dx = backward(
        x1, x2, *args, seed=seed(None, True), **kwargs)

    if ctx.modify:
      ctx.update(changes)

    if ctx.modify and not all(isinstance(x, int) for x in targets):
      selected = {k: ctx[k] for k in selected.keys()}

    if single:
      selected, = selected
      dx, = dx

    return (y, selected, dx, aux) if has_aux else (y, selected, dx)

  return wrapper


@jax.named_scope('cond')
def cond(pred, true_fun, false_fun, *operands):
  true_fun = pure(true_fun, nested=True)
  false_fun = pure(false_fun, nested=True)

  accessed1, modified1 = _prerun(true_fun, *operands)
  accessed2, modified2 = _prerun(false_fun, *operands)
  accessed = accessed1 | accessed2
  modified = modified1 | modified2

  def true_fun_wrapper(state, seed1, seed2, *args):
    state, outs = true_fun(state, *args, seed=seed1)
    changes = {k: v for k, v in state.items() if k in modified}
    return changes, outs

  def false_fun_wrapper(state, seed1, seed2, *args):
    state, outs = false_fun(state, *args, seed=seed2)
    changes = {k: v for k, v in state.items() if k in modified}
    return changes, outs

  needed = {k: v for k, v in context().items() if k in accessed}
  changes, out = jax.lax.cond(
      pred, true_fun_wrapper, false_fun_wrapper,
      needed, *seed(2, True), *operands)
  if context().modify:
    context().update(changes)
  return out


@jax.named_scope('while_loop')
def while_loop(cond_fun, body_fun, carry):
  body_fun = pure(body_fun, nested=True)
  accessed, modified = _prerun(body_fun, carry)

  changing = {k: v for k, v in context().items() if k in modified}
  unchanging = {
      k: v for k, v in context().items()
      if k in accessed and k not in modified}
  shared_seed = seed(optional=True)

  def cond_fun_wrapper(carry):
    return cond_fun(carry[0])

  def body_fun_wrapper(carry):
    carry, changing, index = carry
    if shared_seed is None:
      seed = None
    else:
      seed = jax.random.fold_in(shared_seed, index)
    state = {**unchanging, **changing}
    state, carry = body_fun(state, carry, create=False, seed=seed)
    changing = {k: v for k, v in state.items() if k in modified}
    return carry, changing, index + 1

  carry, changes, _ = jax.lax.while_loop(
      cond_fun_wrapper, body_fun_wrapper, (carry, changing, 0))

  if context().modify:
    context().update(changes)

  return carry


@jax.named_scope('scan')
def scan(fun, carry, xs, length=None, reverse=False, unroll=1, axis=0):
  if axis:
    xs = jax.tree.map(lambda x: x.swapaxes(0, axis), xs)

  fun = pure(fun, nested=True)
  accessed, modified = _prerun(
      fun, carry, jax.tree.map(lambda x: x[0], xs))

  changing = {k: v for k, v in context().items() if k in modified}
  unchanging = {
      k: v for k, v in context().items()
      if k in accessed and k not in modified}

  def inner(carry, x):
    carry, changing = carry
    x, seed = x
    state = {**unchanging, **changing}
    state, (carry, y) = fun(state, carry, x, create=False, seed=seed)
    changing = {k: v for k, v in state.items() if k in modified}
    return (carry, changing), y

  if length is None:
    length = len(jax.tree.leaves(xs)[0])
  seeds = seed(length, True)
  (carry, changes), ys = jax.lax.scan(
      inner, (carry, changing), (xs, seeds), length, reverse, unroll)

  if context().modify:
    context().update(changes)

  if axis:
    ys = jax.tree.map(lambda y: y.swapaxes(0, axis), ys)
  return carry, ys


def checkpoint(fun, **cp_kwargs):
  static = cp_kwargs.get('static_argnums', tuple())
  static = static if isinstance(static, tuple) else (static,)
  static = tuple(x + 1 for x in static)
  cp_kwargs['static_argnums'] = static

  accessed, modified = [None], [None]
  fun = pure(fun, nested=True)

  @functools.partial(jax.checkpoint, **cp_kwargs)
  def inner(*args, **kwargs):
    state, output = fun(*args, **kwargs)
    changes = {k: v for k, v in state.items() if k in modified[0]}
    return changes, output

  @jax.named_scope('checkpoint')
  def outer(*args, **kwargs):
    accessed[0], modified[0] = _prerun(fun, *args, **kwargs)
    needed = {k: v for k, v in context().items() if k in accessed[0]}
    changes, output = inner(needed, *args, seed=seed(None, True), **kwargs)
    if context().modify:
      context().update(changes)
    return output

  return outer


@jax.named_scope('prerun')
def _prerun(fun, *args, **kwargs):
  if not context().modify and not context().create:
    return set(), set()
  # Copy container structure so modifications inside the user function
  # (e.g. popping from a dict) are not applied during prerun.
  args, kwargs = jax.tree.map(lambda x: x, (args, kwargs))
  state, output, accessed, modified, created = fun(
      dict(context()), *args, ignore=True, track=True,
      seed=seed(None, True), **kwargs)
  del output
  creations = {k: v for k, v in state.items() if k in created}
  context().update(creations)
  return accessed, modified


###############################################################################
# Modules
###############################################################################


SCOPE = ''


@contextlib.contextmanager
def scope(name, absolute=False, multi=False):
  """Enter a relative or absolute name scope. Name scopes are used to make
  names of state entries unique."""
  global SCOPE
  if SCOPE is None:
    raise RuntimeError(
        'Purify stateful functions with fn = pure(fn) before running them.')
  outside = SCOPE
  if absolute:
    validate(name)
    SCOPE = name
  elif SCOPE == '':
    SCOPE = name
  else:
    validate(name, single=(not multi))
    SCOPE = outside + '/' + name
  try:
    yield SCOPE
  except Exception as e:
    if not hasattr(e, '_njscope'):
      e._njscope = SCOPE
      add_note(e, f"Exception happened inside Ninjax scope '{SCOPE}'.")
    raise
  finally:
    SCOPE = outside


def validate(path, single=False):
  names = path.split('/')
  assert not single or len(names) == 1, (path, names, single)
  for name in names:
    assert name, ('Name cannot be empty', path, name)
    assert '{' not in name, ('Did you forget to format a string?', path, name)
    assert re.match(r'^[A-Za-z0-9_]+$', name), (
        'Only letters, numbers, and underscores allowed in names', path, name)


class ModuleMeta(type):
  """Meta class that creates a unique path for each module instance and wraps
  the methods and properties of the module to enter the name scope."""

  def __new__(mcs, name, bases, clsdict):
    """This runs once per user module class definition. It wraps the methods of
    the module class to automatically enter the name scope of the module."""
    method_names = []
    # Scope user methods of user modules but not of ninjax.Module.
    if bases != (object,):
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
    for name, typ in cls.__annotations__.items():
      try:
        isinstance(0, typ)
      except Exception:
        raise ValueError(
            f"Annotation '{typ}' for field '{key}' is not a valid type.")
    cls._defaults = {
        k: getattr(cls, k) for k, v in cls.__annotations__.items()
        if hasattr(cls, k)}
    for key, value in cls.__annotations__.items():
      setattr(cls, key, property(lambda self, key=key: self._fields[key]))
    for name in method_names:
      if name in cls._defaults:
        continue
      method = getattr(cls, name)
      method = _scope_method(method)
      setattr(cls, name, method)
    return cls

  def __call__(cls, *args, name=None, **kwargs):
    """This runs once per use module instance creation. It derives a unique
    name and path for the module instance."""
    if not isinstance(name, str):
      raise TypeError(
          "Please provide a module name via Module(..., name='example').")
    validate(name, single=True)
    fields = {}
    for key, typ in cls.__annotations__.items():
      if key in kwargs:
        value = kwargs.pop(key)
      elif key in cls._defaults:
        value = cls._defaults[key]
      else:
        raise TypeError(
            f"Pass a keyword arg for field '{key}' or define a default.")
      if typ is not None and not isinstance(value, typ):
        raise TypeError(
            f"Value '{value}' for field '{key}' is not of type "
            f"'{typ.__name__}'.")
      fields[key] = value
    obj = cls.__new__(cls)
    obj._fields = fields
    with scope(name) as path:
      obj._path = path
    obj._submodules = {}
    init = _scope_method(cls.__init__)
    try:
      init(obj, *args, **kwargs)
    except TypeError as e:
      if kwargs:
        keys = ', '.join(sorted(kwargs.keys()))
        add_note(e, f'Keyword arguments not matched to class fields: {keys}')
      raise
    return obj


def _scope_method(method):
  @hidestack
  @functools.wraps(method)
  def wrapper(self, *args, **kwargs):
    with scope(self._path, absolute=True):
      with jax.named_scope(self._path.split('/')[-1]):
        return method(self, *args, **kwargs)
  wrapper._method = method  # Debug info.
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

  @property
  def name(self):
    """The name of this module instance as a string."""
    return self._path.split('/')[-1]

  @property
  def defaults(self):
    return self._defaults

  @property
  def values(self):
    p = self.path + '/'
    ctx = context()
    # Read keys individually to mark them as accessed.
    return {k.removeprefix(p): ctx[k] for k in ctx if k.startswith(p)}

  def value(self, name, make, *args, **kwargs):
    """Define and read a state entry in the scope of this module."""
    validate(name)
    assert SCOPE == self.path, (
        name, 'Values can only be created in the root scope of a module.')
    path = self.path + '/' + name
    if path not in context():
      if callable(make):
        value = make(*args, **kwargs)
      else:
        assert not args and not kwargs
        value = make
      context()[path] = value
    # Look up the value again to register it as accessed.
    return context()[path]

  def read(self, name):
    """Read a state entry in the scope of this module."""
    validate(name)
    return context()[self.path + '/' + name]

  def write(self, name, value):
    """Update the value of a state entry in the scope of this module."""
    validate(name)
    path = self.path + '/' + name
    existing = context()[path]
    value = jnp.asarray(value, dtype=existing.dtype)
    assert existing.shape == value.shape, (existing.shape, value.shape)
    context()[path] = value
    # Return value without lookup because it was provided by the user and thus
    # has to be available in the pure function already.
    return value

  def sub(self, name, make=None, *args, **kwargs):
    """Define and retrieve a sub module."""
    validate(name)
    assert SCOPE == self.path or SCOPE.startswith(self.path + '/'), (
        name, 'Can only create submodules from inside the parent module.')
    if SCOPE == self.path:
      handle = name
    else:
      assert SCOPE.startswith(self.path + '/')
      handle = SCOPE[len(self.path) + 1:] + '/' + name
    if handle not in self._submodules:
      assert make, 'Provide constructor for submodule that does not exist.'
      module = make(*args, **kwargs, name=name)
      self._submodules[handle] = module
    return self._submodules[handle]


class Variable(Module):

  def __init__(self, make, *args, **kwargs):
    self.make = functools.partial(make, *args, **kwargs)

  def read(self):
    if not self.values:
      self.value('value', self.make)
    return super().read('value')

  def write(self, value):
    if not self.values:
      self.value('value', self.make)
    return super().write('value', value)


class Tree(Module):

  def __init__(self, make, *args, **kwargs):
    self.make = functools.partial(make, *args, **kwargs)

  def read(self):
    if not self.values:
      mapping, self.treedef = flatten(self.make())
      [self.value(k, v) for k, v in mapping.items()]
    return unflatten(self.values, self.treedef)

  def write(self, tree):
    if not self.values:
      mapping, self.treedef = flatten(self.make())
      [self.value(k, v) for k, v in mapping.items()]
    mapping, treedef = flatten(tree)
    assert treedef == self.treedef, (self.treedef, treedef)
    write = super().write
    [write(k, v) for k, v in mapping.items()]
    return jax.tree.map(lambda x: x, tree)


def flatten(tree):
  items, treedef = jax.tree_util.tree_flatten_with_path(tree)
  paths, values = zip(*items)
  tostr = lambda x: re.sub(
      r'[^a-z0-9-_/]+', '_', str(x).lower()).replace('_/', '').replace('_', '')
  strpaths = [[tostr(x) for x in path] for path in paths]
  keys = ['/'.join(x for x in strpath if x) for strpath in strpaths]
  if len(set(keys)) < len(keys):
    raise ValueError(
        'Cannot flatten PyTree to dict because paths are ambiguous '
        'after converting them to string keys.\n'
        'Paths: {paths}\nKeys: {keys}')
  items = sorted(list(zip(keys, values)), key=lambda x: x[0])
  return dict(items), treedef


def unflatten(mapping, treedef):
  items = sorted(list(mapping.items()), key=lambda x: x[0])
  _, values = zip(*items) if items else ([], [])
  return jax.tree.unflatten(treedef, values)


###############################################################################
# Integrations
###############################################################################


def FromFlax(make, postinit=None):

  class FlaxModule(Module):

    def __init__(self, *args, **kwargs):
      if callable(make):
        self.module = make(*args, **kwargs)
      else:
        assert not args and not kwargs
        self.module = make
      self.treedef = None

    def __call__(self, *args, **kwargs):
      if kwargs.get('mutable', False):
        raise NotImplementedError
      params = self._params(*args, **kwargs)
      return self.module.apply(params, *args, **kwargs)

    def _params(self, *args, **kwargs):
      if self.treedef is None:
        params = self.module.init(seed(), *args, **kwargs)
        params = postinit(params) if postinit else params
        mapping, self.treedef = flatten(params)
        [self.value(k, v) for k, v in mapping.items()]
      return unflatten(self.values, self.treedef)

  return FlaxModule


def FromHaiku(make):

  class HaikuModule(Module):

    def __init__(self, *args, **kwargs):
      import haiku as hk
      def net(*a, **k):
        return make(*args, **kwargs)(*a, **k)
      self.transformed = hk.transform(net)
      self.treedef = None

    def __call__(self, *args, **kwargs):
      params = self._params(*args, **kwargs)
      return self.transformed.apply(params, seed(), *args, **kwargs)

    def _params(self, *args, **kwargs):
      if self.treedef is None:
        params = self.transformed.init(seed(), *args, **kwargs)
        mapping, self.treedef = flatten(params)
        [self.value(k, v) for k, v in mapping.items()]
      return unflatten(self.values, self.treedef)

  return HaikuModule


def FromOptax(make):

  class OptaxModule(Module):

    def __init__(self, *args, **kwargs):
      if callable(make):
        self.opt = make(*args, **kwargs)
      else:
        assert not args and not kwargs
        self.opt = make
      self.treedef = None

    def __call__(self, loss, keys, *args, **kwargs):
      import optax
      loss, params, grads = grad(loss, keys)(*args, **kwargs)
      updates = self.update(grads, params)
      context().update(optax.apply_updates(params, updates))
      return loss, params, grads

    def update(self, grads, params):
      if self.treedef is None:
        mapping, self.treedef = flatten(self.opt.init(params))
        [self.value(k, v) for k, v in mapping.items()]
      state = unflatten(self.values, self.treedef)
      updates, state = self.opt.update(grads, state)
      [self.write(k, v) for k, v in flatten(state)[0].items()]
      return updates

  return OptaxModule
