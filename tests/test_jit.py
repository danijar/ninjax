import jax
import jax.numpy as jnp
import ninjax as nj


class TestJit:

  def test_constants(self):
    fun = jax.jit(nj.pure(lambda: jnp.array(42)))
    assert not hasattr(fun, 'keys')
    assert fun({})[1] == 42
    # assert fun.keys == set()
    assert fun({})[1] == 42
    # assert fun.keys == set()

  def test_variables(self):
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    state = nj.init(v.read)({})
    read = jax.jit(nj.pure(v.read))
    write = jax.jit(nj.pure(v.write))
    # assert not hasattr(read, 'keys')
    # assert not hasattr(write, 'keys')
    state = read(state)[0]
    # assert read.keys == {'v/value'}
    assert state == {'v/value': 1}
    state = write(state, 42)[0]
    # assert write.keys == {'v/value'}
    assert state == {'v/value': 42}
    value = read(state)[1]
    assert value == 42
    assert state == {'v/value': 42}

  def test_statics(self):
    @nj.pure
    def program(x, mode='train'):
      return x if mode == 'train' else 2 * x
    fun = jax.jit(program, static_argnames=['mode'])
    assert fun({}, jnp.array(1), mode='train')[1] == 1
    assert fun({}, jnp.array(1), mode='train')[1] == 1
    assert fun({}, jnp.array(1), mode='eval')[1] == 2
    assert fun({}, jnp.array(1), mode='eval')[1] == 2

  def test_overlap(self):
    v1 = nj.Variable(jnp.zeros, (), jnp.int32, name='v1')
    v2 = nj.Variable(jnp.zeros, (), jnp.int32, name='v2')
    v3 = nj.Variable(jnp.zeros, (), jnp.int32, name='v3')
    def foo():
      v1.write(v1.read() + 1)
      return v2.read()
    def bar():
      v3.write(v1.read() + 1)
      return v1.read()
    state = {}
    state = nj.init(foo)(state)
    state = nj.init(bar)(state)
    assert state == {'v1/value': 0, 'v2/value': 0, 'v3/value': 0}
    foo = jax.jit(nj.pure(foo))
    bar = jax.jit(nj.pure(bar))
    state = foo(state)[0]
    assert state == {'v1/value': 1, 'v2/value': 0, 'v3/value': 0}
    state = bar(state)[0]
    assert state == {'v1/value': 1, 'v2/value': 0, 'v3/value': 2}
    state = foo(state)[0]
    assert state == {'v1/value': 2, 'v2/value': 0, 'v3/value': 2}
    state = bar(state)[0]
    assert state == {'v1/value': 2, 'v2/value': 0, 'v3/value': 3}

  def test_side_effect(self):
    counter = nj.Variable(jnp.array, 0, name='counter')
    @nj.pure
    def program(x):
      counter.write(counter.read() + 1)
      return counter.read()
    state = nj.init(program)({}, 0)
    fun = jax.jit(program)
    for index in range(1, 4):
      state, value = fun(state, jnp.array(2.0))
      assert value == index
