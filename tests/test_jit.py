import jax
import jax.numpy as jnp
import ninjax as nj


class TestJit:

  def test_constants(self):
    fun = nj.jit(nj.pure(lambda: jnp.array(42)))
    assert not hasattr(fun, 'keys')
    assert fun({}, jax.random.PRNGKey(0))[0] == 42
    # assert fun.keys == set()
    assert fun({}, jax.random.PRNGKey(0))[0] == 42
    # assert fun.keys == set()

  def test_variables(self):
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    read = nj.jit(nj.pure(v.read))
    write = nj.jit(nj.pure(v.write))
    state = {}
    rng = jax.random.PRNGKey(0)
    # assert not hasattr(read, 'keys')
    # assert not hasattr(write, 'keys')
    state = read(state, rng)[1]
    # assert read.keys == {'v/value'}
    assert state == {'v/value': 1}
    state = write(state, rng, 42)[1]
    # assert write.keys == {'v/value'}
    assert state == {'v/value': 42}
    value = read(state, rng)[0]
    assert value == 42
    assert state == {'v/value': 42}

  def test_statics(self):
    def program(x, mode='train'):
      return x if mode == 'train' else 2 * x
    fun = nj.jit(nj.pure(program), static=['mode'])
    rng = jax.random.PRNGKey(0)
    assert fun({}, rng, jnp.array(1), mode='train')[0] == 1
    assert fun({}, rng, jnp.array(1), mode='train')[0] == 1
    assert fun({}, rng, jnp.array(1), mode='eval')[0] == 2
    assert fun({}, rng, jnp.array(1), mode='eval')[0] == 2

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
    foo = nj.jit(nj.pure(foo))
    bar = nj.jit(nj.pure(bar))
    state = {}
    rng = jax.random.PRNGKey(0)
    # assert not hasattr(foo, 'keys')
    # assert not hasattr(bar, 'keys')
    state = foo(state, rng)[1]
    assert state == {'v1/value': 1, 'v2/value': 0}
    state = bar(state, rng)[1]
    assert state == {'v1/value': 1, 'v2/value': 0, 'v3/value': 2}
    state = foo(state, rng)[1]
    assert state == {'v1/value': 2, 'v2/value': 0, 'v3/value': 2}
    state = bar(state, rng)[1]
    assert state == {'v1/value': 2, 'v2/value': 0, 'v3/value': 3}
    # assert foo.keys == {'v1/value', 'v2/value'}
    # assert bar.keys == {'v1/value', 'v3/value'}

  def test_side_effect(self):
    counter = nj.Variable(jnp.array, 0, name='counter')
    def program(x):
      counter.write(counter.read() + 1)
      return counter.read()
    rng = jax.random.PRNGKey(0)
    fun = nj.jit(nj.pure(program))
    state = {}
    for index in range(1, 4):
      value, state = fun(state, rng, jnp.array(2.0))
      assert value == index
