import jax
import jax.numpy as jnp
import ninjax as nj


class TestPmap:

  def test_plain_pmap(self):
    inner = nj.pure(lambda x: x ** 2)
    fun = jax.pmap(inner, 'devices')
    rng = jax.random.PRNGKey(0)[None]
    x = jnp.ones((1, 128))
    assert fun({}, rng, x)[0].shape == (1, 128)
    assert fun({}, rng, x)[0].shape == (1, 128)

  def test_constants(self):
    fun = nj.pmap(nj.pure(lambda x: x ** 2), 'devices')
    rng = jnp.repeat(jax.random.PRNGKey(0)[None], 1, 0)
    x = jnp.ones((1, 128))
    # assert not hasattr(fun, 'keys')
    assert fun({}, rng, x)[0].shape == (1, 128)
    # assert fun.keys == set()
    assert fun({}, rng, x)[0].shape == (1, 128)
    # assert fun.keys == set()

  def test_variables(self):
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    read = nj.pmap(nj.pure(v.read))
    write = nj.pmap(nj.pure(v.write))
    rng = jax.random.PRNGKey(0)[None]
    state = {}
    value, state = read(state, rng)
    assert value == jnp.array([1])
    assert state == {'v/value': jnp.array([1])}
    value, state = write(state, rng, jnp.array([2]))
    assert value == jnp.array([2])
    assert state == {'v/value': jnp.array([2])}
    value, state = write(state, rng, jnp.array([3]))
    assert value == jnp.array([3])
    assert state == {'v/value': jnp.array([3])}
    value, state = read(state, rng)
    assert value == jnp.array([3])
    assert state == {'v/value': jnp.array([3])}

  def test_statics(self):
    def program(x, mode='train'):
      return x if mode == 'train' else 2 * x
    fun = nj.pmap(nj.pure(program), static=['mode'])
    rng = jax.random.PRNGKey(0)[None]
    x = jnp.array([1])
    assert fun({}, rng, x, mode='train')[0] == jnp.array([1])
    assert fun({}, rng, x, mode='train')[0] == jnp.array([1])
    assert fun({}, rng, x, mode='eval')[0] == jnp.array([2])
    assert fun({}, rng, x, mode='eval')[0] == jnp.array([2])

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
    foo = nj.pmap(nj.pure(foo))
    bar = nj.pmap(nj.pure(bar))
    state = {}
    rng = jax.random.PRNGKey(0)[None]
    assert not hasattr(foo, 'keys')
    assert not hasattr(bar, 'keys')
    state = foo(state, rng)[1]
    assert state == {
        'v1/value': jnp.array([1]),
        'v2/value': jnp.array([0])}
    state = bar(state, rng)[1]
    assert state == {
        'v1/value': jnp.array([1]),
        'v2/value': jnp.array([0]),
        'v3/value': jnp.array([2])}
    state = foo(state, rng)[1]
    assert state == {
        'v1/value': jnp.array([2]),
        'v2/value': jnp.array([0]),
        'v3/value': jnp.array([2])}
    state = bar(state, rng)[1]
    assert state == {
        'v1/value': jnp.array([2]),
        'v2/value': jnp.array([0]),
        'v3/value': jnp.array([3])}
    # assert foo.keys == {'v1/value', 'v2/value'}
    # assert bar.keys == {'v1/value', 'v3/value'}
