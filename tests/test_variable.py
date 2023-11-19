import jax
import jax.numpy as jnp
import ninjax as nj


class TestVariable:

  def test_reassign(self):
    def program(x1, x2):
      v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
      y1 = v.read()
      v.write(x1)
      y2 = v.read()
      v.write(x2)
      y3 = v.read()
      return y1, y2, y3
    rng = jax.random.PRNGKey(0)
    (y1, y2, y3), state = nj.pure(program)({}, rng, 12, 42)
    assert y1 == 1
    assert y2 == 12
    assert y3 == 42
    assert state == {'v/value': 42}

  def test_separate_runs(self):
    state = {}
    rng = jax.random.PRNGKey(0)
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    value, state = nj.pure(v.read)(state, rng)
    assert value == 1
    assert state == {'v/value': 1}
    value, state = nj.pure(v.write)(state, rng, 42)
    assert value == 42
    assert state == {'v/value': 42}
    value, state = nj.pure(v.read)(state, rng)
    assert value == 42
    assert state == {'v/value': 42}

  def test_name_reuse(self):
    state = {}
    rng = jax.random.PRNGKey(0)
    v1 = nj.Variable(lambda: jnp.array(1), name='Foo')
    v2 = nj.Variable(lambda: jnp.array(2), name='Foo')  # Constructor unused.
    v3 = nj.Variable(lambda: jnp.array(3), name='Bar')
    _, state = nj.pure(lambda: [v1.read(), v2.read(), v3.read()])(state, rng)
    assert state == {'Foo/value': 1, 'Bar/value': 3}
