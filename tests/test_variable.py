import jax.numpy as jnp
import ninjax as nj


class TestVariable:

  def test_reassign(self):
    @nj.pure
    def program(x1, x2):
      v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
      a = v.read()
      v.write(x1)
      b = v.read()
      v.write(x2)
      c = v.read()
      return a, b, c
    state = nj.init(program)({}, 0, 0)
    assert state == {'v/value': 1}
    state, (a, b, c) = program(state, 12, 42)
    assert a == 1
    assert b == 12
    assert c == 42
    assert state == {'v/value': 42}

  def test_repeat(self):
    state = {}
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    state, value = nj.pure(v.read)(state, create=True)
    assert value == 1
    assert state == {'v/value': 1}
    state, _ = nj.pure(v.write)(state, 42)
    _, value = nj.pure(v.read)(state)
    assert value == 42
    assert state == {'v/value': 42}
    state, value = nj.pure(v.read)(state)
    assert value == 42
    assert state == {'v/value': 42}

  def test_name_reuse(self):
    a = nj.Variable(lambda: jnp.array(1), name='Foo')
    b = nj.Variable(lambda: jnp.array(2), name='Foo')  # Constructor unused.
    c = nj.Variable(lambda: jnp.array(3), name='Bar')
    state, _ = nj.pure(lambda: [a.read(), b.read(), c.read()])({}, create=True)
    assert state == {'Foo/value': 1, 'Bar/value': 3}
