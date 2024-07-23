import jax
import jax.numpy as jnp
import ninjax as nj
import pytest


class TestInit:

  def test_empty(self):
    fun = nj.pure(lambda: jnp.array(42))
    state = nj.init(fun)({})
    assert state == {}
    assert jax.jit(fun)(state)[1] == 42

  def test_variable(self):
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    read = nj.pure(v.read)
    write = nj.pure(v.write)
    state = nj.init(read)({})
    assert state == {'v/value': 1}
    assert read(state)[1] == 1
    state, _ = write(state, 42)
    assert state == {'v/value': 42}
    assert read(state)[1] == 42

  def test_random(self):
    @nj.pure
    def program():
      v = nj.Variable(jax.random.uniform, nj.seed(), name='v')
      return v.read()
    with pytest.raises(ValueError):
      nj.init(program)({})
    state = nj.init(program)({}, seed=0)
    assert set(state.keys()) == {'v/value'}
    assert jnp.round(state['v/value'], 4) == 0.9892
    state, value = jax.jit(program)(state, seed=1)
    assert jnp.round(value, 4) == 0.9892

  def test_static(self):
    traces = []
    @nj.pure
    def program(mode='a'):
      traces.append(mode)
      a = nj.Variable(jnp.array, 42, name='a')
      b = nj.Variable(jnp.array, 21, name='b')
      if mode == 'a':
        return a.read()
      else:
        return b.read()
    state = nj.init(program, static_argnames=['mode'])({}, mode='a')
    assert traces == ['a']
    state = nj.init(program, static_argnames=['mode'])(state, mode='b')
    assert traces == ['a', 'b']
    assert state == {'a/value': 42, 'b/value': 21}
    apply = jax.jit(program, static_argnames=['mode'])
    assert apply(state, mode='a')[1] == 42
    assert traces == ['a', 'b', 'a']
    assert apply(state, mode='a')[1] == 42
    assert traces == ['a', 'b', 'a']
    assert apply(state, mode='b')[1] == 21
    assert traces == ['a', 'b', 'a', 'b']
    assert apply(state, mode='b')[1] == 21
    assert traces == ['a', 'b', 'a', 'b']

  def test_overlap(self):
    a = nj.Variable(jnp.zeros, (), jnp.int32, name='a')
    b = nj.Variable(jnp.zeros, (), jnp.int32, name='b')
    c = nj.Variable(jnp.zeros, (), jnp.int32, name='c')
    @nj.pure
    def foo():
      a.write(a.read() + 1)
      return b.read()
    @nj.pure
    def bar():
      c.write(a.read() + 1)
      return a.read()
    state = {}
    state = nj.init(foo)(state)
    state = nj.init(bar)(state)
    assert state == {'a/value': 0, 'b/value': 0, 'c/value': 0}

  def test_dependent(self):
    class Module(nj.Module):
      def forward(self):
        a = self.sub('a', nj.Variable, jnp.zeros, (), jnp.int32)
        a.write(a.read() + 1)
        b = self.sub('b', nj.Variable, a.read)
        return b.read()
    module = Module(name='module')
    state = nj.init(nj.pure(module.forward))({})
    assert state == {'module/a/value': 0, 'module/b/value': 0}
