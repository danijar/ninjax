import jax.numpy as jnp
import ninjax as nj


class TestCond:

  def test_basic(self):
    @nj.pure
    def program(x):
      return nj.cond(x, lambda: 12, lambda: 42)
    assert program({}, True)[1] == 12
    assert program({}, False)[1] == 42

  def test_state(self):
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    w = nj.Variable(jnp.ones, (), jnp.int32, name='w')
    @nj.pure
    def program(x):
      def true_fn():
        return v.read() + w.read()
      def false_fn():
        v.write(42)
        return 12
      return nj.cond(x, true_fn, false_fn)
    state = nj.init(program)({}, True)
    assert state == {'v/value': 1, 'w/value': 1}
    state, out = program(state, True)
    assert out == 2
    state, out = program(state, False)
    assert out == 12
    state, out = program(state, True)
    assert out == 43

  def test_seeds(self):
    @nj.pure
    def program(x):
      return nj.cond(x, nj.seed, nj.seed)
    _, draw1 = program({}, True, seed=0)
    _, draw2 = program({}, False, seed=0)
    assert jnp.array(draw1).shape == (2,)
    assert jnp.array(draw2).shape == (2,)
    assert (draw1 != draw2).all()
