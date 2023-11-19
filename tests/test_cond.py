import jax
import jax.numpy as jnp
import ninjax as nj


class TestCond:

  def test_basic(self):
    def program(x):
      return nj.cond(x, lambda: 12, lambda: 42)
    rng = jax.random.PRNGKey(0)
    assert nj.pure(program)({}, rng, True)[0] == 12
    assert nj.pure(program)({}, rng, False)[0] == 42

  def test_state(self):
    v = nj.Variable(jnp.ones, (), jnp.int32, name='v')
    w = nj.Variable(jnp.ones, (), jnp.int32, name='w')
    def program(x):
      def true_fn():
        return v.read() + w.read()
      def false_fn():
        v.write(42)
        return 12
      return nj.cond(x, true_fn, false_fn)
    state = {}
    rng = jax.random.PRNGKey(0)
    out, state = nj.pure(program)(state, rng, True)
    assert out == 2
    out, state = nj.pure(program)(state, rng, False)
    assert out == 12
    out, state = nj.pure(program)(state, rng, True)
    assert out == 43

  def test_branch_rngs(self):
    def program(x):
      return nj.cond(x, nj.rng, nj.rng)
    rng = jax.random.PRNGKey(0)
    key1, _ = nj.pure(program)({}, rng, True)
    key2, _ = nj.pure(program)({}, rng, False)
    assert jnp.array(key1).shape == (2,)
    assert jnp.array(key2).shape == (2,)
    assert (key1 != key2).all()
