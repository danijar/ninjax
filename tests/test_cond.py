import jax as jax
import jax.numpy as jnp
import ninjax as nj


class TestCond:

  def test_constants(self):
    nj.reset()
    def program(x):
      return nj.cond(x, lambda: 12, lambda: 42)
    assert nj.run(program, {}, jax.random.PRNGKey(0), True)[0] == 12
    assert nj.run(program, {}, jax.random.PRNGKey(0), False)[0] == 42

  def test_variables(self):
    nj.reset()
    v = nj.Variable(jnp.ones, (), jnp.int32)
    def program(x):
      def true_fn():
        return v.read()
      def false_fn():
        v.write(42)
        return 12
      return nj.cond(x, true_fn, false_fn)
    state = {}
    rng = jax.random.PRNGKey(0)
    out, state = nj.run(program, state, rng, True)
    assert out == 1
    out, state = nj.run(program, state, rng, False)
    assert out == 12
    out, state = nj.run(program, state, rng, True)
    assert out == 42

  def test_rng(self):
    nj.reset()
    def program(x):
      def true_fn():
        return nj.rng()
      def false_fn():
        return nj.rng()
      return nj.cond(x, true_fn, false_fn)
    key1, _ = nj.run(program, {}, jax.random.PRNGKey(0), True)
    key2, _ = nj.run(program, {}, jax.random.PRNGKey(0), False)
    assert jnp.array(key1).shape == (2,)
    assert jnp.array(key2).shape == (2,)
    assert (key1 != key2).all()

