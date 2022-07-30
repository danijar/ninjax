import jax as jax
import jax.numpy as jnp
import ninjax as nj


class TestScan:

  def test_constants(self):
    nj.reset()
    def program():
      def f(carry, x):
        carry = carry + x
        return carry, carry
      return nj.scan(f, 0, jnp.array([1, 2, 3, 4, 5]))
    (carry, ys), _ = nj.run(program, {}, jax.random.PRNGKey(0))
    assert carry == 15
    assert list(ys) == [1, 3, 6, 10, 15]

  def test_variables(self):
    nj.reset()
    v = nj.Variable(jnp.zeros, (), jnp.int32)
    def program():
      def f(carry, x):
        y = x + v.read()
        v.write(y)
        return y, carry + 1
      return nj.scan(f, 0, jnp.array([1, 1, 1]))
    rng = jax.random.PRNGKey(0)
    _, state = nj.run(v.read, {}, rng)  # Initialize variable.
    (carry, ys), _ = nj.run(program, state, rng)
    assert carry == 3
    assert list(ys) == [1, 2, 3]

  def test_rng(self):
    nj.reset()
    def program():
      def f(carry, x):
        return carry, nj.rng()
      return nj.scan(f, 0, jnp.array([1, 1, 1]))[1]
    keys, _ = nj.run(program, {}, jax.random.PRNGKey(0))
    assert jnp.array(keys).shape == (3, 2)
    assert (keys[0] != keys[1]).all()
    assert (keys[0] != keys[2]).all()
    assert (keys[1] != keys[2]).all()
