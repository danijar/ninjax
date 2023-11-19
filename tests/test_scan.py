import jax
import jax.numpy as jnp
import ninjax as nj


class TestScan:

  def test_constants(self):
    def program():
      def body(carry, x):
        carry = carry + x
        return carry, carry
      return nj.scan(body, 0, jnp.array([1, 2, 3, 4, 5]))
    (carry, ys), _ = nj.pure(program)({}, jax.random.PRNGKey(0))
    assert carry == 15
    assert list(ys) == [1, 3, 6, 10, 15]

  def test_read_state(self):
    v = nj.Variable(jnp.zeros, (), jnp.int32, name='v')
    def program():
      def body(carry, x):
        y = x + v.read()
        return carry + 1, y
      return nj.scan(body, 0, jnp.array([1, 1, 1]), modify=False)
    rng = jax.random.PRNGKey(0)
    (carry, ys), _ = nj.pure(program)({}, rng)
    assert carry == 3
    assert list(ys) == [1, 1, 1]

  def test_write_state(self):
    v = nj.Variable(jnp.zeros, (), jnp.int32, name='v')
    def program():
      def body(carry, x):
        y = x + v.read()
        v.write(y)
        return carry + 1, y
      return nj.scan(body, 0, jnp.array([1, 1, 1]), modify=True)
    rng = jax.random.PRNGKey(0)
    (carry, ys), _ = nj.pure(program)({}, rng)
    assert carry == 3
    assert list(ys) == [1, 2, 3]

  def test_rng(self):
    def program():
      def body(carry, x):
        return carry, nj.rng()
      return nj.scan(body, 0, jnp.array([1, 1, 1]))[1]
    keys, _ = nj.pure(program)({}, jax.random.PRNGKey(0))
    assert jnp.array(keys).shape == (3, 2)
    assert (keys[0] != keys[1]).all()
    assert (keys[0] != keys[2]).all()
    assert (keys[1] != keys[2]).all()
