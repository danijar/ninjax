import jax.numpy as jnp
import ninjax as nj


class TestScan:

  def test_constants(self):
    def program():
      def body(carry, x):
        carry = carry + x
        return carry, carry
      return nj.scan(body, 0, jnp.array([1, 2, 3, 4, 5]))
    _, (carry, ys) = nj.pure(program)({})
    assert carry == 15
    assert list(ys) == [1, 3, 6, 10, 15]

  def test_read_state(self):
    v = nj.Variable(jnp.zeros, (), jnp.int32, name='v')
    def program():
      def body(carry, x):
        y = x + v.read()
        return carry + 1, y
      return nj.scan(body, 0, jnp.array([1, 1, 1]))
    _, (carry, ys) = nj.pure(program)({}, create=True)
    assert carry == 3
    assert list(ys) == [1, 1, 1]

  def test_write_state(self):
    v = nj.Variable(jnp.zeros, (), jnp.int32, name='v')
    def program():
      def body(carry, x):
        y = x + v.read()
        v.write(y)
        return carry + 1, y
      return nj.scan(body, 0, jnp.array([1, 1, 1]))
    _, (carry, ys) = nj.pure(program)({}, create=True)
    assert carry == 3
    assert list(ys) == [1, 2, 3]

  def test_seed(self):
    def program():
      def body(carry, x):
        return carry, nj.seed()
      return nj.scan(body, 0, jnp.array([1, 1, 1]))[1]
    _, keys = nj.pure(program)({}, seed=0, create=True)
    assert jnp.array(keys).shape == (3, 2)
    assert (keys[0] != keys[1]).all()
    assert (keys[0] != keys[2]).all()
    assert (keys[1] != keys[2]).all()
