import functools

import jax
import jax.numpy as jnp
import ninjax as nj


class TestCheckpoint:

  def test_nostate(self):
    @nj.pure
    @nj.checkpoint
    def program(x):
      y = x + 1.0
      return y
    gradfn = jax.value_and_grad(lambda x: program({}, x)[1])
    gradfn = jax.jit(gradfn)
    y, dx = gradfn(12.0)
    assert y == 13
    assert dx == 1

  def test_state(self):

    @functools.partial(nj.checkpoint, static_argnums=(1,))
    def layer(x, name):
      v = nj.Variable(jnp.ones, (), name=name)
      x = x + v.read()
      return x

    def model(x):
      x = layer(x, 'a')
      x = layer(x, 'b')
      return x

    @nj.pure
    def program(x):
      gradfn = nj.grad(model, ['a/value', 'b/value'])
      out, params, grad = gradfn(x)
      return out, params, grad

    state = nj.init(program)({}, 0.0)
    _, (out, params, grad) = jax.jit(program)(state, 12.0)
    assert out == 14.0
    assert grad == {'a/value': 1.0, 'b/value': 1.0}
    assert params == {'a/value': 1.0, 'b/value': 1.0}
