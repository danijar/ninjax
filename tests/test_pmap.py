import functools

import jax as jax
import jax.numpy as jnp
import ninjax as nj


class TestPmap:

  def test_plain_pmap(self):
    nj.reset()
    inner = nj.pure(lambda x: x ** 2)
    fun = jax.pmap(inner, 'devices')
    rng = jax.random.PRNGKey(0)[None]
    x = jnp.ones((1, 128))
    assert fun({}, rng, x)[0].shape == (1, 128)
    assert fun({}, rng, x)[0].shape == (1, 128)

  def test_ninjax_pmap(self):
    nj.reset()
    fun = nj.pmap(nj.pure(lambda x: x ** 2), 'devices')
    rng = jnp.repeat(jax.random.PRNGKey(0)[None], 1, 0)
    x = jnp.ones((1, 128))
    assert not fun.created
    assert fun({}, rng, x)[0].shape == (1, 128)
    assert fun.created
    assert fun({}, rng, x)[0].shape == (1, 128)
    assert fun.created

  def test_variables(self):
    nj.reset()
    v = nj.Variable(jnp.ones, (), jnp.int32)
    read = nj.pmap(nj.pure(v.read))
    write = nj.pmap(nj.pure(v.write))
    rng = jax.random.PRNGKey(0)[None]
    state = {}
    value, state = read(state, rng)
    assert value == jnp.array([1])
    assert state == {'/Variable/value': jnp.array([1])}
    value, state = write(state, rng, jnp.array([2]))
    assert value == jnp.array([2])
    assert state == {'/Variable/value': jnp.array([2])}
    value, state = write(state, rng, jnp.array([3]))
    assert value == jnp.array([3])
    assert state == {'/Variable/value': jnp.array([3])}
    value, state = read(state, rng)
    assert value == jnp.array([3])
    assert state == {'/Variable/value': jnp.array([3])}
