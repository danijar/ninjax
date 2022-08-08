import functools

import jax as jax
import jax.numpy as jnp
import ninjax as nj


class TestPmap:

  def test_plain_pmap(self):
    nj.reset()
    inner = functools.partial(nj.run, lambda x: x ** 2)
    run = jax.pmap(inner, 'devices')
    rng = jax.random.PRNGKey(0)[None]
    x = jnp.ones((1, 128))
    assert run({}, rng, x)[0].shape == (1, 128)
    assert run({}, rng, x)[0].shape == (1, 128)

  def test_ninjax_pmap(self):
    nj.reset()
    inner = functools.partial(nj.run, lambda x: x ** 2)
    run = nj.pmap(inner, 'devices')
    rng = jax.random.PRNGKey(0)[None]
    x = jnp.ones((1, 128))
    assert not hasattr(inner, '_initialized')
    assert run({}, rng, x)[0].shape == (1, 128)
    assert inner._initialized
    assert run({}, rng, x)[0].shape == (1, 128)
    assert inner._initialized

  def test_variables(self):
    nj.reset()
    v = nj.Variable(jnp.ones, (), jnp.int32)
    read = nj.pmap(functools.partial(nj.run, v.read))
    write = nj.pmap(functools.partial(nj.run, v.write))
    state = {}
    rng = jax.random.PRNGKey(0)[None]
    _, state = read(state, rng)
    assert state == {'/Variable/value': jnp.array([1])}
    _, state = write(state, rng, jnp.array([42]))
    assert nj.run(v.read, state, rng)[0] == jnp.array([42])
    assert state == {'/Variable/value': jnp.array([42])}
