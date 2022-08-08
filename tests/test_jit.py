import functools

import jax as jax
import jax.numpy as jnp
import ninjax as nj


class TestJit:

  def test_plain_jit(self):
    nj.reset()
    run = jax.jit(functools.partial(nj.run, lambda: jnp.array(42)))
    assert run({}, jax.random.PRNGKey(0))[0] == 42
    assert run({}, jax.random.PRNGKey(0))[0] == 42

  def test_ninjax_jit(self):
    nj.reset()
    inner = functools.partial(nj.run, lambda: jnp.array(42))
    run = nj.jit(inner)
    assert not hasattr(inner, '_initialized')
    assert run({}, jax.random.PRNGKey(0))[0] == 42
    assert inner._initialized
    assert run({}, jax.random.PRNGKey(0))[0] == 42
    assert inner._initialized

  def test_variables(self):
    nj.reset()
    v = nj.Variable(jnp.ones, (), jnp.int32)
    read = nj.jit(functools.partial(nj.run, v.read))
    write = nj.jit(functools.partial(nj.run, v.write))
    state = {}
    rng = jax.random.PRNGKey(0)
    _, state = read(state, rng)
    assert state == {'/Variable/value': 1}
    _, state = write(state, rng, 42)
    assert nj.run(v.read, state, rng)[0] == 42
    assert state == {'/Variable/value': 42}
