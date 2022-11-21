import jax as jax
import jax.numpy as jnp
import ninjax as nj


class TestModule:

  def test_basic(self):
    nj.reset()
    class Foo(nj.Module):
      def __init__(self):
        self.bar = nj.Variable(jnp.float32, 5, name='bar')
      def method(self):
        baz = self.get('baz', jnp.float32, 3)
        result = self.bar.read() + baz
        self.put('baz', result)
        return result
    params = {}
    rng = jax.random.PRNGKey(0)
    foo = Foo()
    assert foo.path == '/Foo'
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 8
    assert set(params.keys()) == {'/Foo/bar/value', '/Foo/baz'}
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 13
    assert set(params.keys()) == {'/Foo/bar/value', '/Foo/baz'}
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 18
    assert set(params.keys()) == {'/Foo/bar/value', '/Foo/baz'}

  def test_cloudpickle(self):
    import cloudpickle
    nj.reset()
    class Foo(nj.Module):
      def __init__(self):
        self.bar = nj.Variable(jnp.float32, 5, name='bar')
      def method(self):
        baz = self.get('baz', jnp.float32, 3)
        result = self.bar.read() + baz
        self.put('baz', result)
        return result
    params = {}
    rng = jax.random.PRNGKey(0)
    foo = Foo()
    assert foo.path == '/Foo'
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 8
    assert set(params.keys()) == {'/Foo/bar/value', '/Foo/baz'}
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 13
    assert set(params.keys()) == {'/Foo/bar/value', '/Foo/baz'}
    foo2 = cloudpickle.loads(cloudpickle.dumps(foo))
    assert foo2.path == '/Foo'
    result, params = nj.pure(foo2.method)(params, rng)
    assert result == 18
    assert set(params.keys()) == {'/Foo/bar/value', '/Foo/baz'}
