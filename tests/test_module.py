import jax
import jax.numpy as jnp
import ninjax as nj


class TestModule:

  def test_basic(self):
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
    foo = Foo(name='foo')
    assert foo.path == 'foo'
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 8
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 13
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 18
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}

  def test_cloudpickle(self):
    import cloudpickle
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
    foo = Foo(name='foo')
    assert foo.path == 'foo'
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 8
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    result, params = nj.pure(foo.method)(params, rng)
    assert result == 13
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    foo2 = cloudpickle.loads(cloudpickle.dumps(foo))
    assert foo2.path == 'foo'
    result, params = nj.pure(foo2.method)(params, rng)
    assert result == 18
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
