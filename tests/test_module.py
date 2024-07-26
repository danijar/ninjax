import jax.numpy as jnp
import ninjax as nj


class TestModule:

  def test_basic(self):
    class Foo(nj.Module):
      def __init__(self):
        self.bar = nj.Variable(jnp.float32, 5, name='bar')
      def method(self):
        baz = self.value('baz', jnp.float32, 3)
        result = self.bar.read() + baz
        self.write('baz', result)
        return result
    params = {}
    foo = Foo(name='foo')
    assert foo.path == 'foo'
    params, result = nj.pure(foo.method)(params, create=True)
    assert result == 8
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    params, result = nj.pure(foo.method)(params)
    assert result == 13
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    params, result = nj.pure(foo.method)(params)
    assert result == 18
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}

  def test_cloudpickle(self):
    import cloudpickle
    class Foo(nj.Module):
      def __init__(self):
        self.bar = nj.Variable(jnp.float32, 5, name='bar')
      def method(self):
        baz = self.value('baz', jnp.float32, 3)
        result = self.bar.read() + baz
        self.write('baz', result)
        return result
    params = {}
    foo = Foo(name='foo')
    assert foo.path == 'foo'
    params, result = nj.pure(foo.method)(params, create=True)
    assert result == 8
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    params, result = nj.pure(foo.method)(params)
    assert result == 13
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}
    foo2 = cloudpickle.loads(cloudpickle.dumps(foo))
    assert foo2.path == 'foo'
    params, result = nj.pure(foo2.method)(params)
    assert result == 18
    assert set(params.keys()) == {'foo/bar/value', 'foo/baz'}

  def test_scopes(self):
    with nj.scope('foo'):
      with nj.scope('bar'):
        v = nj.Variable(jnp.float32, 5, name='baz')
    assert v.path == 'foo/bar/baz'
    class Foo(nj.Module):
      def method(self):
        from ninjax import ninjax as nj
        with nj.scope('bar'):
          var = self.sub('baz', nj.Variable, jnp.float32, 5)
          assert var.path == 'foo/bar/baz'
          self.value('bav', jnp.float32, 5)
        var.read()
    foo = Foo(name='foo')
    state = nj.init(foo.method)({})
    assert foo.sub('baz').path == 'foo/bar/baz'
    assert set(state.keys()) == {
        'foo/bar/baz/value',
        'foo/bar/bav',
    }
