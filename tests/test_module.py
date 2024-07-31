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

  def test_value(self):
    class Module(nj.Module):
      def method(self):
        x = self.value('value', jnp.zeros, (), jnp.float32)
        self.write('value', x + 1)
        return self.read('value')
    module = Module(name='module')
    state = nj.init(module.method)({})
    for reference in range(1, 5):
      state, value = nj.pure(module.method)(state)
      assert state == {'module/value': float(reference)}
      assert value == float(reference)

  def test_scopes(self):
    with nj.scope('foo'):
      with nj.scope('bar'):
        v = nj.Variable(jnp.float32, 5, name='baz')
    assert v.path == 'foo/bar/baz'

    class Foo(nj.Module):
      def method(self):
        from ninjax import ninjax as nj
        with nj.scope('scope'):
          sub = self.sub('sub1', nj.Variable, jnp.float32, 5)
          sub.read()
          assert sub.path == 'foo/scope/sub1'
        self.value('value', jnp.float32, 5)
      @nj.scope('method2')
      def method2(self):
        sub = self.sub('sub2', nj.Variable, jnp.float32, 2)
        sub.read()

    foo = Foo(name='foo')
    state = {}
    state = nj.init(foo.method)(state)
    state = nj.init(foo.method2)(state)
    assert set(state.keys()) == {
        'foo/scope/sub1/value',
        'foo/value',
        'foo/method2/sub2/value',
    }
