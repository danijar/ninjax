import jax
import jax.numpy as jnp
import ninjax as nj
import pytest


class TestAttrs:

  def test_attrs(self):
    class MyModule(nj.Module):
      foo: int
      bar: float = 12.5
      baz: str = None
      other = True
    module = MyModule(foo=42, baz='baz', name='module')
    assert module.foo == 42
    assert module.bar == 12.5
    assert module.baz == 'baz'

  def test_class_variable(self):
    class MyModule(nj.Module):
      foo: int
      other = True
    module = MyModule(foo=42, name='module')
    assert module.foo == 42
    assert module.other == True
    with pytest.raises(TypeError):
      MyModule(other=False, name='module')

  def test_keyword_only(self):
    class MyModule(nj.Module):
      foo: int
    with pytest.raises(TypeError):
      module = MyModule(42, name='module')

  def test_with_default(self):
    class MyModule(nj.Module):
      foo: int = 42
    module = MyModule(name='module')
    assert module.foo == 42
    module = MyModule(foo=12, name='module')
    assert module.foo == 12

  def test_without_default(self):
    class MyModule(nj.Module):
      foo: int
    module = MyModule(foo=12, name='module')
    assert module.foo == 12
    with pytest.raises(TypeError):
      MyModule(name='module')

  def test_immutable(self):
    class MyModule(nj.Module):
      foo: int
    module = MyModule(foo=42, name='module')
    assert module.foo == 42
    with pytest.raises(AttributeError):
      module.foo = 12

  def test_type_check(self):
    class MyModule(nj.Module):
      foo: int
    with pytest.raises(TypeError):
      module = MyModule(foo=1.5, name='module')

  def test_with_init(self):
    class MyModule(nj.Module):
      foo: int
      def __init__(self, bar):
        self.bar = bar
    module = MyModule(12, foo=42, name='module')
    assert module.foo == 42
    assert module.bar == 12
    module = MyModule(foo=42, bar=12, name='module')
    assert module.foo == 42
    assert module.bar == 12
