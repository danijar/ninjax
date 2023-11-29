import jax
import jax.numpy as jnp
import ninjax as nj


class TestGrad:

  def test_has_aux(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    state = nj.init(w.read)({})
    @nj.pure
    def fun(x):
      return nj.grad(lambda x: (x * w.read(), 42), [w], has_aux=True)(x)
    state, (y, x, dx, aux) = fun(state, jnp.array(2.0), create=True)
    assert y == 1.0
    assert state['w/value'] == 0.5
    assert x['w/value'] == 0.5
    assert dx['w/value'] == 2.0
    assert aux == 42

  def test_create_state_by_keys(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      if nj.creating():
        w.read()  # Create state entry.
      return nj.grad(lambda x: x * w.read(), w.find())(x)
    state, (y, x, dx) = nj.pure(program)({}, jnp.array(2.0), create=True)
    assert y == jnp.array(1.0)
    assert state['w/value'] == jnp.array(0.5)
    assert x['w/value'] == jnp.array(0.5)
    assert dx['w/value'] == jnp.array(2.0)

  def test_create_state_by_modules(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      return nj.grad(lambda x: x * w.read(), [w])(x)
    state, (y, x, dx) = nj.pure(program)({}, jnp.array(2.0), create=True)
    assert y == jnp.array(1.0)
    assert state['w/value'] == jnp.array(0.5)
    assert x['w/value'] == jnp.array(0.5)
    assert dx['w/value'] == jnp.array(2.0)

  def test_create_state_jit(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      def forward(x):
        return x * w.read()
      return nj.grad(forward, [w])(x)
    fun = jax.jit(nj.pure(program), static_argnames=['create'])
    state = {}
    for _ in range(3):
      state, (y, x, dx) = fun(state, jnp.array(2.0), create=True)
      assert y == jnp.array(1.0)
      assert state['w/value'] == jnp.array(0.5)
      assert x['w/value'] == jnp.array(0.5)
      assert dx['w/value'] == jnp.array(2.0)

  def test_side_effect(self):
    counter = nj.Variable(jnp.array, 0, name='counter')
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      def forward(x):
        counter.write(counter.read() + 1)
        return x * w.read()
      nj.grad(forward, [w])(x)
      return counter.read()
    fun = nj.pure(program)
    state = {}
    for index in range(1, 4):
      state, value = fun(state, jnp.array(2.0), create=True)
      assert value == index
      assert state['counter/value'] == index

  def test_side_effect_jit(self):
    counter = nj.Variable(jnp.array, 0, name='counter')
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      def forward(x):
        counter.write(counter.read() + 1)
        return x * w.read()
      nj.grad(forward, [w])(x)
      return counter.read()
    fun = jax.jit(nj.pure(program), static_argnames=['create'])
    state = {}
    for index in range(1, 4):
      state, value = fun(state, jnp.array(2.0), create=True)
      assert value == index
