import jax
import jax.numpy as jnp
import ninjax as nj


class TestGrad:

  def test_has_aux(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    rng = jax.random.PRNGKey(0)
    _, state = nj.pure(w.read)({}, rng)
    @nj.pure
    def fun(x):
      return nj.grad(lambda x: (x * w.read(), 42), [w], has_aux=True)(x)
    (y, x, dx, aux), state = fun(state, rng, jnp.array(2.0))
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
      return nj.grad(lambda x: x * w.read(), w.getm())(x)
    rng = jax.random.PRNGKey(0)
    (y, x, dx), state = nj.pure(program)({}, rng, jnp.array(2.0))
    assert y == jnp.array(1.0)
    assert state['w/value'] == jnp.array(0.5)
    assert x['w/value'] == jnp.array(0.5)
    assert dx['w/value'] == jnp.array(2.0)

  def test_create_state_by_modules(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      return nj.grad(lambda x: x * w.read(), [w])(x)
    rng = jax.random.PRNGKey(0)
    (y, x, dx), state = nj.pure(program)({}, rng, jnp.array(2.0))
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
    rng = jax.random.PRNGKey(0)
    fun = nj.jit(nj.pure(program))
    state = {}
    for _ in range(3):
      (y, x, dx), state = fun(state, rng, jnp.array(2.0))
      assert y == jnp.array(1.0)
      assert state['w/value'] == jnp.array(0.5)
      assert x['w/value'] == jnp.array(0.5)
      assert dx['w/value'] == jnp.array(2.0)

  def test_create_state_pmap(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      def forward(x):
        return x * w.read()
      return nj.grad(forward, [w])(x)
    rng = jax.random.PRNGKey(0)[None]
    fun = nj.pmap(nj.pure(program))
    state = {}
    for _ in range(3):
      (y, x, dx), state = fun(state, rng, jnp.array([2.0]))
      assert y == jnp.array([1.0])
      assert state['w/value'] == jnp.array([0.5])
      assert x['w/value'] == jnp.array([0.5])
      assert dx['w/value'] == jnp.array([2.0])

  def test_side_effect(self):
    counter = nj.Variable(jnp.array, 0, name='counter')
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      def forward(x):
        counter.write(counter.read() + 1)
        return x * w.read()
      nj.grad(forward, [w])(x)
      return counter.read()
    rng = jax.random.PRNGKey(0)
    fun = nj.pure(program)
    state = {}
    for index in range(1, 4):
      value, state = fun(state, rng, jnp.array(2.0))
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
    rng = jax.random.PRNGKey(0)
    fun = nj.jit(nj.pure(program))
    state = {}
    for index in range(1, 4):
      value, state = fun(state, rng, jnp.array(2.0))
      assert value == index
