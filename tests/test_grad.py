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

  def test_create_state_wrt_names(self):
    w = nj.Variable(jnp.array, 0.5, name='w')
    def program(x):
      if nj.creating():
        w.read()  # Create state entry.
      wrt = {w.path + '/' + k: v for k, v in w.values.items()}
      return nj.grad(lambda x: x * w.read(), wrt)(x)
    state, (y, x, dx) = nj.pure(program)({}, jnp.array(2.0), create=True)
    assert y == jnp.array(1.0)
    assert state['w/value'] == jnp.array(0.5)
    assert x['w/value'] == jnp.array(0.5)
    assert dx['w/value'] == jnp.array(2.0)

  def test_create_state_wrt_modules(self):
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

  def test_side_effect_value(self):
    class Module(nj.Module):
      def inner(self, x):
        self.value('value', jnp.zeros, (), jnp.float32)
        self.write('value', self.read('value') + 1)
        return x
      def outer(self, x):
        nj.grad(self.inner, [self.path + '/value'])(x)
        return self.read('value')
    module = Module(name='module')
    state = nj.init(module.outer)({}, 1.0)
    assert state == {'module/value': 0}
    for count in range(1, 4):
      state, output = nj.pure(module.outer)(state, 1.0)
      assert state == {'module/value': count}
      assert output == count

  def test_side_effect_return(self):
    class Module(nj.Module):
      def loss(self):
        self.value('value', jnp.zeros, (), jnp.float32)
        self.write('value', self.read('value') + 1)
        return 0.0
      def update(self):
        key = self.path + '/value'
        loss, inps, grads = nj.grad(self.loss, [key])()
        nj.context()[key] = inps[key] - 0.1 * grads[key]
        return loss
    module = Module(name='module')
    state = nj.init(module.update)({})
    assert state == {'module/value': 0}
    for count in range(1, 4):
      state, _ = nj.pure(module.update)(state)
      assert state == {'module/value': count}

  def test_side_effect_variable(self):
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

  def test_nested_create(self):
    def program(a, b):
      b = jnp.float32(b)
      def forward(a, b):
        def truefn(b):
          w = nj.Variable(jnp.array, 1.0, name='w')
          return w.read()
        def falsefn(b):
          return b
        return nj.cond(a, truefn, falsefn, b)
      return nj.grad(forward, ['w/value'])(a, b)
    state = nj.init(program)({}, False, 0.0)
    assert state == {'w/value': 1.0}
    fun = jax.jit(nj.pure(program))
    _, (y, ws, dw) = fun(state, True, -12)
    assert y == 1
    assert ws == {'w/value': 1}
    assert dw == {'w/value': 1}
    _, (y, ws, dw) = fun(state, False, 42)
    assert y == 42
    assert ws == {'w/value': 1}
    assert dw == {'w/value': 0}
