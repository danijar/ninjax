import flax.linen as nn
import haiku as hk
import jax
import jax.numpy as jnp
import ninjax as nj
import optax

MLP = nj.FromHaiku(hk.nets.MLP)
Linear = nj.FromFlax(nn.Dense)
Adam = nj.FromOptax(optax.adam)


class Module(nj.Module):

  def __init__(self, size):
    self.mlp = MLP([128, 128], name='mlp')
    self.out = Linear(size, name='out')
    self.opt = Adam(1e-3, name='opt')

  def __call__(self, x):
    x = self.mlp(x)
    x = jax.nn.relu(x)
    x = self.out(x)
    return x

  def train(self, x, y):
    return self.opt(self.loss, [self.mlp, self.out], x, y)[0]

  def loss(self, x, y):
    return ((self(x) - y) ** 2).mean()


def main():
  x = jnp.ones((16, 8), jnp.float32)
  y = jnp.ones((16, 4), jnp.float32)
  model = Module(4, name='module')
  state = nj.init(model.train)({}, x, y, seed=0)
  train = jax.jit(nj.pure(model.train))
  for step in range(5):
    state, loss = train(state, x, y)
    print('Loss:', float(loss))


if __name__ == '__main__':
  main()
