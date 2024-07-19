import jax
import pytest


@pytest.fixture(scope='session', autouse=True)
def setup_jax(request):
  jax.config.update('jax_platforms', 'cpu')
