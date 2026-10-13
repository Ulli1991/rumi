import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rumi.forces import make_force_functions
from rumi import reference


@pytest.mark.parametrize("backend,interpret", [("xla", False), ("pallas", True)])
@pytest.mark.parametrize("n", [1, 7, 33])
@pytest.mark.parametrize("softening", [0.0, 0.2])
def test_direct_and_gradient_against_pair_reference(backend, interpret, n, softening):
    rng = np.random.default_rng(7)
    x, m = rng.normal(size=(n, 3)), rng.uniform(0.2, 2, n)
    sid = np.full(n, -1, dtype=np.int32)
    if n >= 7:
        sid[:3] = 0
        sid[3:5] = 1
    acc, grad = make_force_functions(backend=backend, interpret=interpret, softening=softening,
                                    sink_tile=4, source_tile=8)
    args = tuple(jnp.asarray(a) for a in (x, m, sid))
    a = acc(*args)
    q = grad(*args, a)
    expected = reference.acceleration(x, m, softening=softening, subsystem=sid)
    np.testing.assert_allclose(a, expected, rtol=3e-14, atol=3e-14)
    np.testing.assert_allclose(q, reference.gradient(x, m, expected, softening=softening, subsystem=sid),
                               rtol=3e-13, atol=3e-12)
    np.testing.assert_allclose(np.sum(m[:, None]*np.asarray(a), axis=0), 0, atol=2e-13)


def test_explicit_pallas_does_not_fall_back_on_cpu():
    if jax.default_backend() == "cpu":
        with pytest.raises(RuntimeError, match="GPU requested"):
            make_force_functions(backend="pallas")


def test_gradient_is_directional_derivative_of_force():
    rng = np.random.default_rng(31)
    x, m = rng.normal(size=(7, 3)), rng.uniform(0.1, 1, 7)
    a = reference.acceleration(x, m, softening=0.1)
    h = 1e-6
    fd = (reference.acceleration(x+h*a, m, softening=0.1)
          - reference.acceleration(x-h*a, m, softening=0.1))/(2*h)
    np.testing.assert_allclose(reference.gradient(x, m, a, softening=0.1), fd, rtol=2e-8, atol=2e-8)


@pytest.mark.gpu
def test_compiled_gpu_pallas():
    if jax.default_backend() == "cpu":
        pytest.skip("requires real GPU; interpreter checks are separate")
    rng = np.random.default_rng(19)
    x, m, sid = jnp.asarray(rng.normal(size=(127, 3))), jnp.ones(127), jnp.full(127, -1)
    a, q = make_force_functions(backend="pallas")
    ax, qx = make_force_functions(backend="xla")
    np.testing.assert_allclose(a(x, m, sid), ax(x, m, sid), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(q(x, m, sid, a(x, m, sid)), qx(x, m, sid, ax(x, m, sid)),
                               rtol=1e-11, atol=1e-10)
