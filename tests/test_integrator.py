import jax
import numpy as np
import pytest

from rumi import State
from rumi.integrator import make_fsi_step
from rumi.reference import diagnostics


def binary(e=0.6, subsystem=False):
    r, speed = 1+e, np.sqrt((1-e)/(1+e))
    return State.create([[-r/2, 0, 0], [r/2, 0, 0]], [[0, -speed/2, 0], [0, speed/2, 0]],
                        [0.5, 0.5], subsystem=[0, 0] if subsystem else None)


def test_eccentric_orbit_fourth_order_and_conservation():
    initial = binary()
    step = make_fsi_step()
    errors = []
    for n in (128, 256, 512):
        dt = 2*np.pi/n
        def body(state, _):
            return step.compiled(state, dt), None
        final, _ = jax.jit(lambda state: jax.lax.scan(body, state, None, length=n))(initial)
        errors.append(float(np.linalg.norm(np.asarray(final.position)-np.asarray(initial.position))))
        d = diagnostics(final)
        np.testing.assert_allclose(d["momentum"], 0, atol=1e-13)
        np.testing.assert_allclose(d["angular_momentum"], diagnostics(initial)["angular_momentum"], atol=1e-13)
        assert abs(d["energy"]/diagnostics(initial)["energy"]-1) < 1e-4
    orders = np.log2(np.asarray(errors[:-1])/errors[1:])
    assert np.all((orders > 3.6) & (orders < 4.4)), (errors, orders)


def test_reversibility_of_fixed_step():
    state = binary(0.3)
    step = make_fsi_step()
    advanced = step(state, 0.01)
    # Reversal of velocities converts a forward symmetric step into the inverse flow.
    reversed_state = advanced._replace(velocity=-advanced.velocity)
    restored = step(reversed_state, 0.01)
    np.testing.assert_allclose(restored.position, state.position, atol=2e-15)
    np.testing.assert_allclose(restored.velocity, -state.velocity, atol=2e-15)


def test_invalid_step_and_subsystem_refused():
    step = make_fsi_step()
    for dt in (0, -1, np.nan, np.inf):
        with pytest.raises(ValueError):
            step(binary(), dt)
    with pytest.raises(ValueError, match="HybridFSI"):
        step(binary(subsystem=True), 0.01)
