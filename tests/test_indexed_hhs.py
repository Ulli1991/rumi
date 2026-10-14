import jax
import jax.numpy as jnp
import numpy as np
import pytest

from rumi import State
from rumi.hhs import HHS, HHSOptions
from rumi.indexed_hhs import DeviceHHS, make_indexed_kernel
from rumi.interactions import make_interactions, make_pair_steps, take, symmetrized_step


def mixed_state(n=9):
    x = np.zeros((n, 3))
    x[:, 0] = 10 + np.arange(n)*3
    x[:2, 0] = [-.025, .025]
    v = np.zeros_like(x)
    v[:2, 1] = [-np.sqrt(20)/2, np.sqrt(20)/2]
    return State.create(x, v, np.r_[.5, .5, np.full(n-2, .001)])


@pytest.mark.parametrize("backend", ["xla", "pallas"])
def test_active_rectangles_and_padding(backend):
    o = HHSOptions(softening=.02)
    if backend == "pallas":
        from rumi.indexed_pallas import make_indexed_kernel as make
        kernel = make(o, interpret=True, sink_tile=4, source_tile=4, programs=2)
    else:
        kernel = make_indexed_kernel(o, tile=4)
    s = mixed_state(9)
    ti = jnp.array([7, 2, 0, 8, 1, 6, 5, 4, 3], jnp.int32)
    sj = jnp.array([1, 5, 0, 3, 8, 2, 4, 6, 7], jnp.int32)
    force, gradient = make_interactions(G=o.G, softening=o.softening)
    a = force(s, s)
    for nt, ns in ((0, 0), (3, 0), (1, 2), (5, 3), (9, 9)):
        t, source = take(s, ti[:nt]), take(s, sj[:ns])
        for mode, expected in (("force", force(t, source)),
                               ("gradient", gradient(t, source, a[ti[:nt]], a[sj[:ns]])),
                               ("steps", make_pair_steps(softening=o.softening)(t, source)[:, None])):
            actual = jax.jit(lambda: kernel(s, ti, jnp.int32(nt), sj, jnp.int32(ns), a, mode))()
            np.testing.assert_allclose(actual[:nt], expected, rtol=2e-13, atol=1e-13)
            np.testing.assert_array_equal(actual[nt:], 0)


@pytest.mark.parametrize("dt", [.00001, .01])
def test_device_schedule_matches_recursive_hhs(dt):
    s = mixed_state()
    o = HHSOptions(eta_ff=.05, eta_fb=.05)
    device, reference = DeviceHHS(o), HHS(o)
    for _ in range(3):
        expected = reference.step(s, dt)
        actual, work = device.step(s, dt)
        np.testing.assert_allclose(actual.position, expected.position, rtol=1e-13, atol=2e-14)
        np.testing.assert_allclose(actual.velocity, expected.velocity, rtol=1e-12, atol=2e-13)
        assert int(work.depth) == reference.max_depth
        s = actual


def test_deep_work_is_independent_of_inactive_population():
    o = HHSOptions(eta_ff=.05, eta_fb=.05)
    traces = []
    for n in (9, 33):
        _, work = DeviceHHS(o).step(mixed_state(n), .01)
        assert int(work.depth) >= 2
        traces.append(np.asarray(work.pairs_by_depth))
    np.testing.assert_array_equal(traces[0][1:], traces[1][1:])
    assert traces[1][0] > traces[0][0]


def test_failures_are_explicit():
    s = mixed_state()
    with pytest.raises(RuntimeError, match="stack exhausted"):
        DeviceHHS(HHSOptions(eta_ff=.0001, eta_fb=.0001), stack_depth=2).step(s, .01)
    with pytest.raises(RuntimeError, match="call limit"):
        DeviceHHS(HHSOptions(eta_ff=.05, eta_fb=.05, max_calls=1)).step(s, .01)
    grouped = s._replace(subsystem=s.subsystem.at[:2].set(0))
    with pytest.raises(RuntimeError, match="membership unsupported"):
        DeviceHHS().step(grouped, .01)


def test_pallas_interpreter_complete_schedule():
    s = mixed_state(5)
    o = HHSOptions(eta_ff=.1, eta_fb=.1)
    expected, ew = DeviceHHS(o).step(s, .005)
    actual, aw = DeviceHHS(o, backend="pallas", interpret=True).step(s, .005)
    np.testing.assert_allclose(actual.position, expected.position, rtol=2e-13, atol=1e-14)
    np.testing.assert_allclose(actual.velocity, expected.velocity, rtol=2e-13, atol=1e-14)
    np.testing.assert_array_equal(aw.pairs_by_depth, ew.pairs_by_depth)


def test_floor_and_single_particle():
    for s in (mixed_state(2), State.create([[1., 0, 0]], [[.2, 0, 0]], [1.])):
        o = HHSOptions(dt_min=.1)
        expected = HHS(o).step(s, .01)
        actual, work = DeviceHHS(o).step(s, .01)
        np.testing.assert_allclose(actual.position, expected.position, atol=1e-14)
        np.testing.assert_allclose(actual.velocity, expected.velocity, atol=1e-14)
        assert int(work.depth) == 0


def test_symmetrization_retains_positive_approaching_pair_limit():
    value = symmetrized_step(jnp.float64(1), jnp.float64(-1e20), .2)
    np.testing.assert_allclose(value, 1e-20, rtol=2e-15, atol=0)


@pytest.mark.skipif(jax.default_backend() == "cpu", reason="requires actual GPU")
@pytest.mark.gpu
def test_pallas_hhs_on_gpu():
    s = mixed_state(33)
    o = HHSOptions(eta_ff=.05, eta_fb=.05)
    expected, _ = DeviceHHS(o).step(s, .01)
    actual, _ = DeviceHHS(o, backend="pallas").step(s, .01)
    np.testing.assert_allclose(actual.position, expected.position, rtol=2e-12, atol=1e-13)
    np.testing.assert_allclose(actual.velocity, expected.velocity, rtol=2e-12, atol=1e-13)
