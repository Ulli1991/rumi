from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from rumi.cpu import integrate_logh
from rumi.reference import diagnostics
from rumi import State


@pytest.fixture(scope="session")
def cpu_library(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    suffix = ".dylib" if sys.platform == "darwin" else ".so"
    out = tmp_path_factory.mktemp("native")/("librumi_cpu"+suffix)
    subprocess.run([sys.executable, str(root/"scripts/build_cpu.py"), "--output", str(out)], check=True)
    return out


@pytest.mark.parametrize("e", [0.0, 0.7, 0.95])
def test_native_logh_orbit_and_com_boost(cpu_library, e):
    r, speed = 1+e, np.sqrt((1-e)/(1+e))
    boost = np.array([3.0, -2.0, 0.5])
    x = np.array([[-r/2, 0, 0], [r/2, 0, 0]])
    v = np.array([[0, -speed/2, 0], [0, speed/2, 0]])+boost
    m, dt = np.array([0.5, 0.5]), 2*np.pi
    ox, ov = integrate_logh(x, v, m, dt, tolerance=1e-12, library=cpu_library)
    np.testing.assert_allclose(ox-dt*boost, x, atol=3e-7)
    np.testing.assert_allclose(ov, v, atol=3e-7)
    # Internal energy is independent of the large COM kinetic energy.
    before, after = State.create(x, v-boost, m), State.create(ox-dt*boost, ov-boost, m)
    assert abs(diagnostics(after)["energy"]/diagnostics(before)["energy"]-1) < 2e-9


def test_native_refuses_nonfinite_and_incomplete(cpu_library):
    x, v, m = np.array([[-0.5, 0, 0], [0.5, 0, 0]]), np.array([[0, -0.5, 0], [0, 0.5, 0]]), np.ones(2)/2
    with pytest.raises(RuntimeError, match="did not reach"):
        integrate_logh(x, v, m, 10, max_steps=1, library=cpu_library)
    with pytest.raises(ValueError, match="finite"):
        integrate_logh(x*np.nan, v, m, 1, library=cpu_library)
    with pytest.raises(ValueError):
        integrate_logh(x, v, m, np.inf, library=cpu_library)


def test_native_unequal_mass_triple_composition(cpu_library):
    x = np.array([[-0.4, 0, 0], [0.5, 0, 0], [0.1, 2, 0.2]])
    v = np.array([[0, -0.3, 0], [0, 0.25, 0], [-0.05, 0, 0]])
    m = np.array([0.7, 0.5, 0.1])
    initial = State.create(x, v, m)
    one_x, one_v = integrate_logh(x, v, m, 0.2, tolerance=1e-12, library=cpu_library)
    split_x, split_v = x, v
    for _ in range(4):
        split_x, split_v = integrate_logh(split_x, split_v, m, 0.05, tolerance=1e-12, library=cpu_library)
    np.testing.assert_allclose(one_x, split_x, atol=3e-10)
    np.testing.assert_allclose(one_v, split_v, atol=3e-10)
    before, after = diagnostics(initial), diagnostics(State.create(one_x, one_v, m))
    assert abs(after["energy"]/before["energy"]-1) < 2e-10
    np.testing.assert_allclose(after["momentum"], before["momentum"], atol=3e-14)


def test_native_unbound_flyby(cpu_library):
    x = np.array([[-0.5, 0, 0], [0.5, 0, 0]])
    v = np.array([[0, -1.5, 0], [0, 1.5, 0]])
    m = np.ones(2)/2
    ox, ov = integrate_logh(x, v, m, 1.0, tolerance=1e-12, library=cpu_library)
    before, after = diagnostics(State.create(x, v, m)), diagnostics(State.create(ox, ov, m))
    assert np.linalg.norm(ox[1]-ox[0]) > 1
    assert abs(after["energy"]/before["energy"]-1) < 2e-10
