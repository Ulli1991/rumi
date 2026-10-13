import numpy as np
import pytest

from rumi import State
from rumi.hybrid import HybridFSI
from rumi.reference import diagnostics
from test_cpu import cpu_library


def mixed_state():
    x = np.array([[-0.025, 0, 0], [0.025, 0, 0], [2, 0.4, 0], [-2, 0, 1]])
    v = np.array([[0, -np.sqrt(1/0.05)/2, 0], [0, np.sqrt(1/0.05)/2, 0], [0, 0.1, 0.2], [0.1, 0, 0]])
    return State.create(x, v, [0.5, 0.5, 0.02, 0.03], subsystem=[0, 0, -1, -1])


def test_overlap_and_sequential_are_same_physical_update(cpu_library):
    initial, finals = mixed_state(), []
    for overlap in (False, True):
        with HybridFSI(initial, library=cpu_library, overlap=overlap) as runner:
            state = initial
            for _ in range(8):
                state = runner.step(state, 1e-4)
            finals.append(state)
    np.testing.assert_allclose(finals[0].position, finals[1].position, atol=3e-14, rtol=3e-14)
    np.testing.assert_allclose(finals[0].velocity, finals[1].velocity, atol=3e-13, rtol=3e-13)
    before, after = diagnostics(initial), diagnostics(finals[1])
    assert abs(after["energy"]/before["energy"]-1) < 2e-8
    np.testing.assert_allclose(after["momentum"], before["momentum"], atol=2e-12)


def test_hybrid_rejects_topology_changes(cpu_library):
    initial = mixed_state()
    with HybridFSI(initial, library=cpu_library) as runner:
        with pytest.raises(ValueError, match="topology"):
            runner.step(initial._replace(mass=initial.mass*2), 0.01)
