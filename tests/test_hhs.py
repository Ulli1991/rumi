import numpy as np

from rumi import State
from rumi.hhs import HHS,HHSOptions
from rumi.interactions import take,make_interactions,make_pair_steps
from rumi.reference import diagnostics,acceleration


def test_rectangular_id_exclusion_and_force_reassembly():
    rng=np.random.default_rng(17)
    s=State.create(rng.normal(size=(7,3)),np.zeros((7,3)),rng.uniform(.1,1,7),ids=np.arange(7)+100)
    f,_=make_interactions()
    a=f(s,take(s,np.arange(3)))+f(s,take(s,np.arange(3,7)))
    np.testing.assert_allclose(a,acceleration(s.position,s.mass),atol=2e-14)
    single=take(s,np.array([2]))
    np.testing.assert_array_equal(f(single,single),0)


def test_symmetrized_pair_step_at_rest():
    s=State.create([[-.5,0,0],[.5,0,0]],np.zeros((2,3)),[.5,.5])
    dt=make_pair_steps(eta_ff=.2,eta_fb=.2)(s,s)
    np.testing.assert_allclose(dt,[.2,.2],rtol=2e-15)


def test_hhs_adaptive_energy_and_momentum():
    s=State.create([[-.8,0,0],[.8,0,0]],[[0,-.25,0],[0,.25,0]],[.5,.5])
    initial=diagnostics(s)
    h=HHS(HHSOptions(eta_ff=.02,eta_fb=.02))
    for _ in range(20):s=h.step(s,.1)
    final=diagnostics(s)
    assert h.max_depth>=1
    assert abs(final['energy']/initial['energy']-1)<1e-8
    np.testing.assert_allclose(final['momentum'],initial['momentum'],atol=1e-13)
    np.testing.assert_allclose(final['angular_momentum'],initial['angular_momentum'],atol=1e-13)


def test_hhs_mixed_slow_fast_sets_and_components():
    x=np.array([[-.025,0,0],[.025,0,0],[3,0,0],[4,0,0]])
    v=np.array([[0,-np.sqrt(1/.05)/2,0],[0,np.sqrt(1/.05)/2,0],[0,-.01,0],[0,.01,0]])
    initial=State.create(x,v,[.5,.5,.01,.01])
    for cc in (False,True):
        s=initial;h=HHS(HHSOptions(eta_ff=.05,eta_fb=.05,connected_components=cc))
        for _ in range(3):s=h.step(s,.01)
        before,after=diagnostics(initial),diagnostics(s)
        assert abs(after['energy']/before['energy']-1)<1e-5
        np.testing.assert_allclose(after['momentum'],before['momentum'],atol=1e-12)
