import jax.numpy as jnp
import numpy as np
import pytest

from rumi import State
from rumi.config import Config
from rumi.initial_conditions import binary
from rumi.simulation import Simulation
from rumi.stars import Stars
from rumi.timesteps import perturbation_step


def config(**changes):
    values=dict(COLLISIONAL=True,PN_ORDER=0,SSE_ENABLED=False,ETA_GRAD=0.,ETA_RESTART=0.,
                R_NGB=.2,R_NGB_MAX=2.,NGB_MASS_SCALING=False,FLYBY_CM_EPS=0.,R_ESCAPE_RH=0.)
    values.update(changes)
    return Config(**values)


def test_isolated_binary_and_restart(tmp_path):
    state=binary(m1=.7,m2=.3,a=.05,e=.8,G=1.)
    with Simulation(state,config(),G=1.) as run:
        for _ in range(3):run.step(.02)
        path=tmp_path/'restart.npz'
        run.save(path)
        with Simulation.load(path) as resumed:
            for _ in range(3):
                a,b=run.step(.02),resumed.step(.02)
                np.testing.assert_array_equal(run.state.position,resumed.state.position)
                np.testing.assert_array_equal(run.state.velocity,resumed.state.velocity)
                assert a==b
        assert abs(run.report()['relative_energy_error'])<1e-13


def test_mixed_hhs_and_native_group():
    pair=binary(m1=.5,m2=.5,a=.02,e=.2,G=1.)
    state=State.create(np.vstack([pair.position,[1.,0,0]]),np.vstack([pair.velocity,[0,.1,0]]),[.5,.5,.01])
    with Simulation(state,config(R_NGB=.1),G=1.) as run:
        run.step(.005)
        report=run.report()
        assert report['groups']==1
        assert report['encounters']['kepler']>0
        assert abs(report['relative_energy_error'])<1e-7


def test_contact_merges_and_books_energy():
    state=binary(m1=.5,m2=.5,a=.05,e=0.,G=1.)
    stars=Stars(state,radius=.06/(2*2.254e-8))
    with Simulation(state,config(),G=1.,stars=stars) as run:
        run.step(.001)
        assert len(run.state.mass)==1
        np.testing.assert_allclose(run.state.mass,[1.])
        assert abs(run.report()['relative_energy_error'])<1e-12
        assert len(run.encounters.events)==1
        assert run.encounters.events[0]['kind']=='stellar_collision'


def test_step_failure_rolls_back_events_and_native_state(monkeypatch):
    state=binary(m1=.5,m2=.5,a=.05,e=.2,G=1.)
    with Simulation(state,config(),G=1.) as run:
        carry=run.carry()
        def fail(state,dt):
            run.books.energy['bad']=3.
            run.encounters.events.append(dict(kind='bad'))
            raise RuntimeError('injected failure')
        monkeypatch.setattr(run,'_global_pn',fail)
        with pytest.raises(RuntimeError,match='injected'):
            run.step(.001)
        assert run.carry()==carry
        np.testing.assert_array_equal(run.state.position,state.position)
        assert run.time==0 and run.step_number==0


def test_perturbation_bound_and_unbound_are_finite():
    x=np.array([[-.5,0,0],[.5,0,0]])
    f=np.array([[0,-.01,0],[0,.01,0]])
    for speed in (.5,2.):
        v=np.array([[0,-speed,0],[0,speed,0]])
        dt=perturbation_step(x,v,np.array([.5,.5]),f,1.,.02)
        assert np.isfinite(dt) and dt>0


def test_collision_defers_mass_loss_and_preserves_global_accounts():
    import copy
    from rumi.reference import diagnostics

    state = State.create(
        [[-.01, 0, 0], [.02, 0, 0], [2., .3, 0]],
        [[0, -.2, .1], [0, .4, 0], [.1, 0, 0]],
        [2., 1., .3],
    )
    with Simulation(state, config(COLL_MASS_LOSS_C=.2), G=1.) as run:
        manager = run.encounters
        before = diagnostics(state, G=1.)
        properties = copy.deepcopy(run.stars.data)
        staged, record = manager.stage_collision(
            state, np.array([0, 1]), 0, 1, .01, properties, (0, 1)
        )
        assert float(staged.mass[0]) == 3.
        assert float(staged.mass[1]) == 0.
        assert manager.events == []
        assert 1 in run.stars.data
        finished = manager.finish_collision(staged, record, {1: 0})
        assert float(finished.mass[0]) == pytest.approx(3. * (1. - .2 * 2. / 9.))
        after = diagnostics(finished, G=1.)
        assert after['energy'] + run.books.total_energy == pytest.approx(before['energy'])
        np.testing.assert_allclose(after['momentum'] + run.books.momentum, before['momentum'], atol=1e-15)
        np.testing.assert_allclose(
            after['angular_momentum'] + run.books.angular_momentum,
            before['angular_momentum'], atol=1e-15,
        )
        assert len(manager.events) == 1
        assert 1 not in run.stars.data
