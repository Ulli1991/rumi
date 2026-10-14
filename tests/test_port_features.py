from dataclasses import replace
import json

import jax.numpy as jnp
import numpy as np
import pytest

from rumi import State
from rumi.config import Config
from rumi.diagnostics import cluster
from rumi.distributed import Replicas
from rumi.initial_conditions import binary,G_CODE
from rumi.integer_positions import IntegerPositions
from rumi.interactions import make_interactions
from rumi.physics import NativePhysics,StellarEvolution
from rumi.problems import create
from rumi.simulation import Simulation
from rumi.stars import Stars
from rumi.subsystems import Detector
from rumi.tree import make_interactions as tree_interactions
from test_simulation import config


def test_integer_drift_reversibility_and_restart(tmp_path):
    state=State.create([[.123,0,0]],[[.013,.07,0]],[1.])
    grid=IntegerPositions(2.)
    start=grid.sync(state)
    advanced=grid.drift(start,np.array([0]),.031)
    returned=grid.drift(advanced,np.array([0]),-.031)
    np.testing.assert_array_equal(returned.position,start.position)
    pair=binary(m1=.5,m2=.5,a=.1,e=.2,G=1.)
    with Simulation(pair,config(INTEGER_POSITIONS=True),G=1.) as run:
        run.step(.001);path=tmp_path/'integer.npz';run.save(path)
        with Simulation.load(path) as resumed:
            run.step(.001);resumed.step(.001)
            np.testing.assert_array_equal(run.state.position,resumed.state.position)
            assert run.integer.carry()==resumed.integer.carry()


def test_tree_exact_limit_and_subsystem_exclusion():
    rng=np.random.default_rng(45)
    state=State.create(rng.normal(size=(31,3)),rng.normal(size=(31,3)),rng.uniform(.1,1,31),
                       subsystem=np.r_[0,0,np.full(29,-1)])
    direct,gradient=make_interactions(softening=.01)
    tree,tg=tree_interactions(softening=.01,theta=0.,leaf_size=3)
    a=direct(state,state)
    np.testing.assert_allclose(tree(state,state),a,rtol=2e-13,atol=2e-13)
    np.testing.assert_allclose(tg(state,state,a,a),gradient(state,state,a,a),rtol=5e-13,atol=5e-13)


def test_sse_winds_and_kicks_preserve_accounts(tmp_path):
    state=State.create([[0,0,0]],[[.01,.02,0]],[30.])
    stars=Stars(state)
    with Simulation(state,config(SSE_ENABLED=True,R_NGB=.01,R_NGB_MAX=.1),G=G_CODE,stars=stars) as run:
        run.step(100/.977792)
        assert run.stars[0].type in (13,14)
        assert float(run.state.mass[0])<30
        np.testing.assert_allclose(run.report()['momentum_accounted'],[.3,.6,0],atol=1e-12)
        before=np.asarray(run.state.velocity).copy()
        run.step(1.)
        np.testing.assert_array_equal(run.state.velocity,before)


def test_native_logh_and_sdar_are_used_for_triples(tmp_path):
    pair=binary(m1=.5,m2=.5,a=.01,e=.2,G=1.)
    state=State.create(np.vstack([pair.position,[2.,0,0]]),np.vstack([pair.velocity,[0,.1,0]]),[.5,.5,.1])
    with Simulation(state,config(R_NGB=3.,R_NGB_MAX=4.,SECULAR_TRIPLES=False,SDAR=True),G=1.) as run:
        run.step(.001)
        assert run.report()['encounters']['logh']>0
        assert run.report()['encounters']['gbs_steps']>0
        assert abs(run.report()['relative_energy_error'])<1e-7
        path=tmp_path/'logh.npz';run.save(path)
        with Simulation.load(path) as resumed:
            run.step(.001);resumed.step(.001)
            np.testing.assert_array_equal(run.state.position,resumed.state.position)
            np.testing.assert_array_equal(run.state.velocity,resumed.state.velocity)


def test_secular_triple_evolves_all_three_members():
    c=config(TEST_CASE='triple_test',TRIPLE_M1=.5,TRIPLE_M2=.5,TRIPLE_M3=.1,
             TRIPLE_A_IN_PC=.01,TRIPLE_A_OUT_PC=.5,TRIPLE_E_IN=.2,TRIPLE_E_OUT=.1,
             R_NGB=1.,R_NGB_MAX=2.,TRIPLE_OUTER_FIRST=True)
    state,stars=create(c)
    with Simulation(state,c,G=G_CODE,stars=stars) as run:
        run.step(3.)
        assert run.report()['encounters']['secular_triple']>0
        assert np.all(np.linalg.norm(np.asarray(run.state.position-state.position),axis=1)>0)
        np.testing.assert_allclose(np.average(run.state.position,weights=run.state.mass,axis=0),[0,0,0],atol=1e-14)


def test_escaper_removal_preserves_energy_and_identity():
    state=State.create([[-1,0,0],[0,0,0],[1,0,0],[100,0,0]],[[0,.1,0],[0,0,0],[0,-.1,0],[2,0,0]],
                       [1.,1.,1.,.001],ids=[10,20,30,99])
    with Simulation(state,config(R_NGB=.001,R_NGB_MAX=.1,R_ESCAPE_RH=2.),G=1.) as run:
        run.step(.0001)
        assert 99 not in np.asarray(run.state.ids)
        assert run.encounters.events[-1]['kind']=='escape'
        assert abs(run.report()['relative_energy_error'])<1e-10


def test_cluster_binary_census_uses_particles():
    pair=binary(m1=.5,m2=.5,a=.01,e=.2,G=1.)
    value=cluster(pair,G=1.,pair_radius=.1)
    assert value['binaries']==1 and value['binary_energy']<0
    np.testing.assert_allclose(value['binary_energy'],-12.5,rtol=1e-14)


def test_problem_factories_and_primordial_binaries():
    for problem in ('outer_planets','binary_test','binary_test_peri','triple_test'):
        state,stars=create(config(TEST_CASE=problem,BINARY_A_PC=1e-5,TRIPLE_A_IN_PC=1e-5,TRIPLE_A_OUT_PC=1e-3))
        assert len(stars.data)==len(state.mass)
        assert np.isfinite(np.asarray(state.position)).all()
    state,stars=create(config(TEST_CASE='plummer',PLUMMER_N=20,PLUMMER_R_HALF='1.',PLUMMER_M_TOTAL=20.,
                             BINARY_FRAC=.4,BINARY_A_MIN_AU=.01,BINARY_A_MAX_AU=10.,BINARY_PERI_CHECK=True))
    groups=Detector(G=G_CODE,r_ngb=.001,r_max=.01,mass_scaling=False).detect(state,1.)
    assert sum(len(g.ids)==2 for g in groups)==4
    np.testing.assert_allclose(np.sum(np.asarray(state.mass)),20.)
    np.testing.assert_allclose(np.average(state.velocity,weights=state.mass,axis=0),[0,0,0],atol=1e-13)


def test_sink_distributed_force_api():
    state=State.create([[i,0,0] for i in range(7)],np.zeros((7,3)),np.ones(7))
    force,gradient=make_interactions()
    expected=np.asarray(force(state,state))
    class Comm:
        def __init__(self,rank):self.rank=rank
        def Get_rank(self):return self.rank
        def Get_size(self):return 2
        def allgather(self,value):
            if value is None:return [None,None]
            lo,hi=7*self.rank//2,7*(self.rank+1)//2
            np.testing.assert_allclose(value,expected[lo:hi],rtol=1e-14)
            return [expected[:3],expected[3:]]
    for rank in range(2):
        distributed,_=Replicas(Comm(rank)).interactions(force,gradient)
        np.testing.assert_array_equal(distributed(state,state),expected)


def test_bh_remnant_is_deterministic_and_respects_kerr_limit():
    native=NativePhysics()
    a=native.bh_remnant([20.,10.],[[0,.3,.5],[0,0,0]],[0,0,1])
    b=native.bh_remnant([20.,10.],[[0,.3,.5],[0,0,0]],[0,0,1])
    assert 29.<a[0]<30 and np.linalg.norm(a[1])<.998
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)
