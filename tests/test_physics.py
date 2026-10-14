import numpy as np
import pytest

from rumi.physics import NativePhysics,Encounter,EncounterOptions,StellarEvolution
from rumi.reference import diagnostics
from rumi import State
from test_cpu import cpu_library


def orbit(e=0.6):
    r=1+e;v=np.sqrt((1-e)/(1+e))
    return np.array([[-r/2,0,0],[r/2,0,0]]),np.array([[0,-v/2,0],[0,v/2,0]]),np.array([.5,.5])


def test_retained_kepler_reference_and_composition(cpu_library):
    native=NativePhysics(cpu_library)
    r,v=native.kepler([1,0,0],[0,1,0],1,2*np.pi)
    np.testing.assert_allclose(r,[1,0,0],atol=3e-14)
    r1,v1=native.kepler([1,0,0],[0,1,0],1,.3)
    r2,v2=native.kepler(r1,v1,1,.7)
    expected=native.kepler([1,0,0],[0,1,0],1,1)
    np.testing.assert_allclose(r2,expected[0],atol=3e-14)


def test_retained_logh_persistent_continuation(cpu_library):
    x,v,m=orbit()
    enc=Encounter(x,v,m,native=NativePhysics(cpu_library),options=EncounterOptions(tolerance=1e-12))
    initial=State.create(x,v,m)
    for _ in range(4):
        result=enc.advance(x,v,np.pi/2)
        assert abs(result.time_reached-np.pi/2)<1e-11
        x,v=result.position,result.velocity
    np.testing.assert_allclose(x,initial.position,atol=2e-7)
    assert abs(diagnostics(State.create(x,v,m))['energy']/diagnostics(initial)['energy']-1)<1e-9


def test_pn_pair_swap_and_disabled_limit(cpu_library):
    n=NativePhysics(cpu_library);x,v,m=orbit()
    np.testing.assert_array_equal(n.pn_pair(x,v,m,mask=0),0)
    a=n.pn_pair(x,v,m,c=100,mask=31)
    assert np.linalg.norm(a)>0
    np.testing.assert_allclose(n.pn_pair(x[::-1],v[::-1],m[::-1],c=100,mask=31),a[::-1],atol=1e-16)


def test_sdar_hierarchical_triple(cpu_library):
    a=.01;speed=np.sqrt(1/a)/2
    x=np.array([[-a/2,0,0],[a/2,0,0],[2,0,0]])
    v=np.array([[0,-speed,0],[0,speed,0],[0,.2,0]])
    e=Encounter(x,v,[.5,.5,.1],native=NativePhysics(cpu_library),options=EncounterOptions(sdar=True))
    r=e.advance(x,v,.01)
    assert r.kappa_peak>1
    assert np.isfinite(r.position).all() and abs(r.time_reached-.01)<1e-11


def test_sse_models_and_recoil(cpu_library):
    n=NativePhysics(cpu_library)
    with StellarEvolution(native=n) as s:
        star=s.evolve(1,0)
        assert star['mass']==1 and .7<star['radius']<1.1 and .5<star['luminosity']<1.1
    with StellarEvolution(native=n,remnant='FRYER_RAPID') as s:
        rem=s.evolve(30,100)
        assert rem['type'] in (13,14) and 0<rem['mass']<30
    np.testing.assert_array_equal(n.recoil([10,10],np.zeros((2,3)),[0,0,1]),0)
    assert 156.6<np.linalg.norm(n.recoil([20,10],np.zeros((2,3)),[0,0,1]))<156.8


def test_retained_collision_and_failure_are_explicit(cpu_library):
    x,v,m=orbit(0)
    e=Encounter(x,v,m,native=NativePhysics(cpu_library))
    coll=np.ones((2,2))*2;np.fill_diagonal(coll,0)
    r=e.advance(x,v,.1,collision_distances=coll)
    assert r.collision is not None and r.time_reached<.1
    e=Encounter(x,v,m,native=NativePhysics(cpu_library),options=EncounterOptions(max_steps=1))
    with pytest.raises(RuntimeError,match="target"):
        e.advance(x,v,10)
