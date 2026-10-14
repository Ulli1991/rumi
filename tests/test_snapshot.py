import numpy as np
import pytest

from rumi.initial_conditions import binary
from rumi.simulation import Simulation
from rumi.snapshot import read,write
from test_simulation import config


def test_hdf5_snapshot_roundtrip_and_units(tmp_path):
    h5py=pytest.importorskip('h5py')
    state=binary(m1=.7,m2=.3,a=.1,e=.2,G=1.)
    with Simulation(state,config(),G=1.) as run:
        run.step(.001)
        path=tmp_path/'snapshot.hdf5'
        write(path,run)
        restored,stars,header=read(path)
        np.testing.assert_array_equal(restored.position,run.state.position)
        np.testing.assert_array_equal(restored.velocity,run.state.velocity)
        np.testing.assert_array_equal(restored.ids,run.state.ids)
        assert stars.carry()==run.stars.carry()
        assert header['Time']==run.time
        with h5py.File(path,'r') as f:
            assert f['cells/U'].shape==(5,1,1,1)
            assert f['units'].attrs['UnitVelocity_in_cm_per_s']==1e5
            np.testing.assert_allclose(f['Header'].attrs['E_total'],run.report()['energy'])
