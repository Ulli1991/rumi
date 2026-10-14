"""Compare two-rank stepping with the same serial run."""

import jax
import numpy as np
from mpi4py import MPI

jax.config.update('jax_enable_x64',True)

from rumi import State
from rumi.config import Config
from rumi.simulation import Simulation

comm=MPI.COMM_WORLD
state=State.create([[-.5,0,0],[.5,0,0],[4,0,0]],[[0,-.5,0],[0,.5,0],[0,.1,0]],[.5,.5,.01])
config=Config(COLLISIONAL=False,SOFTENING_PC=0.,ETA_GRAD=0.,HHS_CC=True)
with Simulation(state,config,G=1.,comm=comm) as distributed:
    distributed.step(.001)
    distributed.replicas.agree(distributed.state,distributed.carry())
    if comm.Get_rank()==0:
        with Simulation(state,config,G=1.) as serial:
            serial.step(.001)
            np.testing.assert_allclose(distributed.state.position,serial.state.position,rtol=1e-13,atol=1e-14)
            np.testing.assert_allclose(distributed.state.velocity,serial.state.velocity,rtol=1e-13,atol=1e-14)
        print('MPI/serial complete-step agreement passed')
