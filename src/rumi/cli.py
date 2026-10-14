"""Run, checkpoint and resume the CPU validation port."""

import argparse
import json
import os
from pathlib import Path
import signal
import time


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path)
    parser.add_argument('--restart',type=Path)
    parser.add_argument('--input',type=Path,help='frozen NPZ particle arrays or Gadget/LUNA HDF5 snapshot')
    parser.add_argument('--output',type=Path,default=Path('results/run'))
    parser.add_argument('--steps',type=int,help='stop after this many additional steps')
    parser.add_argument('--set',action='append',default=[],metavar='KEY=VALUE')
    parser.add_argument('--mpi',action='store_true')
    args=parser.parse_args()
    if args.restart and (args.config or args.input or args.set):
        parser.error('a restart restores its own physical configuration; use --steps to extend it')
    if args.steps is not None and args.steps<1:
        parser.error('--steps must be positive')
    comm=None
    if args.mpi:
        try:
            from mpi4py import MPI
        except ImportError as e:
            raise RuntimeError('MPI requires the optional mpi4py dependency') from e
        comm=MPI.COMM_WORLD
    import jax
    jax.config.update('jax_enable_x64',True)
    from . import State
    from .config import Config
    from .initial_conditions import TIME_UNIT_MYR
    from .problems import create
    from .simulation import Simulation
    from .snapshot import read,write
    from .stars import Stars
    from .diagnostics import cluster
    import numpy as np

    rank=comm.Get_rank() if comm else 0
    if args.restart:
        run=Simulation.load(args.restart,comm=comm)
    else:
        config=Config.read(args.config) if args.config else Config()
        for override in args.set:
            if '=' not in override:parser.error('--set expects KEY=VALUE')
            key,value=override.split('=',1)
            from .config import parse_value
            config.values[key]=parse_value(key,value)
        if args.input:
            if args.input.suffix.lower() in ('.h5','.hdf5'):
                state,stars,_=read(args.input)
            else:
                with np.load(args.input,allow_pickle=False) as f:
                    state=State.create(f['position'],f['velocity'],f['mass'],ids=f['ids'] if 'ids' in f else None)
                stars=Stars(state)
        else:
            state,stars=create(config)
        run=Simulation(state,config,stars=stars,comm=comm)
    c=run.config
    if rank==0:
        args.output.mkdir(parents=True,exist_ok=True)
        c.write(args.output/'effective_config.txt')
    if comm:comm.Barrier()
    stop=[False]
    signal.signal(signal.SIGINT,lambda *_:stop.__setitem__(0,True))
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,lambda *_:stop.__setitem__(0,True))
    start=time.monotonic();slowest=0.;initial_step=run.step_number

    snapshot_dir=args.output if c.SNAPSHOT_DIR=='.' else Path(c.SNAPSHOT_DIR)
    restart_dir=args.output if c.RESTART_DIR=='.' else Path(c.RESTART_DIR)

    def output(report,initial=False):
        if rank==0:
            with (args.output/'diagnostics.jsonl').open('a') as f:f.write(json.dumps(report,allow_nan=False)+'\n')
            if c.PRINT_INTERVAL>0 and run.step_number%c.PRINT_INTERVAL==0:
                print(json.dumps(report,allow_nan=False),flush=True)
            if c.CLUSTER_DIAG_INTERVAL>0 and run.step_number%c.CLUSTER_DIAG_INTERVAL==0 and len(run.state.mass):
                value=cluster(run.state,G=run.G,time_myr=run.time*TIME_UNIT_MYR,pair_radius=c.CLUSTER_DIAG_RPAIR_PC)
                with (args.output/Path(c.CLUSTER_DIAG_FILE).name).open('a') as f:f.write(json.dumps(value,allow_nan=False)+'\n')
        if c.SNAPSHOT_INTERVAL>0 and run.step_number%c.SNAPSHOT_INTERVAL==0 and not (initial and args.restart):
            path=snapshot_dir/f'snapdir_{run.snapshot_index:03d}'/f'{c.SNAPSHOT_BASE}_{run.snapshot_index:03d}.0.hdf5'
            if run.replicas:
                run.replicas.call(lambda:write(path,run) if rank==0 else None)
                if rank!=0:run.snapshot_index+=1
            else:write(path,run)

    try:
        output(run.report(),initial=True)
        while len(run.state.mass):
            if args.steps is not None:
                if run.step_number-initial_step>=args.steps:break
                dt=c.DT_MAX_MYR/TIME_UNIT_MYR
            else:
                remaining=c.TOTAL_TIME_MYR/TIME_UNIT_MYR-run.time
                if remaining<=0:break
                dt=min(c.DT_MAX_MYR/TIME_UNIT_MYR,remaining)
            before=time.monotonic()
            report=run.step(dt)
            slowest=max(slowest,time.monotonic()-before)
            output(report)
            if c.RESTART_INTERVAL>0 and run.step_number%c.RESTART_INTERVAL==0:
                run.save(restart_dir/f'restart_{run.step_number:08d}.npz')
                if rank==0 and c.RESTART_KEEP>0:
                    for old in sorted(restart_dir.glob('restart_*.npz'))[:-c.RESTART_KEEP]:old.unlink()
            requested=stop[0] or (c.WALLTIME_LIMIT>0 and time.monotonic()-start+1.5*slowest+60>c.WALLTIME_LIMIT)
            if comm:requested=any(comm.allgather(requested))
            if requested:break
        run.save(restart_dir/'restart.npz')
        if rank==0:
            events=run.encounters.events if run.encounters else []
            (args.output/'events.json').write_text(json.dumps(events,indent=2,allow_nan=False)+'\n')
            print(json.dumps(run.report(),allow_nan=False),flush=True)
    finally:
        run.close()
