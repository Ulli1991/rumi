"""CPU validation runtime for the complete collisional integration flow."""

from dataclasses import asdict, replace
import copy
import math

import jax.numpy as jnp
import numpy as np

from . import checkpoint
from .config import Config
from .encounters import Encounters
from .events import Books, gw_coalescence_time
from .hhs import HHS, HHSOptions
from .initial_conditions import G_CODE, TIME_UNIT_MYR
from .physics import StellarEvolution
from .reference import diagnostics
from .stars import Stars
from .subsystems import elements
from .tides import TidalField


class Simulation:
    def __init__(self,state,config=None,*,G=G_CODE,stars=None,native=None,comm=None):
        self.config=copy.deepcopy(config or Config())
        self.config.validate()
        self.G=G
        from .distributed import Replicas
        self.replicas=Replicas(comm) if comm is not None else None
        if not math.isfinite(G) or G<=0:
            raise ValueError('positive finite G required')
        c=self.config
        if c.INTEGRATOR not in ('hhs','fsi'):
            raise ValueError('INTEGRATOR must be hhs or fsi')
        self.softening=0. if c.COLLISIONAL else c.SOFTENING_PC
        self.state=state
        self.integer=None
        if c.INTEGER_POSITIONS:
            from .integer_positions import IntegerPositions
            self.integer=IntegerPositions(c.DOMAIN_HALFSIZE_PC)
            self.state=self.integer.sync(state)
            state=self.state
        self.time,self.step_number,self.snapshot_index=0.,0,0
        self.last_work=dict(hhs_calls=0,hhs_depth=0,attempts=0)
        self.books=Books()
        self.stars=stars or Stars(state,radius=c.TEST_STELLAR_RADIUS_RSUN)
        self.initial=diagnostics(state,G=G,softening=self.softening)
        self.encounters=None
        self.sse=None
        if c.COLLISIONAL:
            self.encounters=Encounters(c,self.stars,self.books,G=G,mass_reference=float(np.asarray(state.mass).mean()) if len(state.mass) else 1.,native=native)
            if c.SSE_ENABLED:
                self.sse=StellarEvolution(metallicity=c.SSE_METALLICITY,wind=c.SSE_WIND_MODEL,
                    remnant=c.SSE_REMNANT_MODEL,ppisn=c.SSE_PPISN_MODEL,ecsn=c.SSE_ECSN_MODEL,
                    max_ns_mass=c.SSE_MXNS,lbv_factor=c.SSE_FLBV,native=self.encounters.native)
        self.tides=TidalField(c.TIDAL_FIELD,G,c.GALAXY_MASS_MSUN,c.GALAXY_VC_KMS,c.GALAXY_RC_PC)
        if self.tides.kind!='none':
            self.tides.initialize_circular(c.GALAXY_R_PC)
        self._build_integrator()

    def _build_integrator(self,factor=1.):
        gradient_cache=self.integrator.gradient_cache if hasattr(self,'integrator') else {}
        compensation=self.integrator.kick_compensation if hasattr(self,'integrator') else {}
        c=self.config
        options=HHSOptions(G=self.G,softening=self.softening,dt_min=c.DT_MIN_MYR/TIME_UNIT_MYR,
            eta_ff=c.ETA_FF*factor,eta_fb=c.ETA_FB*factor,eta_grad=c.ETA_GRAD*factor,
            eta_jerk=c.ETA_JERK*factor,eta_va=c.ETA*factor,connected_components=c.HHS_CC,
            cc_min=c.HHS_CC_MIN,cc_max=max(c.SERIAL_N2_GPU,16384) if c.HHS_CC_MAX<0 else c.HHS_CC_MAX,
            grad_fresh=c.GRAD_FRESH,correct_momentum=c.USE_TREE)
        callbacks={}
        if self.encounters:
            e=self.encounters
            e.options=replace(e.options,tolerance=c.ETA_GBS*factor)
            drift=(lambda s,i,h:self.replicas.call(e.drift,s,i,h)) if self.replicas else e.drift
            callbacks=dict(drift=drift,detect=e.detect,constraints=e.constraints,
                           pair_state=e.pair_state,graph_steps=e.graph_steps,snapshot=e.snapshot,restore=e.restore)
        self.integrator=HHS(options,**callbacks)
        self.integrator.gradient_cache=gradient_cache.copy()
        self.integrator.kick_compensation=copy.deepcopy(compensation)
        if self.encounters:
            self.encounters.graph_pair_steps=self.integrator.pair_steps
            self.encounters.reset_velocity=self.integrator.reset_velocity
            if self.integer:
                self.encounters.field_drift=self.integer.drift
                self.encounters.quantize=self.integer.sync
        elif self.integer:
            self.integrator.drift=self.integer.drift
        if c.USE_TREE:
            from .tree import make_interactions
            self.integrator.force,self.integrator.gradient=make_interactions(G=self.G,softening=self.softening,
                theta=c.TREE_THETA,leaf_size=c.TREE_N_LEAF,threshold=c.TREE_THRESHOLD)
        if self.encounters and c.FLYBY_CM_KICK:
            force,gradient=self.integrator.force,self.integrator.gradient
            project=self.encounters.force_state
            self.integrator.force=lambda t,s:force(project(t),project(s))
            self.integrator.gradient=lambda t,s,ta,sa:gradient(project(t),project(s),ta,sa)
        if self.replicas:
            self.integrator.force,self.integrator.gradient=self.replicas.interactions(self.integrator.force,self.integrator.gradient)

    def _global_pn(self,state,dt):
        if not self.encounters or not self.config.PN_ORDER or dt==0:
            return state
        rows=np.flatnonzero(np.asarray(state.mass)>=self.config.M_PN)
        if len(rows)<2:
            return state
        x,v,m,sid=map(np.asarray,(state.position,state.velocity,state.mass,state.subsystem))
        dv=np.zeros_like(v)
        for q,i in enumerate(rows):
            for j in rows[q+1:]:
                if sid[i]>=0 and sid[i]==sid[j]:
                    continue
                a=self.encounters.native.pn_pair(x[[i,j]],v[[i,j]],m[[i,j]],G=self.G,
                                                 c=self.encounters.options.c,mask=1)
                dv[[i,j]]+=dt*a
        result=state._replace(velocity=state.velocity+jnp.asarray(dv))
        self.integrator.kick_compensation.clear()
        self.books.change(diagnostics(state,G=self.G),diagnostics(result,G=self.G),'pn')
        return result

    def _compact(self,state):
        keep=np.asarray(state.mass)>0
        if keep.all():
            return state
        state=type(state)(*(a[keep] for a in state))
        alive=set(map(int,np.asarray(state.ids)))
        self.stars.data={pid:star for pid,star in self.stars.data.items() if pid in alive}
        if self.encounters:
            self.encounters.tasks={key:t for key,t in self.encounters.tasks.items() if set(key)<=alive}
            state=self.encounters.flags(state)
        if self.integer:
            self.integer.words={pid:row for pid,row in self.integer.words.items() if pid in alive}
        return state

    def _end_mergers(self,state,dt):
        if not self.encounters:
            return state
        lookup={int(pid):i for i,pid in enumerate(np.asarray(state.ids))}
        for key in list(self.encounters.groups):
            if len(key)!=2:
                continue
            i,j=(lookup[pid] for pid in key)
            m=np.asarray(state.mass[np.array([i,j])])
            if np.any(m<=0):
                continue
            coll=self.encounters.collision_matrix(key,m)[0,1]
            r=np.asarray(state.position[j]-state.position[i])
            v=np.asarray(state.velocity[j]-state.velocity[i])
            orbit=elements(r,v,self.G*m.sum())
            kind=None
            if np.linalg.norm(r)>=coll:
                tau=gw_coalescence_time(orbit['a'],orbit['e'],*m,G=self.G,c=self.encounters.options.c)
                if tau>=dt:
                    continue
                kind='gw_coalescence'
            keep,gone=i,j
            state=self.encounters.merge_pair(state,keep,gone,self.time+dt,kind)
            self.integrator.reset_velocity([state.ids[keep],state.ids[gone]])
            if self.integer:state=self.integer.sync(state,np.array([keep,gone]))
        return self.encounters.flags(state)

    def _escapers(self,state):
        if not self.encounters or not len(state.mass):
            return state
        c=self.config
        x,v,m=map(np.asarray,state[:3])
        center,speed=np.average(x,weights=m,axis=0),np.average(v,weights=m,axis=0)
        radius=np.linalg.norm(x-center,axis=1)
        order=np.argsort(radius,kind='stable')
        half=radius[order[np.searchsorted(np.cumsum(m[order]),m.sum()/2)]]
        limit=c.R_ESCAPE_RH*half
        jacobi=self.tides.kind!='none' and c.R_ESCAPE_RJ>0
        if jacobi:
            for _ in range(10):
                rj=self.tides.jacobi_radius(m.sum(),position=self.tides.position+center)
                inside=radius<=rj
                if rj<=0 or not inside.any():
                    break
                new_center=np.average(x[inside],weights=m[inside],axis=0)
                speed=np.average(v[inside],weights=m[inside],axis=0)
                shift=np.linalg.norm(new_center-center)
                center=new_center
                radius=np.linalg.norm(x-center,axis=1)
                if shift<=1e-6*rj:
                    break
            if rj>0:
                limit=c.R_ESCAPE_RJ*rj
        if limit<=0:
            return state
        unbound=.5*np.sum((v-speed)**2,axis=1)-self.G*m.sum()/np.maximum(radius,1e-300)>0
        outward=np.sum((x-center)*(v-speed),axis=1)>0
        removed=(radius>=limit)&(np.asarray(state.subsystem)<0)&(jacobi|(unbound&outward))
        if not removed.any():
            return state
        before=diagnostics(state,G=self.G)
        result=state._replace(mass=jnp.where(removed,0.,state.mass))
        self.books.change(before,diagnostics(result,G=self.G),'escape')
        for i in np.flatnonzero(removed):
            self.encounters.events.append(dict(kind='escape',time=self.time,float_mass=float(m[i]),id=int(state.ids[i])))
        return self._compact(result)

    def carry(self):
        return dict(config=self.config.values.copy(),G=self.G,initial={k:np.asarray(v).tolist() for k,v in self.initial.items()},
                    books=self.books.carry(),stars=self.stars.carry(),snapshot_index=self.snapshot_index,
                    tides=dict(position=self.tides.position.tolist(),velocity=self.tides.velocity.tolist()),
                    encounters=self.encounters.carry() if self.encounters else None,
                    gradient_cache={str(pid):value for pid,value in self.integrator.gradient_cache.items()},
                    kick_compensation={str(pid):value for pid,value in self.integrator.kick_compensation.items()},
                    integer=self.integer.carry() if self.integer else None,last_work=self.last_work.copy())

    def _restore_carry(self,carry):
        self.books=Books.restore(carry['books'])
        self.stars=Stars.restore(carry['stars'])
        self.initial={k:np.asarray(v) if k!='energy' else float(v) for k,v in carry['initial'].items()}
        self.snapshot_index=carry['snapshot_index']
        self.last_work=carry.get('last_work',dict(hhs_calls=0,hhs_depth=0,attempts=0)).copy()
        self.integrator.gradient_cache={int(pid):value for pid,value in carry.get('gradient_cache',{}).items()}
        self.integrator.kick_compensation={int(pid):value.copy() for pid,value in carry.get('kick_compensation',{}).items()}
        if self.integer:self.integer.restore(carry['integer'])
        self.tides.position=np.asarray(carry['tides']['position'])
        self.tides.velocity=np.asarray(carry['tides']['velocity'])
        if self.encounters:
            self.encounters.books,self.encounters.stars=self.books,self.stars
            self.encounters.load_carry(carry['encounters'])

    def step(self,dt=None):
        if not len(self.state.mass):
            raise RuntimeError('the simulation has no particles left')
        dt=self.config.DT_MAX_MYR/TIME_UNIT_MYR if dt is None else dt
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('finite positive dt required')
        original_state,original_carry=self.state,self.carry()
        original_time,original_step=self.time,self.step_number
        c=self.config
        original_radius=self.encounters.detector.r_ngb if self.encounters else 0.
        original_pert=c.ETA_PERT
        try:
            state=self.tides.kick(self.state,dt/2,self.books)
            if self.tides.kind!='none':self.integrator.kick_compensation.clear()
            if self.sse:
                state=self.stars.evolve(state,self.sse,dt*TIME_UNIT_MYR,self.books,G=self.G,kicks=c.SSE_NATAL_KICKS,seed=c.PLUMMER_SEED)
                state=self._compact(state)
                self.integrator.kick_compensation.clear()
            before=diagnostics(state,G=self.G,softening=self.softening)['energy']+self.books.total_energy
            retry_state,retry_carry=state,self.carry()
            attempts=11 if self.encounters and c.ETA_RESTART>0 else 1
            for attempt in range(attempts):
                if attempt:
                    state=retry_state
                    self._restore_carry(retry_carry)
                factor=2.**(-attempt)
                c.values['ETA_PERT']=original_pert*factor
                if self.encounters:
                    e=self.encounters
                    e.detector.r_ngb=min(c.R_NGB_MAX,original_radius*1.15**attempt)
                    e.begin(state,self.time,abs(self.initial['energy']))
                    state=e.detect(state,np.arange(len(state.mass)),dt)
                self._build_integrator(factor)
                state=self._global_pn(state,dt/2)
                if c.INTEGRATOR=='hhs':
                    state=self.integrator.step(state,dt)
                elif len(state.mass):
                    rows=np.arange(len(state.mass))
                    constraints=self.integrator._steps(state,rows)
                    fastest=np.min(constraints)
                    substeps=max(1,2**max(0,math.ceil(math.log2(dt/max(fastest,c.DT_MIN_MYR/TIME_UNIT_MYR)))))
                    if substeps>self.integrator.options.max_calls:
                        raise RuntimeError('FSI exceeded step limit')
                    for _ in range(substeps):
                        state=self.integrator.fsi(state,rows,dt/substeps)
                state=self._global_pn(state,dt/2)
                state=self._end_mergers(state,dt)
                after=diagnostics(state,G=self.G,softening=self.softening)['energy']+self.books.total_energy
                if not math.isfinite(after) or not np.isfinite(np.asarray(state.position)).all() or not np.isfinite(np.asarray(state.velocity)).all():
                    raise RuntimeError('non-finite simulation state')
                error=abs(after-before)/max(abs(self.initial['energy']),1e-30)
                if attempts==1 or error<c.ETA_RESTART:
                    break
            else:
                raise RuntimeError(f'step failed energy tolerance after {attempts} attempts: {error:.3e}')
            state=self._compact(state)
            self.last_work=dict(hhs_calls=self.integrator.calls,hhs_depth=self.integrator.max_depth,attempts=attempt+1)
            self.tides.drift(dt)
            state=self.tides.kick(state,dt/2,self.books)
            if self.tides.kind!='none':self.integrator.kick_compensation.clear()
            self.state=self._escapers(state)
            self.time+=dt
            self.step_number+=1
            if self.replicas and c.REPLICA_CHECK_INTERVAL>0 and self.step_number%c.REPLICA_CHECK_INTERVAL==0:
                self.replicas.agree(self.state,self.carry())
            return self.report()
        except BaseException:
            self.state=original_state
            self.time,self.step_number=original_time,original_step
            self._restore_carry(original_carry)
            raise
        finally:
            c.values['ETA_PERT']=original_pert
            if self.encounters:
                self.encounters.detector.r_ngb=original_radius
            self._build_integrator()

    def report(self):
        d=diagnostics(self.state,G=self.G,softening=self.softening)
        reference=max(abs(self.initial['energy']),1e-30)
        return dict(time=self.time,time_myr=self.time*TIME_UNIT_MYR,step=self.step_number,n=len(self.state.mass),
            energy=d['energy'],energy_accounted=d['energy']+self.books.total_energy,
            relative_energy_error=(d['energy']+self.books.total_energy-self.initial['energy'])/reference,
            momentum_accounted=(d['momentum']+self.books.momentum).tolist(),
            angular_momentum_accounted=(d['angular_momentum']+self.books.angular_momentum).tolist(),
            groups=len(self.encounters.groups) if self.encounters else 0,
            encounters=self.encounters.stats.copy() if self.encounters else {},**self.last_work)

    def save(self,path):
        carry=self.carry()
        if self.replicas:
            self.replicas.agree(self.state,carry)
            self.replicas.call(lambda:checkpoint.save(path,self.state,time=self.time,step=self.step_number,metadata=carry) if self.replicas.rank==0 else None)
        else:
            checkpoint.save(path,self.state,time=self.time,step=self.step_number,metadata=carry)

    @classmethod
    def load(cls,path,*,native=None,comm=None):
        state,header=checkpoint.load(path)
        carry=header['metadata']
        result=cls(state,Config(**carry['config']),G=carry['G'],stars=Stars.restore(carry['stars']),native=native,comm=comm)
        result._restore_carry(carry)
        result.time,result.step_number=header['time'],header['step']
        return result

    def close(self):
        if self.sse:
            self.sse.close()

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
