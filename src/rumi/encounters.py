"""Adaptive subsystem ownership, native advances and collision completion."""

from dataclasses import asdict, replace
import copy

import jax.numpy as jnp
import numpy as np

from .events import collision_distance, remnant_mass
from .interactions import make_interactions, make_pair_steps, take
from .physics import Encounter, EncounterOptions, NativePhysics
from .reference import diagnostics
from .subsystems import Detector, Group, elements, mardling_stable
from .timesteps import perturbation_step, sampling_constants


class Encounters:
    def __init__(self, config, stars, books, *, G, mass_reference, native=None):
        self.config, self.stars, self.books, self.G = config, stars, books, G
        self.native = native or NativePhysics()
        self.detector = Detector(G=G, r_ngb=config.R_NGB, r_max=config.R_NGB_MAX,
            mass_scaling=config.NGB_MASS_SCALING, mass_reference=config.NGB_MREF or mass_reference,
            wide_factor=config.NGB_WIDE_FAC, mass_gamma=config.NGB_MASS_GAMMA, flyby_cm=config.FLYBY_NGB_CM)
        self.options = EncounterOptions(G=G, pn_mask=config.PN_ORDER, sdar=config.SDAR,
            k_ref=config.SDAR_KREF, kappa_cap=config.SDAR_C, tolerance=config.ETA_GBS,
            deuflhard=config.GBS_SEQUENCE=='deuflhard', max_stages=config.GBS_KMAX)
        self.force, _ = make_interactions(G=G)
        self.groups, self.tasks, self.group_ids = {}, {}, {}
        self.next_id = 0
        self.events = []
        self.time, self.energy_reference = 0., 0.
        self.clocks = None
        self.stats = {}
        self.mollified=set()
        self.flags_updated=False
        self.field_drift=None
        self.quantize=None
        self.reset_velocity=None

    def begin(self, state, time, energy_reference):
        self.time, self.energy_reference = time, energy_reference
        self.clocks = np.zeros(len(state.mass))
        self.stats = dict(kepler=0, secular_binary=0, secular_triple=0, logh=0, gbs_steps=0, collisions=0)
        self.flags_updated=False

    def flags(self, state):
        ids, mass = np.asarray(state.ids), np.asarray(state.mass)
        lookup = {int(pid): i for i,pid in enumerate(ids) if mass[i]>0}
        flags = np.full(len(ids), -1, np.int32)
        groups = {}
        for old, group in self.groups.items():
            key = tuple(pid for pid in old if pid in lookup)
            if len(key)<2:
                continue
            if key not in self.group_ids:
                self.group_ids[key] = self.next_id
                self.next_id += 1
            if key != old:
                group = replace(group, ids=key, inner=None, stable=False,
                                kind='binary' if len(key)==2 else 'multiplet')
            groups[key] = group
            for pid in key:
                if flags[lookup[pid]]>=0:
                    raise RuntimeError('overlapping subsystem ownership')
                flags[lookup[pid]] = self.group_ids[key]
        self.groups = groups
        return state._replace(subsystem=jnp.asarray(flags))

    def detect(self, state, indices, dt):
        subset = take(state, indices)
        acceleration = self.force(subset, state)
        found = self.detector.detect(subset, dt, acceleration)
        selected = set(map(int, np.asarray(subset.ids)))
        self.groups = {key:g for key,g in self.groups.items() if not selected.intersection(key)}
        self.groups.update({g.ids:g for g in found})
        state=self.flags(state)
        if not self.flags_updated and len(indices)==len(state.mass):
            self.update_mollified(state)
            self.flags_updated=True
        return state

    def update_mollified(self,state):
        self.mollified=set()
        if not self.config.FLYBY_CM_KICK:
            return
        x,v,m,ids=map(np.asarray,state[:4])
        lookup={int(pid):i for i,pid in enumerate(ids)}
        tolerance=self.config.FLYBY_CM_KICK_EPS*self.energy_reference
        for key in self.groups:
            if len(key)!=2:continue
            rows=np.array([lookup[pid] for pid in key])
            orbit=elements(x[rows[1]]-x[rows[0]],v[rows[1]]-v[rows[0]],self.G*m[rows].sum())
            if orbit['energy']>=0:continue
            center=np.average(x[rows],weights=m[rows],axis=0)
            outside=np.ones(len(m),bool);outside[rows]=False
            distance=np.linalg.norm(x[outside]-center,axis=1)
            tide=self.G*np.sum(m[outside][distance>0]/distance[distance>0]**3)
            bound=m[rows].prod()/m[rows].sum()*(orbit['a']*(1+orbit['e']))**2*tide
            if bound<=tolerance:self.mollified.add(key)

    def force_state(self,state):
        if not self.mollified:return state
        ids,m,x=map(np.asarray,(state.ids,state.mass,state.position))
        x=x.copy();lookup={int(pid):i for i,pid in enumerate(ids)}
        for key in self.mollified:
            if all(pid in lookup and m[lookup[pid]]>0 for pid in key):
                rows=np.array([lookup[pid] for pid in key])
                x[rows]=np.average(x[rows],weights=m[rows],axis=0)
        return state._replace(position=jnp.asarray(x))

    def snapshot(self):
        return self.groups.copy()

    def restore(self, state, saved):
        self.groups = saved.copy()
        return self.flags(state)

    def constraints(self, state, indices, steps):
        subset = take(state, indices)
        x,v,m,ids = map(np.asarray, subset[:4])
        lookup = {int(pid):i for i,pid in enumerate(ids)}
        acceleration = np.asarray(self.force(subset,state))
        coefficient, floor = np.full(len(m),np.inf), np.zeros(len(m))
        groups = []
        for key,group in self.groups.items():
            if not all(pid in lookup for pid in key):
                continue
            rows = np.array([lookup[pid] for pid in key])
            groups.append(rows)
            if self.config.FLYBY_CM_EPS>0:
                coefficient[rows],floor[rows] = sampling_constants(x[rows],v[rows],m[rows],self.G,
                    self.config.FLYBY_CM_EPS*self.energy_reference,self.config.TIDAL_STEP_RES)
        if self.config.FLYBY_CM_EPS>0 and groups:
            sid=np.asarray(subset.subsystem)
            for j in range(len(m)):
                valid=(np.arange(len(m))!=j)&~((sid>=0)&(sid==sid[j]))
                r=np.linalg.norm(x-x[j],axis=1)
                sampling=np.minimum(coefficient/m[j],coefficient[j]/m)*r**3
                sampling=np.maximum(sampling,np.maximum(floor,floor[j]))
                steps=np.minimum(steps,np.where(valid,sampling,np.inf))
        for rows in groups:
            limit=perturbation_step(x[rows],v[rows],m[rows],acceleration[rows],self.G,self.config.ETA_PERT)
            steps[rows]=min(steps[rows].min(),limit)
        return steps

    def pair_state(self,state,indices):
        subset=take(state,indices)
        if not self.config.FLYBY_CM_VELOCITY:
            return subset
        ids,mass,velocity=map(np.asarray,(subset.ids,subset.mass,subset.velocity))
        velocity=velocity.copy()
        lookup={int(pid):i for i,pid in enumerate(ids)}
        for key in self.groups:
            if all(pid in lookup for pid in key):
                rows=np.array([lookup[pid] for pid in key])
                velocity[rows]=np.average(velocity[rows],weights=mass[rows],axis=0)
        return subset._replace(velocity=jnp.asarray(velocity))

    def graph_steps(self,state,indices,column=None):
        physical=take(state,indices)
        subset=self.pair_state(state,indices)
        source=subset if column is None else take(subset,np.array([column]))
        result=np.asarray(self.graph_pair_steps(subset,source)).copy()
        if self.config.FLYBY_CM_EPS<=0:return result
        x,v,m,ids=map(np.asarray,physical[:4])
        coefficient,floor=np.full(len(m),np.inf),np.zeros(len(m))
        lookup={int(pid):i for i,pid in enumerate(ids)}
        for key in self.groups:
            if all(pid in lookup for pid in key):
                rows=np.array([lookup[pid] for pid in key])
                coefficient[rows],floor[rows]=sampling_constants(x[rows],v[rows],m[rows],self.G,
                    self.config.FLYBY_CM_EPS*self.energy_reference,self.config.TIDAL_STEP_RES)
        sid=np.asarray(physical.subsystem)
        columns=range(len(m)) if column is None else [column]
        for j in columns:
            valid=(np.arange(len(m))!=j)&~((sid>=0)&(sid==sid[j]))
            radius=np.linalg.norm(x-x[j],axis=1)
            step=np.maximum(np.minimum(coefficient/m[j],coefficient[j]/m)*radius**3,np.maximum(floor,floor[j]))
            result=np.minimum(result,np.where(valid,step,np.inf))
        return result

    def collision_matrix(self, ids, mass, properties=None):
        matrix=np.zeros((len(ids),len(ids)))
        for i in range(len(ids)):
            for j in range(i+1,len(ids)):
                stars=self.stars if properties is None else properties
                a,b=stars[ids[i]],stars[ids[j]]
                matrix[i,j]=matrix[j,i]=collision_distance(a.type,mass[i],a.radius*2.254e-8,
                    b.type,mass[j],b.radius*2.254e-8,G=self.G,c=self.options.c,
                    bh_radius=self.config.BH_MERGE_RADIUS)[0]
        return matrix

    def _secular(self, x, v, m, ids, group, dt, mask, coll):
        if len(m)==2:
            orbit=elements(x[1]-x[0],v[1]-v[0],self.G*m.sum())
            tau,peri=self.native.pericentre(x[1]-x[0],v[1]-v[0],self.G*m.sum())
            candidate=peri<coll[0,1] and (tau<=dt*(1+1e-5) or np.linalg.norm(x[1]-x[0])<coll[0,1])
            if candidate or (mask and dt<self.config.N_ORB_SECULAR*orbit['period']):
                return None
            self.stats['secular_binary' if mask else 'kepler']+=1
            ox,ov=self.native.secular_binary(x,v,m,dt,G=self.G,c=self.options.c,mask=mask)
            return ox,ov,None
        if len(m)!=3 or not group.inner or not self.config.SECULAR_TRIPLES:
            return None
        rows=[ids.index(pid) for pid in group.inner]+[next(i for i,pid in enumerate(ids) if pid not in group.inner)]
        i,j,k=rows
        inside=elements(x[j]-x[i],v[j]-v[i],self.G*(m[i]+m[j]))
        center=np.average(x[[i,j]],weights=m[[i,j]],axis=0)
        speed=np.average(v[[i,j]],weights=m[[i,j]],axis=0)
        outside=elements(x[k]-center,v[k]-speed,self.G*m.sum())
        hi,ho=inside['angular_momentum'],outside['angular_momentum']
        inclination=np.arccos(np.clip(hi@ho/max(np.linalg.norm(hi)*np.linalg.norm(ho),1e-300),-1,1))
        stable=mardling_stable(inside['a'],outside['a'],outside['e'],m[i],m[j],m[k],inclination)
        tau,peri=self.native.pericentre(x[j]-x[i],v[j]-v[i],self.G*(m[i]+m[j]))
        if not stable or dt<self.config.N_ORB_SECULAR*inside['period'] or (peri<coll[i,j] and tau<=dt*(1+1e-5)):
            return None
        output=self.native.secular_triple(x[rows],v[rows],m[rows],dt,G=self.G,c=self.options.c,mask=mask)
        if output is None:
            return None
        ox,ov,radiation=output
        result_x,result_v=x.copy(),v.copy()
        result_x[rows],result_v[rows]=ox,ov
        self.stats['secular_triple']+=1
        return result_x,result_v,radiation

    def drift(self, state, indices, dt):
        lookup={int(pid):i for i,pid in enumerate(np.asarray(state.ids))}
        selected=set(map(int,indices))
        eligible=[]
        owned=set()
        for group in self.groups.values():
            rows=np.array([lookup[pid] for pid in group.ids])
            intersection=selected.intersection(rows)
            if intersection and len(intersection)!=len(rows):
                raise RuntimeError('HHS split subsystem members')
            if intersection:
                eligible.append((group,rows));owned.update(rows)
        field=np.array(sorted(selected-owned),dtype=int)
        state=self.field_drift(state,field,dt) if self.field_drift else state._replace(position=state.position.at[field].add(dt*state.velocity[field]))
        self.clocks[field]+=dt
        for group,rows in eligible:
            if np.ptp(self.clocks[rows])>1e-12*max(dt,1.):
                raise RuntimeError('subsystem members have different times')
            state=self._advance_group(state,group,rows,dt)
            if self.quantize:state=self.quantize(state,rows)
            if self.reset_velocity:self.reset_velocity(group.ids)
            self.clocks[rows]+=dt
        return state

    def _advance_group(self,state,group,rows,dt):
        x,v,m=[np.asarray(a[rows]).copy() for a in state[:3]]
        ids=list(group.ids)
        mask=self.options.pn_mask if m.max()>=self.config.M_PN else 0
        coll=self.collision_matrix(ids,m)
        secular=self._secular(x,v,m,ids,group,dt,mask,coll)
        if secular is not None:
            ox,ov,radiation=secular
            if mask:
                before=diagnostics(take(state,rows),G=self.G)
                after=diagnostics(take(state,rows)._replace(position=ox,velocity=ov),G=self.G)
                self.books.change(before,after,'gw' if radiation is not None or mask&20 else 'pn')
            self.tasks.pop(group.ids,None)
            return state._replace(position=state.position.at[rows].set(ox),velocity=state.velocity.at[rows].set(ov))
        remaining,elapsed=dt,0.
        local_rows=rows.copy()
        properties={pid:copy.deepcopy(self.stars[pid]) for pid in ids}
        pending=[]
        descendants={}
        while remaining>0:
            if len(ids)==1:
                x+=remaining*v
                elapsed+=remaining
                break
            key=tuple(ids)
            task=self.tasks.get(key)
            options=replace(self.options,pn_mask=mask)
            reset=task is None or not np.array_equal(task.mass,m)
            if task is not None:
                reset=reset or (task.options.pn_mask,task.options.G,task.options.c)!=(options.pn_mask,options.G,options.c)
            if reset:
                spins=np.array([self.stars[pid].spin for pid in ids]) if len(ids)==2 else None
                task=Encounter(x,v,m,options=options,spins=spins,native=self.native)
                self.tasks[key]=task
            else:
                task.options=options
            before=diagnostics(take(state,local_rows)._replace(position=x,velocity=v,mass=m),G=self.G)
            result=self.advance_with_retry(task,x,v,remaining,self.collision_matrix(ids,m,properties))
            self.stats['logh']+=1
            self.stats['gbs_steps']+=result.steps
            after=diagnostics(take(state,local_rows)._replace(position=result.position,velocity=result.velocity,mass=m),G=self.G)
            if mask:
                self.books.change(before,after,'gw' if mask&20 else 'pn')
            elif result.slowdown_energy_jump:
                self.books.energy['sdar']=self.books.energy.get('sdar',0.)-result.slowdown_energy_jump
            x,v=result.position,result.velocity
            elapsed+=result.time_reached
            remaining=max(0.,dt-elapsed)
            if len(ids)==2 and task.options.pn_mask&32:
                for pid,spin in zip(ids,task.spins):self.stars[pid].spin=spin.tolist()
            if result.collision is None:
                break
            state=state._replace(position=state.position.at[local_rows].set(x),velocity=state.velocity.at[local_rows].set(v))
            i,j,_=result.collision
            compact_i,compact_j=properties[ids[i]].type>=10,properties[ids[j]].type>=10
            i_keeps=compact_i if compact_i!=compact_j else m[i]>=m[j]
            keep,gone=(i,j) if i_keeps else (j,i)
            event_time=self.time+self.clocks[rows[0]]+elapsed
            old_aux=task.aux.copy()
            old_aux[keep]=(m[keep]*old_aux[keep]+m[gone]*old_aux[gone])/(m[keep]+m[gone])
            state,record=self.stage_collision(state,local_rows,keep,gone,event_time,properties,tuple(group.ids))
            pending.append(record)
            descendants[record['gone']]=record['keep']
            self.tasks.pop(key,None)
            live=np.asarray(state.mass[local_rows])>0
            local_rows=local_rows[live]
            ids=[pid for pid,alive in zip(ids,live) if alive]
            x,v,m=[np.asarray(a[local_rows]).copy() for a in state[:3]]
            if len(ids)>1:
                mask &= ~32
                next_task=Encounter(x,v,m,options=replace(self.options,pn_mask=mask),native=self.native)
                next_task.aux=old_aux[live].copy()
                self.tasks[tuple(ids)]=next_task
            self.stats['collisions']+=1
        state=state._replace(position=state.position.at[local_rows].set(x),velocity=state.velocity.at[local_rows].set(v))
        for record in pending:
            state=self.finish_collision(state,record,descendants)
        return self.flags(state)

    def stage_collision(self,state,rows,keep,gone,time,properties,members):
        subset=take(state,rows)
        x,v,m=[np.asarray(a).copy() for a in subset[:3]]
        ids=list(map(int,np.asarray(subset.ids)))
        a,b=properties[ids[keep]],properties[ids[gone]]
        radius,kind=collision_distance(a.type,m[keep],a.radius*2.254e-8,b.type,m[gone],b.radius*2.254e-8,
            G=self.G,c=self.options.c,bh_radius=self.config.BH_MERGE_RADIUS)
        mass=m[[keep,gone]].copy()
        center=np.average(x[[keep,gone]],weights=mass,axis=0)
        speed=np.average(v[[keep,gone]],weights=mass,axis=0)
        record=dict(keep=ids[keep],gone=ids[gone],mass=mass,types=(a.type,b.type),kind=kind,time=float(time),
            offsets=x[[keep,gone]]-center,velocities=v[[keep,gone]]-speed,members=members,
            separation=float(np.linalg.norm(x[gone]-x[keep])),collision_distance=float(radius))
        before=diagnostics(subset,G=self.G)['energy']
        x[keep],v[keep],m[keep]=center,speed,mass.sum()
        x[gone],v[gone],m[gone]=center,speed,0.
        after=subset._replace(position=x,velocity=v,mass=m)
        record['internal_change']=diagnostics(after,G=self.G)['energy']-before
        a.type=max(a.type,b.type) if max(a.type,b.type)>=10 else a.type
        a.radius=0. if a.type==14 else max(a.radius,b.radius)
        return state._replace(position=state.position.at[rows].set(x),velocity=state.velocity.at[rows].set(v),
                              mass=state.mass.at[rows].set(m)),record

    def finish_collision(self,state,record,descendants):
        keep=record['keep']
        while keep in descendants:keep=descendants[keep]
        ids=np.asarray(state.ids)
        row=int(np.flatnonzero(ids==keep)[0])
        x,v,m=map(np.asarray,state[:3])
        center,speed=x[row],v[row]
        mass=record['mass'];total=mass.sum()
        positions=center+record['offsets']
        velocities=speed+record['velocities']
        external=~np.isin(ids,record['members'])&(m>0)
        delta=0.
        for point,value in zip(positions,mass):
            delta-=self.G*value*np.sum(m[external]/np.linalg.norm(x[external]-point,axis=1))
        delta+=self.G*total*np.sum(m[external]/np.linalg.norm(x[external]-center,axis=1))
        self.books.energy['merge']=self.books.energy.get('merge',0.)-record['internal_change']+delta
        self.books.momentum+=np.sum(mass[:,None]*velocities,axis=0)-total*speed
        self.books.angular_momentum+=np.sum(mass[:,None]*np.cross(positions,velocities),axis=0)-total*np.cross(center,speed)
        remnant=remnant_mass(record['kind'],record['types'],mass,tde_fraction=self.config.TDE_ACCRETION_FRACTION,
                             loss_coefficient=self.config.COLL_MASS_LOSS_C)
        spin,kick=np.zeros(3),np.zeros(3)
        if record['types']==(14,14):
            angular=np.sum(mass[:,None]*np.cross(record['offsets'],record['velocities']),axis=0)
            spins=[self.stars[record['keep']].spin,self.stars[record['gone']].spin]
            remnant,spin,kick=self.native.bh_remnant(mass,spins,angular,G=self.G,c=self.options.c)
        before=diagnostics(state,G=self.G)
        new_mass=m[row]+remnant-total
        if not new_mass>0:raise RuntimeError('nonpositive collision remnant mass')
        impulse=kick if m[row]==total else remnant/new_mass*kick
        state=state._replace(mass=state.mass.at[row].set(new_mass),velocity=state.velocity.at[row].add(jnp.asarray(impulse)))
        self.books.change(before,diagnostics(state,G=self.G),'merge')
        self.stars.merged(record['keep'],record['gone'],new_mass if keep==record['keep'] else remnant,record['kind'],spin)
        if new_mass!=m[row]:
            self.tasks={key:task for key,task in self.tasks.items() if keep not in key}
        self.events.append(dict(kind=record['kind'],time=record['time'],keep=record['keep'],gone=record['gone'],
            progenitor_mass=mass.tolist(),mass=float(remnant),kick=kick.tolist(),spin=spin.tolist(),
            descendant=keep,separation=record['separation'],collision_distance=record['collision_distance']))
        return state

    def advance_with_retry(self,task,x,v,dt,collisions,depth=0):
        saved=task.carry()
        try:
            return task.advance(x,v,dt,collision_distances=collisions)
        except RuntimeError:
            task.__dict__.update(Encounter.restore(saved,options=task.options,native=self.native).__dict__)
            if depth>=8:
                raise
        try:
            first=self.advance_with_retry(task,x,v,dt/2,collisions,depth+1)
            if first.collision is not None:return first
            second=self.advance_with_retry(task,first.position,first.velocity,dt/2,collisions,depth+1)
            return replace(second,time_reached=first.time_reached+second.time_reached,
                steps=first.steps+second.steps,slowdown_energy_jump=first.slowdown_energy_jump+second.slowdown_energy_jump,
                kappa_peak=max(first.kappa_peak,second.kappa_peak))
        except BaseException:
            task.__dict__.update(Encounter.restore(saved,options=task.options,native=self.native).__dict__)
            raise

    def merge_pair(self,state,keep,gone,time,kind=None):
        x,v,m=[np.asarray(a).copy() for a in state[:3]]
        pids=np.asarray(state.ids)
        a,b=self.stars[pids[keep]],self.stars[pids[gone]]
        if kind is None:
            _,kind=collision_distance(a.type,m[keep],a.radius*2.254e-8,b.type,m[gone],b.radius*2.254e-8,
                                      G=self.G,c=self.options.c,bh_radius=self.config.BH_MERGE_RADIUS)
        before=diagnostics(state,G=self.G)
        progenitor_mass=m[[keep,gone]].copy()
        angular=m[keep]*m[gone]/(m[keep]+m[gone])*np.cross(x[gone]-x[keep],v[gone]-v[keep])
        spin,kick=np.zeros(3),np.zeros(3)
        mass=remnant_mass(kind,(a.type,b.type),progenitor_mass,
                         tde_fraction=self.config.TDE_ACCRETION_FRACTION,loss_coefficient=self.config.COLL_MASS_LOSS_C)
        if a.type==14 and b.type==14:
            mass,spin,kick=self.native.bh_remnant(progenitor_mass,[a.spin,b.spin],angular,G=self.G,c=self.options.c)
        total=progenitor_mass.sum()
        x[keep]=(m[keep]*x[keep]+m[gone]*x[gone])/total
        v[keep]=(m[keep]*v[keep]+m[gone]*v[gone])/total+kick
        m[keep],m[gone]=mass,0.
        x[gone],v[gone]=x[keep],v[keep]
        result=state._replace(position=jnp.asarray(x),velocity=jnp.asarray(v),mass=jnp.asarray(m))
        self.books.change(before,diagnostics(result,G=self.G),'merge')
        self.stars.merged(pids[keep],pids[gone],mass,kind,spin)
        self.events.append(dict(kind=kind,time=float(time),keep=int(pids[keep]),gone=int(pids[gone]),
                               progenitor_mass=progenitor_mass.tolist(),mass=float(mass),kick=kick.tolist(),spin=spin.tolist()))
        return result

    def carry(self):
        return dict(groups=[asdict(g) for g in self.groups.values()],
                    tasks=[dict(ids=list(key),carry=task.carry(),options=asdict(task.options)) for key,task in self.tasks.items()],
                    group_ids=[dict(ids=list(key),id=value) for key,value in self.group_ids.items()],next_id=self.next_id,
                    events=self.events.copy(),mass_reference=self.detector.mass_reference,mollified=[list(k) for k in sorted(self.mollified)])

    def load_carry(self,carry):
        self.groups={tuple(g['ids']):Group(tuple(g['ids']),g['kind'],tuple(g['inner']) if g['inner'] else None,g['stable']) for g in carry['groups']}
        self.tasks={tuple(t['ids']):Encounter.restore(t['carry'],options=EncounterOptions(**t['options']),native=self.native) for t in carry['tasks']}
        self.group_ids={tuple(g['ids']):g['id'] for g in carry['group_ids']}
        self.next_id,self.events=carry['next_id'],carry['events'].copy()
        self.detector.mass_reference=carry['mass_reference']
        self.mollified={tuple(k) for k in carry.get('mollified',[])}
