"""Host-recursive HHS validation reference; not the production GPU scheduler."""

from dataclasses import dataclass,field
import math

import jax.numpy as jnp
import numpy as np

from .interactions import make_interactions,make_pair_steps,take,put


@dataclass
class HHSOptions:
    G: float = 1.0
    softening: float = 0.0
    dt_min: float = 1e-12
    eta_ff: float = 0.2
    eta_fb: float = 0.2
    eta_grad: float = 0.0
    eta_jerk: float = 0.0
    eta_va: float = 0.0
    connected_components: bool = False
    max_calls: int = 1000000
    cc_min: int = 0
    cc_max: int = -1
    grad_fresh: bool = False
    correct_momentum: bool = False


class HHS:
    """Correctness-first recursion; subset shape specialization is deliberately untuned."""
    def __init__(self,options=None,*,drift=None,detect=None,constraints=None,pair_state=None,graph_steps=None,snapshot=None,restore=None):
        self.options = options or HHSOptions()
        o = self.options
        if not all(math.isfinite(v) and v>0 for v in (o.G,o.dt_min,o.eta_ff,o.eta_fb)):
            raise ValueError("positive finite HHS parameters required")
        self.force,self.gradient = make_interactions(G=o.G,softening=o.softening)
        self.pair_steps = make_pair_steps(G=o.G,softening=o.softening,eta_ff=o.eta_ff,eta_fb=o.eta_fb)
        self.drift = drift
        self.detect = detect
        self.constraints = constraints
        self.pair_state = pair_state
        self.graph_steps = graph_steps
        self.snapshot = snapshot
        self.restore = restore
        self.calls = 0
        self.max_depth = 0
        self.gradient_cache = {}
        self.kick_compensation = {}

    def reset_velocity(self,ids):
        for pid in ids:self.kick_compensation.pop(int(pid),None)

    def _record_gradient(self,subset,gradient):
        norms=np.linalg.norm(np.asarray(gradient),axis=1)/24
        self.gradient_cache.update(zip(map(int,np.asarray(subset.ids)),map(float,norms)))

    def _drift(self,state,indices,dt):
        if self.drift is not None:
            return self.drift(state,indices,dt)
        return state._replace(position=state.position.at[indices].add(dt*state.velocity[indices]))

    def _kick(self,state,indices,acceleration,dt):
        ids=np.asarray(state.ids[indices])
        compensation=np.array([self.kick_compensation.get(int(pid),np.zeros(3)) for pid in ids]).reshape(-1,3)
        before=state.velocity[indices]
        increment=dt*acceleration-jnp.asarray(compensation)
        velocity=before+increment
        remainder=np.asarray((velocity-before)-increment)
        self.kick_compensation.update((int(pid),value.tolist()) for pid,value in zip(ids,remainder))
        return state._replace(velocity=state.velocity.at[indices].set(velocity))

    def _momentum(self,subset,acceleration):
        if not self.options.correct_momentum or len(subset.mass)<2:
            return acceleration
        mean=jnp.sum(subset.mass[:,None]*acceleration,axis=0)/jnp.sum(subset.mass)
        return acceleration-mean

    def fsi(self,state,indices,dt):
        indices=indices[np.asarray(state.mass[indices])>0]
        if len(indices)==0:
            return state
        t = take(state,indices)
        state = self._kick(state,indices,self._momentum(t,self.force(t,t)),dt/6)
        state = self._drift(state,indices,dt/2)
        t = take(state,indices)
        a = self._momentum(t,self.force(t,t))
        q = self.gradient(t,t,a,a)
        self._record_gradient(t,q)
        a = a+(dt*dt/24)*q
        state = self._kick(state,indices,self._momentum(t,a),2*dt/3)
        state = self._drift(state,indices,dt/2)
        t = take(state,indices)
        return self._kick(state,indices,self._momentum(t,self.force(t,t)),dt/6)

    def _cross(self,state,groups,dt,modified=False):
        groups=[g[np.asarray(state.mass[g])>0] for g in groups]
        subsets = [take(state,g) for g in groups]
        acc = [jnp.zeros_like(s.position) for s in subsets]
        for i,s in enumerate(subsets):
            for j,t in enumerate(subsets):
                if i!=j:
                    acc[i] = acc[i]+self.force(s,t)
        def correct(values):
            if not self.options.correct_momentum:
                return values
            total=sum(jnp.sum(s.mass) for s in subsets)
            mean=sum(jnp.sum(s.mass[:,None]*a,axis=0) for s,a in zip(subsets,values))/jnp.maximum(total,1e-300)
            return [a-mean for a in values]
        acc=correct(acc)
        if modified:
            corrected=[]
            for i,s in enumerate(subsets):
                q=jnp.zeros_like(s.position)
                for j,t in enumerate(subsets):
                    if i!=j:
                        q=q+self.gradient(s,t,acc[i],acc[j])
                self._record_gradient(s,q)
                corrected.append(acc[i]+(dt*dt/24)*q)
            acc=correct(corrected)
        for group,a in zip(groups,acc):
            state=self._kick(state,group,a,(2*dt/3 if modified else dt/6))
        return state

    def _steps(self,state,indices,depth=0):
        subset=take(state,indices)
        pair_subset=self.pair_state(state,indices) if self.pair_state is not None else subset
        result=np.asarray(self.pair_steps(pair_subset,pair_subset)).copy()
        o=self.options
        if o.eta_grad>0 or o.eta_jerk>0 or o.eta_va>0:
            # Fresh acceleration/gradient from the full hierarchy, not a stale previous modified kick.
            a=self.force(subset,state)
            if o.eta_grad>0:
                if o.grad_fresh and depth==0:
                    all_a=self.force(state,state)
                    q=self.gradient(subset,state,a,all_a)
                    self._record_gradient(subset,q)
                norm_a=np.linalg.norm(np.asarray(a),axis=1)
                norm_q=np.array([self.gradient_cache.get(int(pid),0.) for pid in np.asarray(subset.ids)])
                norm_v=np.linalg.norm(np.asarray(subset.velocity),axis=1)
                limit=np.where((norm_a>1e-30)&(norm_q>1e-30),o.eta_grad*np.sqrt(norm_a/np.maximum(norm_q,1e-300)),np.inf)
                velocity_limit=np.where((norm_v>1e-30)&(norm_q>1e-30),o.eta_grad*np.cbrt(norm_v/np.maximum(norm_q,1e-300)),np.inf)
                limit=np.minimum(limit,velocity_limit)
                result=np.minimum(result,limit)
            if o.eta_va>0:
                an=np.linalg.norm(np.asarray(a),axis=1)
                vn=np.linalg.norm(np.asarray(subset.velocity),axis=1)
                result=np.minimum(result,np.where((an>1e-30)&(vn>1e-30),o.eta_va*vn/np.maximum(an,1e-300),np.inf))
            if o.eta_jerk>0:
                jerk=self.gradient(subset,state,subset.velocity,state.velocity)
                an=np.linalg.norm(np.asarray(a),axis=1)
                jn=np.linalg.norm(np.asarray(jerk),axis=1)
                result=np.minimum(result,np.where((an>1e-30)&(jn>1e-30),o.eta_jerk*an/np.maximum(jn,1e-300),np.inf))
        if depth>0 and self.root_steps is not None:
            result=np.minimum(result,self.root_steps[indices])
        if self.constraints is not None:
            result=self.constraints(state,indices,result)
        if np.isnan(result).any() or np.any(result<=0):
            raise RuntimeError("invalid HHS time-step constraint")
        sid=np.asarray(subset.subsystem)
        for s in np.unique(sid[sid>=0]):
            member=sid==s
            result[member]=result[member].min()
        return result

    def _components(self,state,fast,pivot):
        # Exact pair graph, O(N_fast^2) baseline with O(N_fast) storage.
        n=len(fast);parent=list(range(n))
        def find(k):
            while parent[k]!=k:
                parent[k]=parent[parent[k]];k=parent[k]
            return k
        subset=self.pair_state(state,fast) if self.pair_state is not None else take(state,fast)
        minimum=np.asarray(self.graph_steps(state,fast,None) if self.graph_steps else self.pair_steps(subset,subset))
        for j in range(n):
            dt=np.asarray(self.graph_steps(state,fast,j) if self.graph_steps else self.pair_steps(subset,take(subset,np.array([j]))))
            edges=(dt<pivot)&((dt<=minimum)|(dt<=minimum[j]))
            for i in np.flatnonzero(edges):
                parent[find(int(i))]=find(j)
        sid=np.asarray(subset.subsystem)
        for s in np.unique(sid[sid>=0]):
            group=np.flatnonzero(sid==s)
            for i in group[1:]:parent[find(int(i))]=find(int(group[0]))
        groups={}
        for i in range(n):groups.setdefault(find(i),[]).append(fast[i])
        result=[np.asarray(g,dtype=int) for g in groups.values()]
        small=[g for g in result if len(g)<self.options.cc_min]
        if len(small)>=2:
            result=[g for g in result if len(g)>=self.options.cc_min]+[np.concatenate(small)]
        return result

    def _recursive(self,state,indices,pivot,depth):
        indices=indices[np.asarray(state.mass[indices])>0]
        if not len(indices):return state
        self.calls+=1;self.max_depth=max(self.max_depth,depth)
        if self.calls>self.options.max_calls:
            raise RuntimeError("HHS exceeded call limit before reaching target")
        if pivot/2<self.options.dt_min or len(indices)==1:
            return self.fsi(state,indices,pivot)
        if self.detect is not None:
            state=self.detect(state,indices,pivot)
        sid=np.asarray(state.subsystem[indices])
        if len(indices)!=3 and len(indices)<=8 and np.all(sid>=0) and np.all(sid==sid[0]):
            return self.fsi(state,indices,pivot)
        steps=self._steps(state,indices,depth)
        if depth==0:
            self.root_steps=np.full(len(state.mass),np.inf)
            self.root_steps[indices]=steps
        slow,fast=indices[steps>=pivot],indices[steps<pivot]
        if not len(fast):
            return self.fsi(self.fsi(state,slow,pivot/2),slow,pivot/2)
        cc=self.options.connected_components and len(fast)>1
        cc=cc and (self.options.cc_max<0 or len(indices)<=self.options.cc_max)
        groups=self._components(state,fast,pivot) if cc else [fast]
        saved=self.snapshot() if self.snapshot is not None else None
        def restore(current):
            return self.restore(current,saved) if self.restore is not None else current
        if not len(slow) and len(groups)==1:
            state=self._recursive(state,fast,pivot/2,depth+1)
            state=restore(state)
            return restore(self._recursive(state,fast,pivot/2,depth+1))
        interactions=[slow]+groups if len(slow) else groups
        state=self._cross(state,interactions,pivot)
        state=self.fsi(state,slow,pivot/2)
        for g in groups:state=self._recursive(state,g,pivot/2,depth+1)
        state=restore(state)
        state=self._cross(state,interactions,pivot,True)
        for g in groups:state=self._recursive(state,g,pivot/2,depth+1)
        state=restore(state)
        state=self.fsi(state,slow,pivot/2)
        return self._cross(state,interactions,pivot)

    def step(self,state,dt):
        if not math.isfinite(dt) or dt<=0:raise ValueError("finite positive dt required")
        self.calls=self.max_depth=0
        self.root_steps=None
        result=self._recursive(state,np.arange(len(state.mass)),dt,0)
        if not np.isfinite(np.asarray(result.position)).all() or not np.isfinite(np.asarray(result.velocity)).all():
            raise RuntimeError("HHS returned a non-finite state")
        return result
