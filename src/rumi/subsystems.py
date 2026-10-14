"""Python neighbour graph and subsystem classifications with stable particle identities."""

from dataclasses import dataclass
import math

import numpy as np


def elements(r,v,mu):
    r,v=np.asarray(r),np.asarray(v)
    distance=np.linalg.norm(r)
    if not distance>0:raise ValueError('coincident particles')
    energy=float(v@v/2-mu/distance)
    h=np.cross(r,v); h2=float(h@h)
    a=-mu/(2*energy) if energy!=0 else math.inf
    e=np.sqrt(max(0.,1+2*energy*h2/mu**2))
    return dict(a=a,e=float(e),energy=energy,angular_momentum=h,
                period=2*np.pi*np.sqrt(a**3/mu) if a>0 and energy<0 else math.inf)


def mardling_stable(a_in,a_out,e_out,m1,m2,m3,inclination):
    if not (a_in>0 and a_out>0 and 0<=e_out<1):return False
    critical=2.8*((1+m3/(m1+m2))*(1+e_out)/np.sqrt(1-e_out))**.4*(1-.3*np.clip(inclination,0,np.pi)/np.pi)*a_in
    return bool(a_out*(1-e_out)>critical)


@dataclass(frozen=True)
class Group:
    ids: tuple
    kind: str
    inner: tuple | None = None
    stable: bool = False


class Detector:
    def __init__(self,*,G=1,r_ngb=1e-3,r_max=.1,mass_scaling=True,mass_reference=0,
                 wide_factor=0,mass_gamma=1e-3,flyby_cm=True):
        if not 0<r_ngb<=r_max:raise ValueError('invalid neighbour radii')
        self.G,self.r_ngb,self.r_max=G,r_ngb,r_max
        self.mass_scaling,self.mass_reference,self.wide_factor=mass_scaling,mass_reference,wide_factor
        self.mass_gamma,self.flyby_cm=mass_gamma,flyby_cm

    def detect(self,state,dt,acceleration=None):
        x,v,m,ids=(np.asarray(a) for a in state[:4])
        n=len(m);mean=self.mass_reference or float(m.mean())
        parent=list(range(n));pairs=[]
        def find(k):
            while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
            return k
        for i in range(n):
            for j in range(i+1,n):
                r=x[j]-x[i];distance=np.linalg.norm(r)
                if not distance<self.r_max:continue
                orbit=elements(r,v[j]-v[i],self.G*(m[i]+m[j]))
                radius=self.r_ngb*max(1,(max(m[i],m[j])/mean)**(1/3)) if self.mass_scaling else self.r_ngb
                bound=orbit['energy']<0
                near=(distance<radius or (bound and orbit['a']<max(radius,self.wide_factor*self.r_ngb)))
                if not near and not bound and orbit['a']*(1-orbit['e'])<radius:
                    dv=v[j]-v[i];speed2=float(dv@dv)
                    near=speed2>1e-30 and .5*distance/np.sqrt(speed2)<dt
                if near:pairs.append((i,j,orbit,bound))
        if acceleration is not None:
            acc=np.asarray(acceleration)
            filtered=[]
            for i,j,o,bound in pairs:
                distance=np.linalg.norm(x[j]-x[i])
                radius=self.r_ngb*max(1,(max(m[i],m[j])/mean)**(1/3)) if self.mass_scaling else self.r_ngb
                speed=np.linalg.norm(v[j]-v[i])
                curved=not bound and distance>=radius and speed>0 and np.linalg.norm(acc[j]-acc[i])*.5*distance/speed>speed
                if not curved:filtered.append((i,j,o,bound))
            pairs=filtered
        # Extra mass-scaled neighbours join a bound pair only if their tide is strong enough.
        bound_partner={}
        for i,j,o,bound in pairs:
            if bound:
                for a,b in ((i,j),(j,i)):
                    bound_partner[a]=b if a not in bound_partner else -1
        def passes(r,u,mu,radius):
            o=elements(r,u,mu);distance=np.linalg.norm(r);speed=np.linalg.norm(u)
            if o['energy']<0:
                return o['a']<radius or distance<radius or o['a']<self.wide_factor*self.r_ngb
            return distance<radius or (abs(o['a'])*(o['e']-1)<radius and speed>0 and .5*distance/speed<dt)
        def body(i,other):
            partner=bound_partner.get(i,-1)
            members=[i,partner] if partner>=0 and partner!=other else [i]
            mass=m[members].sum()
            return mass,np.average(x[members],weights=m[members],axis=0),np.average(v[members],weights=m[members],axis=0),len(members)==2
        base_scales={}
        for i,j,o,bound in pairs:
            if passes(x[j]-x[i],v[j]-v[i],self.G*(m[i]+m[j]),self.r_ngb):
                scale=o['a']*(1+o['e']) if bound else np.linalg.norm(x[j]-x[i])
                base_scales.setdefault(i,[]).append((j,scale))
                base_scales.setdefault(j,[]).append((i,scale))
        def scale(i,other):
            partner=bound_partner.get(i,-1)
            if partner>=0 and partner!=other:
                inner=elements(x[partner]-x[i],v[partner]-v[i],self.G*(m[i]+m[partner]))
                return inner['a']*(1+inner['e'])
            return max((s for j,s in base_scales.get(i,[]) if j!=other),default=0.)
        kept=[]
        for i,j,o,bound in pairs:
            distance=np.linalg.norm(x[j]-x[i])
            A,B=body(i,j),body(j,i)
            if self.mass_scaling and not passes(x[j]-x[i],v[j]-v[i],self.G*(m[i]+m[j]),self.r_ngb):
                outer=elements(B[1]-A[1],B[2]-A[2],self.G*(A[0]+B[0]))
                d,speed=np.linalg.norm(B[1]-A[1]),np.linalg.norm(B[2]-A[2])
                radius=self.r_ngb*max(1,(max(m[i],m[j])/mean)**(1/3))
                soon=outer['energy']<0 or d<radius or (speed>0 and .5*d/speed<dt)
                peri=max(abs(outer['a'])*abs(1-outer['e']),1e-30)
                gamma=max(2*B[0]/A[0]*(scale(i,j)/peri)**3,2*A[0]/B[0]*(scale(j,i)/peri)**3)
                if not soon or gamma<self.mass_gamma:continue
            elif not bound and self.flyby_cm and distance>=self.r_ngb and (A[3] or B[3]):
                if not passes(B[1]-A[1],B[2]-A[2],self.G*(A[0]+B[0]),self.r_ngb):continue
            kept.append((i,j,o,bound));parent[find(i)]=find(j)
        components={}
        for i,j,_,_ in kept:
            components.setdefault(find(i),set()).update((i,j))
        groups=[]
        expanded=[]
        for component in components.values():
            if len(component)<=8:
                expanded.append(component)
                continue
            roots={i:i for i in component}
            def root(i):
                while roots[i]!=i:
                    roots[i]=roots[roots[i]];i=roots[i]
                return i
            for i,j,_,bound in kept:
                if bound and i in component and j in component:roots[root(i)]=root(j)
            parts={}
            for i in component:parts.setdefault(root(i),set()).add(i)
            expanded.extend(p for p in parts.values() if 2<=len(p)<=8)
        for component in expanded:
            members=sorted(component,key=lambda i:int(ids[i]))
            kind='multiplet';inner=None;stable=False
            if len(members)==2:
                i,j=members;o=elements(x[j]-x[i],v[j]-v[i],self.G*(m[i]+m[j]))
                kind='binary' if o['energy']<0 else 'flyby'
            if len(members)==3:
                candidates=[]
                for i,j,o,bound in kept:
                    if bound and i in component and j in component:candidates.append((o['a'],i,j,o))
                kind='triple_unstable'
                if candidates:
                    _,i,j,inside=min(candidates,key=lambda a:a[0]);k=next(k for k in members if k not in (i,j))
                    com=(m[i]*x[i]+m[j]*x[j])/(m[i]+m[j]);cv=(m[i]*v[i]+m[j]*v[j])/(m[i]+m[j])
                    outside=elements(x[k]-com,v[k]-cv,self.G*sum(m[q] for q in members))
                    hi,ho=inside['angular_momentum'],outside['angular_momentum']
                    inclination=np.arccos(np.clip(hi@ho/max(np.linalg.norm(hi)*np.linalg.norm(ho),1e-300),-1,1))
                    stable=mardling_stable(inside['a'],outside['a'],outside['e'],m[i],m[j],m[k],inclination)
                    kind='triple_stable' if stable else kind;inner=(int(ids[i]),int(ids[j]))
            groups.append(Group(tuple(int(ids[i]) for i in members),kind,inner,stable))
        return groups
