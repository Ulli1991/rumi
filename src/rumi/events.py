"""Physical event policies and exact changes of global Newtonian E/P/L accounts."""

from dataclasses import dataclass,field
import numpy as np

from .reference import diagnostics


@dataclass
class Books:
    energy: dict = field(default_factory=dict)
    momentum: np.ndarray = field(default_factory=lambda:np.zeros(3))
    angular_momentum: np.ndarray = field(default_factory=lambda:np.zeros(3))

    def change(self,before,after,kind):
        self.energy[kind]=self.energy.get(kind,0.)+before['energy']-after['energy']
        self.momentum+=before['momentum']-after['momentum']
        self.angular_momentum+=before['angular_momentum']-after['angular_momentum']

    @property
    def total_energy(self):return sum(self.energy.values())

    def carry(self):
        return dict(energy=self.energy.copy(), momentum=self.momentum.tolist(),
                    angular_momentum=self.angular_momentum.tolist())

    @classmethod
    def restore(cls, carry):
        return cls(carry['energy'].copy(),np.asarray(carry['momentum'],float),
                   np.asarray(carry['angular_momentum'],float))


def gw_coalescence_time(a,e,m1,m2,*,G,c):
    if a<=0 or not 0<=e<1 or c<=0:
        return np.inf
    beta=64/5*G**3*m1*m2*(m1+m2)/c**5
    circular=a**4/(4*beta)
    return circular if e<.01 else circular*(1-e*e)**3.5/(1+73/24*e*e+37/96*e**4)


def remnant_mass(kind,types,mass,*,tde_fraction=.5,loss_coefficient=0.):
    m1,m2=mass
    if kind=='tde':
        return m1+tde_fraction*m2 if types[0]==14 else m2+tde_fraction*m1
    q=min(mass)/max(mass)
    return sum(mass)*(1-loss_coefficient*q/(1+q)**2) if kind=='stellar_collision' else sum(mass)


def collision_distance(kind1,m1,r1,kind2,m2,r2,*,G=1,c=299792.458,bh_radius='total'):
    bh1,bh2=kind1==14,kind2==14
    isco=6*G*(max(m1,m2) if bh1 and bh2 and bh_radius=='max_single' else m1+m2)/c**2 if c>0 else 0
    if bh1 and bh2:return isco,'isco'
    if bh1 or bh2:
        mb,ms,rs=(m1,m2,r2) if bh1 else (m2,m1,r1)
        tidal=max(1.3*(mb/ms)**(1/3)*rs,rs) if ms>0 else 0
        return (isco,'isco') if isco>=tidal else (tidal,'tde')
    return r1+r2,'stellar_collision'


def natal_kick(pid,seed,new_type,fallback,ecsn=False):
    # Exact integer hash and Box-Muller choices used in cpp-nbody runtime_sse.cpp.
    mask=(1<<64)-1;h=1469598103934665603
    for value in (pid,seed,new_type,0x4b49434b):h=((h^(int(value)&mask))*1099511628211)&mask
    u=[]
    for _ in range(4):
        h=(h+0x9e3779b97f4a7c15)&mask;z=h
        z=((z^(z>>30))*0xbf58476d1ce4e5b9)&mask
        z=((z^(z>>27))*0x94d049bb133111eb)&mask
        z^=z>>31;u.append(((z>>11)+.5)*2**-53)
    scale=3 if ecsn else 265*(1-np.clip(fallback,0,1))
    r1,r2=np.sqrt(-2*np.log(u[0])),np.sqrt(-2*np.log(u[2]))
    return scale*np.array([r1*np.cos(2*np.pi*u[1]),r1*np.sin(2*np.pi*u[1]),r2*np.cos(2*np.pi*u[3])])


def merge(state,pid1,pid2,*,mass=None,kick=None,books=None,G=1,softening=0):
    """Merge at the COM while retaining one stable ID; accounts include external potential changes."""
    from . import State
    ids=np.asarray(state.ids);i=int(np.flatnonzero(ids==pid1)[0]);j=int(np.flatnonzero(ids==pid2)[0])
    x,v,m=[np.asarray(a).copy() for a in state[:3]]
    before=diagnostics(state,G=G,softening=softening)
    total=m[i]+m[j]
    x[i]=(m[i]*x[i]+m[j]*x[j])/total
    v[i]=(m[i]*v[i]+m[j]*v[j])/total+(np.zeros(3) if kick is None else kick)
    m[i]=total if mass is None else mass
    keep=np.arange(len(m))!=j
    result=State.create(x[keep],v[keep],m[keep],ids=ids[keep])
    if books:books.change(before,diagnostics(result,G=G,softening=softening),'merge')
    return result
