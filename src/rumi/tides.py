"""Galaxy reference-frame tides, kicks and Jacobi radii matching cpp-nbody tidal_field.cpp."""

from dataclasses import dataclass,field
import numpy as np
import jax.numpy as jnp


@dataclass
class TidalField:
    kind: str = 'none'
    G: float = 1.
    galaxy_mass: float = 0.
    circular_speed: float = 220.
    core_radius: float = 0.
    position: np.ndarray = field(default_factory=lambda:np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda:np.zeros(3))

    def acceleration(self,x):
        x=np.asarray(x);r2=np.sum(x*x,axis=-1)
        if self.kind=='pointmass':f=np.where(r2>0,-self.G*self.galaxy_mass/np.maximum(r2,1e-300)**1.5,0)
        elif self.kind=='loghalo':f=-self.circular_speed**2/(self.core_radius**2+r2)
        elif self.kind=='none':return np.zeros_like(x)
        else:raise ValueError('unknown tidal field')
        return np.asarray(f)[...,None]*x

    def initialize_circular(self,radius):
        self.position=np.array([radius,0.,0.])
        speed=np.sqrt(self.G*self.galaxy_mass/radius) if self.kind=='pointmass' else self.circular_speed*radius/np.sqrt(self.core_radius**2+radius**2)
        self.velocity=np.array([0.,speed,0.])

    def jacobi_radius(self,mass,position=None):
        position=self.position if position is None else np.asarray(position)
        r2=float(position@position)
        if self.kind=='pointmass':k=3*self.G*self.galaxy_mass/r2**1.5
        elif self.kind=='loghalo':k=2*self.circular_speed**2*r2/(self.core_radius**2+r2)**2
        else:return np.inf
        return np.cbrt(self.G*mass/k) if k>0 and mass>0 else 0.

    def kick(self,state,dt,books=None):
        if self.kind=='none':return state
        x,v,m=(np.asarray(a) for a in state[:3])
        ref=self.acceleration(self.position)
        dv=dt*(self.acceleration(x+self.position)-ref)
        if books:
            change=float(np.sum(m[:,None]*(v*dv+.5*dv*dv)))
            books.energy['tidal']=books.energy.get('tidal',0.)-change
            books.momentum-=np.sum(m[:,None]*dv,axis=0)
            books.angular_momentum-=np.sum(m[:,None]*np.cross(x,dv),axis=0)
        self.velocity=self.velocity+dt*ref
        return state._replace(velocity=state.velocity+jnp.asarray(dv))

    def drift(self,dt):self.position=self.position+dt*self.velocity
