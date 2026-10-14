"""Cluster structure and mutual most-bound binary diagnostics."""

import numpy as np


FRACTIONS=np.array([.01,.05,.1,.25,.5,.75,.9])


def cluster(state,*,G,time_myr=0.,pair_radius=.02):
    x,v,m=map(np.asarray,state[:3])
    live=m>0;x,v,m=x[live],v[live],m[live]
    n=len(m)
    if not n or pair_radius<=0:
        raise ValueError('nonempty cluster and positive pair radius required')
    total=m.sum();center=np.average(x,weights=m,axis=0)
    radius2=np.max(np.sum((x-center)**2,axis=1))
    stop=max(min(n,1000),n//100)
    for _ in range(2000):
        radius2*=.975**2
        inside=np.sum((x-center)**2,axis=1)<radius2
        if inside.sum()<stop:
            break
        center=np.average(x[inside],weights=m[inside],axis=0)
    sample=np.argsort(np.sum((x-center)**2,axis=1),kind='stable')[:20000]
    density=np.zeros(len(sample))
    core_radius,core_density=0.,0.
    if len(sample)>6:
        for q,i in enumerate(sample):
            distance2=np.sum((x[sample]-x[i])**2,axis=1)
            distance2[q]=np.inf
            neighbours=np.argsort(distance2,kind='stable')[:6]
            radius=np.sqrt(distance2[neighbours[-1]])
            if radius>0:
                density[q]=3*m[sample[neighbours[:5]]].sum()/(4*np.pi*radius**3)
        if density.sum()>0:
            center=np.average(x[sample],weights=density,axis=0)
            core_radius=np.sqrt(np.average(np.sum((x[sample]-center)**2,axis=1),weights=density**2))
            core_density=np.sum(density**2)/density.sum()
    radius=np.linalg.norm(x-center,axis=1)
    order=np.argsort(radius,kind='stable');cumulative=np.cumsum(m[order])
    lagrangian=radius[order[np.searchsorted(cumulative,total*FRACTIONS)]]
    central=radius<=lagrangian[2]
    velocity_center=np.average(v[central],weights=m[central],axis=0)
    dispersion=np.sqrt(np.sum(m[central,None]*(v[central]-velocity_center)**2)/(3*m[central].sum()))
    partner=np.full(n,-1);energy=np.zeros(n)
    for i in range(n):
        distance=np.linalg.norm(x-x[i],axis=1)
        rows=np.flatnonzero((distance>0)&(distance<pair_radius))
        if len(rows):
            reduced=m[i]*m[rows]/(m[i]+m[rows])
            binding=.5*reduced*np.sum((v[rows]-v[i])**2,axis=1)-G*m[i]*m[rows]/distance[rows]
            q=int(np.argmin(binding))
            if binding[q]<0:partner[i],energy[i]=rows[q],binding[q]
    pairs=[i for i,j in enumerate(partner) if j>i and partner[j]==i]
    return dict(time_myr=time_myr,mass=float(total),center=center.tolist(),velocity_center=velocity_center.tolist(),
                core_radius=float(core_radius),core_density=float(core_density),lagrangian=lagrangian.tolist(),
                sample_size=len(sample) if len(sample)>6 else 0,dispersion=float(dispersion),
                binaries=len(pairs),hard_binaries=sum(-energy[i]>total/n*dispersion**2 for i in pairs),
                binary_energy=float(energy[pairs].sum()))


def potential(state,*,G,softening=0.):
    x,m=map(np.asarray,(state.position,state.mass))
    phi=np.zeros(len(m))
    for i in range(len(m)):
        valid=(np.arange(len(m))!=i)&(m>0)
        r2=np.sum((x[valid]-x[i])**2,axis=1)+softening**2
        phi[i]=-G*np.sum(m[valid]/np.sqrt(r2))
    return phi
