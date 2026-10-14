"""Python IC utilities; imported particle arrays are the parity reference, not RNG equivalence."""

import numpy as np

from . import State

G_CODE=4.302194e-3
TIME_UNIT_MYR=.977792
AU_TO_PC=4.84814e-6


def binary(*,m1=10.,m2=10.,a=1e-8,e=.9,G=G_CODE,angle=0.):
    if not (m1>0 and m2>0 and a>0 and 0<=e<1):raise ValueError('invalid bound binary parameters')
    total=m1+m2;r=a*(1+e);v=np.sqrt(G*total/a*(1-e)/(1+e))
    position=np.array([[-m2/total*r,0,0],[m1/total*r,0,0]])
    velocity=np.array([[0,-m2/total*v,0],[0,m1/total*v,0]])
    c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return State.create(position@rotation.T,velocity@rotation.T,[m1,m2])


def plummer(n,*,mass=1.,half_mass_radius=1.,G=1.,seed=42,mass_distribution='equal',mmin=.08,mmax=100.):
    if n<2 or mass<=0 or half_mass_radius<=0:raise ValueError('invalid Plummer arguments')
    rng=np.random.default_rng(seed)
    if mass_distribution=='equal':m=np.ones(n)*mass/n
    elif mass_distribution=='kroupa':
        # Continuous broken power law (alpha 1.3 below 0.5 Msun, 2.3 above).
        segments=[]
        for lo,hi,alpha,weight in ((mmin,min(.5,mmax),1.3,1.),(max(.5,mmin),mmax,2.3,.5)):
            if hi>lo:segments.append((lo,hi,alpha,weight*(hi**(1-alpha)-lo**(1-alpha))/(1-alpha)))
        weights=np.array([s[3] for s in segments]);weights/=weights.sum()
        chosen=rng.choice(len(segments),n,p=weights);m=np.empty(n)
        for k,(lo,hi,alpha,_) in enumerate(segments):
            mask=chosen==k;u=rng.random(mask.sum());m[mask]=(lo**(1-alpha)+u*(hi**(1-alpha)-lo**(1-alpha)))**(1/(1-alpha))
    else:raise ValueError('unsupported mass distribution')
    scale=half_mass_radius*np.sqrt(2**(2/3)-1)
    radius=scale/np.sqrt(rng.random(n)**(-2/3)-1)
    directions=rng.normal(size=(n,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    x=radius[:,None]*directions
    q=np.empty(n)
    for i in range(n):
        while True:
            trial=rng.random()
            if rng.random()*.1<=trial**2*(1-trial**2)**3.5:q[i]=trial;break
    speed=q*np.sqrt(2*G*m.sum()/np.sqrt(radius**2+scale**2))
    directions=rng.normal(size=(n,3));directions/=np.linalg.norm(directions,axis=1)[:,None]
    v=speed[:,None]*directions
    x-=np.average(x,weights=m,axis=0);v-=np.average(v,weights=m,axis=0)
    return State.create(x,v,m)


def add_binaries(state,fraction,*,a_min=.01*AU_TO_PC,a_max=100*AU_TO_PC,G=G_CODE,seed=42):
    x,v,m=(np.asarray(a).copy() for a in state[:3])
    rng=np.random.default_rng(seed);n_pair=int(len(m)*fraction)//2
    for i,j in zip(np.arange(n_pair)*2,np.arange(n_pair)*2+1):
        a=np.exp(rng.uniform(np.log(a_min),np.log(a_max)));e=np.sqrt(rng.random())
        pair=binary(m1=m[i],m2=m[j],a=a,e=e,G=G,angle=rng.uniform(0,2*np.pi))
        center,speed=x[i].copy(),v[i].copy()
        x[[i,j]]=np.asarray(pair.position)+center;v[[i,j]]=np.asarray(pair.velocity)+speed
    return State.create(x,v,m,ids=state.ids)
