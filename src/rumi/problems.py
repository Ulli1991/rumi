"""Upstream test problems and stellar-cluster initial conditions."""

import numpy as np

from . import State
from .events import collision_distance, gw_coalescence_time
from .initial_conditions import AU_TO_PC, G_CODE, TIME_UNIT_MYR, plummer
from .physics import NativePhysics, StellarEvolution
from .stars import Star, Stars
from .subsystems import elements


def orbit(a,e,mu,*,peri=False,start=5.,rotation=0.):
    if a<=0 or e<0 or start<1:
        raise ValueError('invalid orbital initial conditions')
    if e>=1:
        p=a*(1+e);distance=start*a
        f=-np.arccos(np.clip((p/distance-1)/e,-1,1))
        r=np.array([distance*np.cos(f),distance*np.sin(f),0.])
        v=np.sqrt(mu/p)*np.array([-np.sin(f),e+np.cos(f),0.])
    else:
        distance=a*(1-e if peri else 1+e)
        speed=np.sqrt(mu/a*(1+e if peri else 1-e)/(1-e if peri else 1+e))
        sign=1 if peri else -1
        r,v=np.array([sign*distance,0.,0.]),np.array([0.,sign*speed,0.])
    c,s=np.cos(rotation),np.sin(rotation)
    matrix=np.array([[c,-s,0],[s,c,0],[0,0,1]])
    return matrix@r,matrix@v


def split_pair(r,v,mass):
    m1,m2=mass
    fractions=np.array([-m2,m1])/(m1+m2)
    return fractions[:,None]*r,fractions[:,None]*v


def outer_planets():
    x=np.array([[0,0,0],[-3.5023653,-3.8169847,-1.5507963],[9.0755314,-3.0458353,-1.6483708],
                [8.3101420,-16.2901086,-7.2521278],[11.4707666,-25.7294829,-10.8169456]])*AU_TO_PC
    v=np.array([[0,0,0],[.00565429,-.00412490,-.00190589],[.00168318,.00483525,.00192462],
                [.00354178,.00137102,.00055029],[.00288930,.00114527,.00039677]])*(149597870.7/86400.)
    return State.create(x,v,[1.,9.54786104043e-4,2.85583733151e-4,4.37273164546e-5,5.17759138449e-5])


def create(config):
    c=config
    if c.TEST_CASE=='outer_planets':
        state=outer_planets()
        return state,Stars(state)
    if c.TEST_CASE in ('binary_test','binary_test_peri'):
        mass=np.array([c.BINARY_M1,c.BINARY_M2])
        r,v=orbit(c.BINARY_A_PC,c.BINARY_E,G_CODE*mass.sum(),peri=c.TEST_CASE=='binary_test_peri',
                  rotation=np.deg2rad(c.BINARY_ROT_DEG))
        x,v=split_pair(r,v,mass)
    elif c.TEST_CASE=='triple_test':
        mass=np.array([c.TRIPLE_M1,c.TRIPLE_M2,c.TRIPLE_M3])
        r,v=orbit(c.TRIPLE_A_IN_PC,c.TRIPLE_E_IN,G_CODE*sum(mass[:2]),start=c.TRIPLE_IN_START_Q,
                  rotation=np.deg2rad(c.TRIPLE_IN_ROT_DEG))
        x,v=split_pair(r,v,mass[:2])
        R,V=orbit(c.TRIPLE_A_OUT_PC,c.TRIPLE_E_OUT,G_CODE*mass.sum(),start=c.TRIPLE_OUT_START_Q)
        if c.TRIPLE_E_OUT<1:
            R,V=-R,-V
        angle=np.deg2rad(c.TRIPLE_INCL_DEG)
        tilt=np.array([[1,0,0],[0,np.cos(angle),np.sin(angle)],[0,-np.sin(angle),np.cos(angle)]])
        R,V=tilt@R,tilt@V
        x=np.vstack([x-mass[2]/mass.sum()*R,sum(mass[:2])/mass.sum()*R])
        v=np.vstack([v-mass[2]/mass.sum()*V,sum(mass[:2])/mass.sum()*V])
        if c.TRIPLE_OUTER_FIRST:
            x,v,mass=x[[2,0,1]],v[[2,0,1]],mass[[2,0,1]]
    elif c.TEST_CASE=='plummer':
        return cluster(c)
    else:
        raise ValueError(f'unsupported TEST_CASE {c.TEST_CASE}')
    if c.TEST_BOOST_KMS:
        v+=parse_vector(c.TEST_BOOST_KMS)
    state=State.create(x,v,mass)
    stars=Stars(state,radius=c.TEST_STELLAR_RADIUS_RSUN)
    for pid,m in zip(np.asarray(state.ids),mass):
        stars[pid].type=14 if m>=5 else 0
    for pid,value in enumerate((c.BINARY_CHI1,c.BINARY_CHI2)):
        if value and c.TEST_CASE.startswith('binary'):
            stars[pid].spin=parse_vector(value).tolist()
    return state,stars


def parse_vector(text):
    values=np.array([float(v) for v in text.replace(',',' ').split()])
    if values.shape!=(3,) or not np.isfinite(values).all():
        raise ValueError('expected three finite vector components')
    return values


def cluster(c):
    distribution='kroupa' if c.PLUMMER_MASS_DIST!='equal' else 'equal'
    sample=plummer(c.PLUMMER_N,mass=c.PLUMMER_M_TOTAL,seed=c.PLUMMER_SEED,mass_distribution=distribution,G=G_CODE,mmax=150.)
    m=np.asarray(sample.mass).copy()
    if distribution=='kroupa' and c.PLUMMER_MASS_DIST=='kroupa':
        m*=c.PLUMMER_M_TOTAL/m.sum()
    stars=Stars(sample)
    if c.PLUMMER_MASS_DIST=='kroupa_evolved':
        with StellarEvolution(metallicity=c.PLUMMER_METALLICITY,wind=c.SSE_WIND_MODEL,remnant=c.SSE_REMNANT_MODEL,
            ppisn=c.SSE_PPISN_MODEL,ecsn=c.SSE_ECSN_MODEL,max_ns_mass=c.SSE_MXNS,lbv_factor=c.SSE_FLBV) as sse:
            for pid in stars.data:
                result=sse.evolve(float(m[pid]),c.PLUMMER_AGE_GYR*1000)
                stars.data[pid]=Star(float(m[pid]),type=result['type'],age=c.PLUMMER_AGE_GYR*1000,
                    radius=result['radius'],core_mass=result['core_mass'],luminosity=result['luminosity'])
                m[pid]=result['mass']
    else:
        for pid,mass in enumerate(m):
            stars.data[pid]=Star(float(mass),radius=float(mass**.8),luminosity=float(mass**3.5))
    radius=(2.55*(m.sum()/1e4)**.24 if m.sum()<10**6.5 else (2**(2/3)-1)**(-.5)*10**(.5*np.log10(m.sum())-2.75)) if c.PLUMMER_R_HALF=='mass_size' else float(c.PLUMMER_R_HALF)
    rng=np.random.default_rng(c.PLUMMER_SEED)
    points=[plummer_point(float(m.sum()),radius,rng) for _ in m]
    x,v=np.array([p[0] for p in points]),np.array([p[1] for p in points])
    x-=np.average(x,weights=m,axis=0)
    v-=np.average(v,weights=m,axis=0)
    keep=np.array([m[i]>0 and not (c.REMOVE_BH and stars[i].type==14) for i in range(len(m))])
    ids=np.arange(len(m))[keep]
    state=State.create(x[keep],v[keep],m[keep],ids=ids)
    stars.data={int(pid):stars[pid] for pid in ids}
    state=primordial_binaries(state,stars,c,radius)
    if c.ADD_IMBH:
        mass=np.array([c.IMBH_M1,c.IMBH_M2])
        r,v=orbit(c.IMBH_A_PC,c.IMBH_E,G_CODE*mass.sum())
        bx,bv=split_pair(r,v,mass)
        new_ids=np.arange(int(np.asarray(state.ids).max())+1,int(np.asarray(state.ids).max())+3)
        state=State.create(np.vstack([state.position,bx]),np.vstack([state.velocity,bv]),np.r_[state.mass,mass],ids=np.r_[state.ids,new_ids])
        for pid,value in zip(new_ids,mass):stars.data[int(pid)]=Star(float(value),type=14,spin=[0.,0.,c.IMBH_SPIN])
    return state,stars


def primordial_binaries(state,stars,c,half_mass_radius):
    rng=np.random.default_rng(c.PLUMMER_SEED+1)
    x,v,m,ids=(np.asarray(a).copy() for a in state[:4])
    count=min(len(m)//2,int(c.BINARY_FRAC*len(m)/2)) if c.COLLISIONAL else 0
    if not 0<=c.BINARY_FRAC<=1 or not 0<c.BINARY_A_MIN_AU<=c.BINARY_A_MAX_AU:
        raise ValueError('invalid primordial binary parameters')
    pairs=[]
    remaining=set(range(len(m)))
    if c.BINARY_POP=='mass_rank':
        order=np.argsort(-m,kind='stable')
        pairs=list(zip(order[:2*count:2],order[1:2*count:2]))
    else:
        while len(pairs)<count and len(remaining)>=2:
            i=max(remaining,key=lambda k:(m[k],k))
            if m[i]<=5:
                break
            remaining.remove(i)
            target=rng.uniform(.1,1)*m[i]
            j=min(remaining,key=lambda k:(abs(m[k]-target),m[k],k))
            remaining.remove(j);pairs.append((i,j))
        rest=rng.permutation(sorted(remaining))
        pairs.extend((i,j) if m[i]>=m[j] else (j,i) for i,j in zip(rest[:2*(count-len(pairs)):2],rest[1:2*(count-len(pairs)):2]))
    break_au=c.R_BREAK_AU if c.R_BREAK_AU>0 else c.IC_R_BREAK*c.BINARY_A_MAX_AU
    if break_au>0:
        avoid_artificial_pairs(x,v,m,pairs,break_au*AU_TO_PC,half_mass_radius,rng)
    if not pairs:
        return State.create(x,v,m,ids=ids)
    native=NativePhysics()
    for i,j in pairs:
        a,b=stars[ids[i]],stars[ids[j]]
        contact=collision_distance(a.type,m[i],a.radius*2.254e-8,b.type,m[j],b.radius*2.254e-8,G=G_CODE,bh_radius=c.BH_MERGE_RADIUS)[0]
        semimajor=np.exp(rng.uniform(np.log(c.BINARY_A_MIN_AU*AU_TO_PC),np.log(c.BINARY_A_MAX_AU*AU_TO_PC)))
        e=np.sqrt(rng.random())
        if c.BINARY_PERI_CHECK:
            if c.BINARY_A_MAX_AU*AU_TO_PC<=contact:
                continue
            if semimajor<=contact:
                semimajor=np.exp(rng.uniform(np.log(max(c.BINARY_A_MIN_AU*AU_TO_PC,contact)),np.log(c.BINARY_A_MAX_AU*AU_TO_PC)))
            e=min(e, (1-contact/semimajor)*np.sqrt(rng.random())) if semimajor*(1-e)<=contact else e
            if a.type>=10 and b.type>=10:
                for _ in range(1000):
                    if gw_coalescence_time(semimajor,e,m[i],m[j],G=G_CODE,c=299792.458)>=.01/TIME_UNIT_MYR:
                        break
                    e*=np.sqrt(rng.random())
                else:
                    continue
        r,u=native.elements(semimajor,e,np.arccos(rng.uniform(-1,1)),rng.uniform(0,2*np.pi),
                             rng.uniform(0,2*np.pi),rng.uniform(0,2*np.pi),G_CODE*(m[i]+m[j]))
        center,speed=x[i].copy(),v[i].copy()
        offsets,velocities=split_pair(r,u,m[[i,j]])
        x[[i,j]],v[[i,j]]=center+offsets,speed+velocities
    x-=np.average(x,weights=m,axis=0)
    v-=np.average(v,weights=m,axis=0)
    return State.create(x,v,m,ids=ids)


def avoid_artificial_pairs(x,v,m,pairs,distance,half_mass_radius,rng):
    object_mass=m.copy();objects=set(range(len(m)))
    for i,j in pairs:
        object_mass[i]+=m[j];objects.remove(j)
    objects=np.array(sorted(objects))
    for iteration in range(20):
        flagged=set()
        for q,i in enumerate(objects):
            for j in objects[q+1:]:
                r=np.linalg.norm(x[j]-x[i])
                if r>=2*distance:
                    continue
                close=r<distance
                if not close:
                    o=elements(x[j]-x[i],v[j]-v[i],G_CODE*(object_mass[i]+object_mass[j]))
                    close=o['energy']<0 and o['a']<distance
                if close:flagged.update((i,j))
        if not flagged:
            break
        for i in sorted(flagged):
            radius,speed=np.linalg.norm(x[i]),np.linalg.norm(v[i])
            if iteration>=10:
                px,pv=plummer_point(float(m.sum()),half_mass_radius,rng)
                radius,speed=np.linalg.norm(px),np.linalg.norm(pv)
            for array,norm in ((x,radius),(v,speed)):
                direction=rng.normal(size=3);array[i]=norm*direction/np.linalg.norm(direction)
    x[objects]-=np.average(x[objects],weights=object_mass[objects],axis=0)
    v[objects]-=np.average(v[objects],weights=object_mass[objects],axis=0)


def plummer_point(mass,half_mass_radius,rng):
    scale=half_mass_radius*np.sqrt(2**(2/3)-1)
    radius=scale/np.sqrt(np.clip(rng.random(),1e-10,1-1e-10)**(-2/3)-1)
    while True:
        q=rng.random()
        if .1*rng.random()<q*q*(1-q*q)**3.5:
            break
    speed=q*np.sqrt(2*G_CODE*mass/np.sqrt(radius*radius+scale*scale))
    directions=rng.normal(size=(2,3))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    return radius*directions[0],speed*directions[1]
