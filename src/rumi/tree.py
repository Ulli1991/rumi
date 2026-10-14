"""CPU Barnes-Hut force/gradient backend for functional validation."""

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np


@dataclass
class Cell:
    mass: float
    center: np.ndarray
    width: float
    rows: np.ndarray
    ids: set
    subsystems: set
    children: tuple


def build(position,mass,ids,subsystem,rows,leaf_size,depth=0):
    lower,upper=position[rows].min(axis=0),position[rows].max(axis=0)
    total=float(mass[rows].sum())
    center=np.average(position[rows],weights=mass[rows],axis=0)
    width=float(np.max(upper-lower))
    children=()
    if len(rows)>leaf_size and width>0 and depth<48:
        middle=(lower+upper)/2
        octants=np.sum((position[rows]>=middle)*np.array([1,2,4]),axis=1)
        children=tuple(build(position,mass,ids,subsystem,rows[octants==k],leaf_size,depth+1)
                       for k in range(8) if np.any(octants==k))
    return Cell(total,center,width,rows,set(map(int,ids[rows])),set(map(int,subsystem[rows])),children)


def make_interactions(*,G=1.,softening=0.,theta=.7,leaf_size=32,threshold=0):
    if theta<0 or leaf_size<1 or threshold<0:
        raise ValueError('invalid tree parameters')
    from .interactions import make_interactions as direct
    direct_force,direct_gradient=direct(G=G,softening=softening)

    def evaluate(target,source,target_acceleration=None,source_acceleration=None):
        if len(source.mass)<threshold or len(source.mass)==0:
            return direct_force(target,source) if target_acceleration is None else direct_gradient(target,source,target_acceleration,source_acceleration)
        tx,tm,tid,tsid=map(np.asarray,(target.position,target.mass,target.ids,target.subsystem))
        sx,sm,sid,ssid=map(np.asarray,(source.position,source.mass,source.ids,source.subsystem))
        live=np.flatnonzero(sm>0)
        output=np.zeros_like(tx)
        if not len(live):
            return jnp.asarray(output)
        root=build(sx,sm,sid,ssid,live,leaf_size)
        ta=np.asarray(target_acceleration) if target_acceleration is not None else None
        sa=np.asarray(source_acceleration) if source_acceleration is not None else None

        def term(r,mass,delta=None):
            r2=np.sum(r*r,axis=-1)+softening**2
            if np.any(r2<=0):
                raise RuntimeError('singular tree interaction')
            if delta is None:
                return G*mass[...,None]*r/r2[...,None]**1.5
            return G*mass[...,None]*(r2[...,None]*delta-3*np.sum(r*delta,axis=-1)[...,None]*r)/r2[...,None]**2.5

        for i in range(len(tm)):
            if tm[i]<=0:
                continue
            stack=[root]
            while stack:
                cell=stack.pop()
                r=cell.center-tx[i]
                contains=int(tid[i]) in cell.ids or (tsid[i]>=0 and int(tsid[i]) in cell.subsystems)
                if cell.children and not contains and cell.width**2<theta**2*float(r@r):
                    delta=np.average(sa[cell.rows],weights=sm[cell.rows],axis=0)-ta[i] if ta is not None else None
                    output[i]+=term(r,np.asarray(cell.mass),delta)
                elif cell.children:
                    stack.extend(reversed(cell.children))
                else:
                    rows=cell.rows
                    valid=(sid[rows]!=tid[i])&~((tsid[i]>=0)&(ssid[rows]==tsid[i]))
                    rows=rows[valid]
                    delta=sa[rows]-ta[i] if ta is not None else None
                    output[i]+=np.sum(term(sx[rows]-tx[i],sm[rows],delta),axis=0)
        return jnp.asarray(output)

    return (lambda t,s:evaluate(t,s)),(lambda t,s,ta,sa:evaluate(t,s,ta,sa))
