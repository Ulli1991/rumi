"""Gadget/LUNA-compatible HDF5 snapshots and frozen-array imports."""

import json
import os
from pathlib import Path

import numpy as np

from . import State
from .diagnostics import cluster, potential, FRACTIONS
from .initial_conditions import TIME_UNIT_MYR
from .stars import Star, Stars


def hdf5():
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError('HDF5 requires h5py: install the rumi-nbody dependencies') from e
    return h5py


def write(path,run):
    h5py=hdf5()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.partial')
    state,c=run.state,run.config
    x,v,m,ids,sid=map(np.asarray,state)
    types=np.array([run.stars[pid].type for pid in ids])
    particle_type=np.where(types==14,5,4)
    counts=np.bincount(particle_type,minlength=6).astype(np.uint64)
    box=max(1e-30,2*float(np.max(np.abs(x)))*1.01) if len(m) else 1.
    phi=potential(state,G=run.G,softening=run.softening) if c.SNAPSHOT_POTENTIAL else None
    kinetic=float(.5*np.sum(m[:,None]*v*v))
    report=run.report()
    steps=run.integrator._steps(state,np.arange(len(m))) if len(m) else np.empty(0)
    maximum=c.DT_MAX_MYR/TIME_UNIT_MYR
    bins=np.zeros(len(m),np.int32)
    fast=steps<maximum
    bins[fast]=np.ceil(np.log2(maximum/np.maximum(steps[fast],c.DT_MIN_MYR/TIME_UNIT_MYR))).astype(np.int32)
    partners=np.full(len(m),-1,np.int64)
    for i in range(len(m)):
        candidates=np.flatnonzero((sid[i]>=0)&(sid==sid[i])&(ids!=ids[i]))
        if len(candidates):
            r=np.linalg.norm(x[candidates]-x[i],axis=1)
            energy=.5*np.sum((v[candidates]-v[i])**2,axis=1)-run.G*(m[i]+m[candidates])/r
            q=int(np.argmin(energy))
            if energy[q]<0:partners[i]=ids[candidates[q]]
    try:
        with h5py.File(temp,'w') as f:
            header=f.create_group('Header')
            header.attrs.update(BoxSize=box,Time=run.time,Time_Myr=run.time*TIME_UNIT_MYR,Step=run.step_number,
                SnapshotIndex=run.snapshot_index,N=len(m),G=run.G,Softening_pc=run.softening,
                NumPart_ThisFile=counts,NumPart_Total=counts,NumPart_Total_HighWord=np.zeros(6,np.uint64),
                NumFilesPerSnapshot=1,MassTable=np.zeros(6),Flag_DoublePrecision=1,DoublePrecision=1,
                Redshift=0.,Omega0=0.,OmegaLambda=0.,HubbleParam=1.,Code='rumi',CodeVersion='0.1.0',
                config_json=json.dumps(c.values),E_kin=kinetic,E_pot=report['energy']-kinetic,
                E_total=report['energy'],E_total_initial=run.initial['energy'],dE_rel=report['relative_energy_error'],
                P_sink=run.books.momentum,L_sink=run.books.angular_momentum,
                UnitLength_in_cm=3.08567758e18,UnitMass_in_g=1.989e33,UnitVelocity_in_cm_per_s=1e5)
            for name,key in (('E_escape','escape'),('E_gw','gw'),('E_merge','merge'),('E_sse','sse'),('E_pn','pn'),('E_sd','sdar'),('E_tidal','tidal')):
                header.attrs[name]=run.books.energy.get(key,0.)
            if c.SNAPSHOT_CLUSTER_DIAG and len(m):
                value=cluster(state,G=run.G,time_myr=run.time*TIME_UNIT_MYR,pair_radius=c.CLUSTER_DIAG_RPAIR_PC)
                mapping={'mass':'Cluster_MassTotal','center':'Cluster_DensityCentre','velocity_center':'Cluster_DensityCentreVelocity',
                    'core_radius':'Cluster_CoreRadius_pc','core_density':'Cluster_CoreDensity','dispersion':'Cluster_Sigma1D_kms',
                    'lagrangian':'Cluster_LagrangianRadii_pc','binaries':'Cluster_NBinaries','hard_binaries':'Cluster_NBinariesHard','binary_energy':'Cluster_EBinaries'}
                for key,name in mapping.items():header.attrs[name]=value[key]
                header.attrs['Cluster_LagrangianFractions']=FRACTIONS
            for kind in (4,5):
                rows=np.flatnonzero(particle_type==kind)
                if not len(rows):continue
                p=f.create_group(f'PartType{kind}')
                values=dict(Coordinates=x[rows],Velocities=v[rows],Masses=m[rows],ParticleIDs=ids[rows],
                    StellarType=types[rows],Spin=np.array([run.stars[ids[i]].spin for i in rows]),
                    SubsystemID=sid[rows],BinaryPartnerID=partners[rows],TimeStep=steps[rows],TimeBin=bins[rows])
                for name,key in (('StellarRadius','radius'),('Luminosity','luminosity'),('CoreMass','core_mass'),('ZAMSMass','zams_mass'),('StellarAge','age')):
                    values[name]=np.array([getattr(run.stars[ids[i]],key) for i in rows])
                if phi is not None:values['Potential']=phi[rows]
                for name,value in values.items():p.create_dataset(name,data=value)
            unit=f.create_group('units')
            length,mass,velocity=3.08567758e18,1.989e33,1e5
            time=length/velocity
            unit.attrs.update(UnitLength_in_cm=length,UnitMass_in_g=mass,UnitVelocity_in_cm_per_s=velocity,
                UnitTime_in_s=time,UnitDensity_in_cgs=mass/length**3,UnitPressure_in_cgs=mass/(length*time*time),
                UnitEnergy_in_cgs=mass*velocity**2)
            cells=f.create_group('cells');speed=np.average(v,weights=m,axis=0) if len(m) else np.zeros(3);density=float(m.sum()/box**3)
            cells.attrs.update(time=run.time,step=run.step_number,process_id=0,process_count=1,shard_idx=0,
                gamma=5/3,spatial_dimension=3,order=0,limiter=0,config_json=json.dumps(c.values),placeholder=1)
            for name in ('Lx','Ly','Lz','dx','dy','dz'):cells.attrs[name]=box
            for name in ('global_Nx','Nx_global','global_Ny','Ny_global','global_Nz','Nz_global','local_Nx','local_Ny','local_Nz'):cells.attrs[name]=1
            for name in ('x_start','y_start','z_start'):cells.attrs[name]=0
            for name,value in dict(density=density,velocity_x=speed[0],velocity_y=speed[1],velocity_z=speed[2],internal_energy=0.,pressure=0.).items():
                cells.create_dataset(name,data=np.full((1,1,1),value))
            cells.create_dataset('U',data=np.array([density,*(density*speed),.5*density*(speed@speed)]).reshape(5,1,1,1))
        os.replace(temp,path)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    run.snapshot_index+=1


def read(path):
    h5py=hdf5()
    arrays=[];properties={}
    with h5py.File(path,'r') as f:
        for kind in range(6):
            key=f'PartType{kind}'
            if key not in f:continue
            p=f[key];ids=np.asarray(p['ParticleIDs'],np.int64)
            if 'Masses' in p:mass=np.asarray(p['Masses'])
            else:mass=np.full(len(ids),f['Header'].attrs['MassTable'][kind])
            arrays.append((np.asarray(p['Coordinates']),np.asarray(p['Velocities']),mass,ids))
            for i,pid in enumerate(ids):
                value=Star(float(p['ZAMSMass'][i]) if 'ZAMSMass' in p else float(mass[i]),type=14 if kind==5 else 1)
                for name,attr in (('StellarType','type'),('StellarRadius','radius'),('Luminosity','luminosity'),('CoreMass','core_mass'),('StellarAge','age'),('Spin','spin')):
                    if name in p:
                        item=np.asarray(p[name][i])
                        setattr(value,attr,item.tolist())
                properties[int(pid)]=value
        header={key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in f['Header'].attrs.items()}
    if not arrays:raise ValueError('snapshot contains no particles')
    x,v,m,ids=(np.concatenate([a[k] for a in arrays]) for k in range(4))
    order=np.argsort(ids,kind='stable')
    state=State.create(x[order],v[order],m[order],ids=ids[order])
    stars=Stars(state);stars.data=properties
    return state,stars,header
