"""Atomic portable checkpoints of particle and Python-owned physics carries."""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import State


def save(path,state,*,time,step,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays={name:np.asarray(value) for name,value in zip(State._fields,state)}
    header=dict(format='rumi-checkpoint',version=2,time=float(time),step=int(step),metadata=metadata or {})
    if not np.isfinite(header['time']) or header['time']<0 or header['step']<0:
        raise ValueError('invalid checkpoint time or step')
    header['metadata_sha256']=hashlib.sha256(json.dumps(header['metadata'],sort_keys=True,allow_nan=False).encode()).hexdigest()
    hashes={name:hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest() for name,a in arrays.items()}
    header['hashes']=hashes
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',delete=False) as f:
        temp=Path(f.name)
        try:
            np.savez(f,**arrays,header=np.array(json.dumps(header,allow_nan=False)))
            f.flush();os.fsync(f.fileno())
        except BaseException:
            temp.unlink(missing_ok=True);raise
    try:os.replace(temp,path)
    except BaseException:temp.unlink(missing_ok=True);raise


def load(path):
    with np.load(path,allow_pickle=False) as data:
        header=json.loads(str(data['header']))
        if header.get('format')!='rumi-checkpoint' or header.get('version') not in (1,2):
            raise ValueError('unsupported checkpoint format')
        arrays={name:np.asarray(data[name]) for name in State._fields}
    for name,a in arrays.items():
        if hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()!=header['hashes'][name]:
            raise ValueError(f'checkpoint checksum failed: {name}')
    if header['version']==2 and hashlib.sha256(json.dumps(header['metadata'],sort_keys=True,allow_nan=False).encode()).hexdigest()!=header['metadata_sha256']:
        raise ValueError('checkpoint metadata checksum failed')
    if not np.isfinite(header['time']) or header['time']<0 or header['step']<0:
        raise ValueError('invalid checkpoint time or step')
    state=State.create(arrays['position'],arrays['velocity'],arrays['mass'],ids=arrays['ids'],subsystem=arrays['subsystem'],allow_empty=True)
    return state,header
