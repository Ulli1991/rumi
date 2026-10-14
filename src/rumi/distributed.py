"""Replicated-state, sink-distributed interactions through an explicit MPI communicator."""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np

from .interactions import take


class Replicas:
    def __init__(self,comm):
        self.comm=comm
        self.rank,self.size=comm.Get_rank(),comm.Get_size()

    def call(self,function,*args):
        result,error=None,None
        try:
            result=function(*args)
            jax.block_until_ready(result)
        except Exception as e:
            error=f'{type(e).__name__}: {e}'
        failures=self.comm.allgather(error)
        if any(failures):
            raise RuntimeError(f'MPI operation failed: {failures}')
        return result

    def interactions(self,force,gradient):
        def evaluate(target,source,ta=None,sa=None):
            count=len(target.mass)
            lo,hi=count*self.rank//self.size,count*(self.rank+1)//self.size
            subset=take(target,np.arange(lo,hi))
            if ta is None:
                value=self.call(force,subset,source)
            else:
                value=self.call(gradient,subset,source,ta[lo:hi],sa)
            chunks=self.comm.allgather(np.asarray(value))
            return jnp.asarray(np.concatenate(chunks,axis=0))
        return (lambda t,s:evaluate(t,s)),(lambda t,s,ta,sa:evaluate(t,s,ta,sa))

    def agree(self,state,carry):
        digest=hashlib.sha256()
        for array in state:digest.update(np.ascontiguousarray(array).tobytes())
        digest.update(json.dumps(carry,sort_keys=True,allow_nan=False).encode())
        hashes=self.comm.allgather(digest.hexdigest())
        if len(set(hashes))!=1:
            raise RuntimeError('MPI replicas disagree; checkpoint refused')
