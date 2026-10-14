"""Optional reversible integer field drifts with task-owned position words."""

import jax.numpy as jnp
import numpy as np


class IntegerPositions:
    def __init__(self,half_size):
        if not np.isfinite(half_size) or half_size<=0:
            raise ValueError('positive finite domain half-size required')
        self.scale=half_size/2**62
        self.words={}

    def quantize(self,values):
        rounded=np.rint(np.asarray(values)/self.scale)
        if np.any(np.abs(rounded)>=2.**63):
            raise RuntimeError('integer position exceeds its representable domain')
        return rounded.astype(np.int64)

    def sync(self,state,indices=None):
        indices=np.arange(len(state.mass)) if indices is None else np.asarray(indices)
        words=self.quantize(np.asarray(state.position[indices]))
        for pid,row in zip(np.asarray(state.ids[indices]),words):self.words[int(pid)]=row.tolist()
        return state._replace(position=state.position.at[indices].set(jnp.asarray(words*self.scale)))

    def drift(self,state,indices,dt):
        indices=np.asarray(indices)
        delta=self.quantize(dt*np.asarray(state.velocity[indices]))
        words=[]
        for pid,change in zip(np.asarray(state.ids[indices]),delta):
            current=self.words[int(pid)]
            value=[a+int(b) for a,b in zip(current,change)]
            if any(v<-(1<<63) or v>=(1<<63) for v in value):
                raise RuntimeError('integer drift overflow')
            self.words[int(pid)]=value
            words.append(value)
        values=np.asarray(words,dtype=np.int64).reshape(-1,3)*self.scale
        return state._replace(position=state.position.at[indices].set(jnp.asarray(values)))

    def carry(self):
        return {str(pid):row.copy() for pid,row in self.words.items()}

    def restore(self,carry):
        self.words={int(pid):row.copy() for pid,row in carry.items()}
