"""Stellar properties keyed by stable particle ID."""

from dataclasses import asdict, dataclass, field, replace

import jax.numpy as jnp
import numpy as np

from .events import natal_kick
from .reference import diagnostics


@dataclass
class Star:
    zams_mass: float
    type: int = 1
    age: float = 0.
    radius: float = 0.
    core_mass: float = 0.
    luminosity: float = 0.
    spin: list = field(default_factory=lambda: [0., 0., 0.])


class Stars:
    def __init__(self, state, *, type=1, radius=0.):
        self.data = {int(pid): Star(float(mass), type=type, radius=radius)
                     for pid, mass in zip(np.asarray(state.ids), np.asarray(state.mass))}

    def __getitem__(self, pid):
        return self.data[int(pid)]

    def carry(self):
        return {str(pid): asdict(star) for pid, star in self.data.items()}

    @classmethod
    def restore(cls, carry):
        result = cls.__new__(cls)
        result.data = {int(pid): Star(**star) for pid, star in carry.items()}
        return result

    def evolve(self, state, engine, dt_myr, books, *, G, softening=0., kicks="hobbs", seed=42):
        before = diagnostics(state, G=G, softening=softening)
        mass, velocity = np.asarray(state.mass).copy(), np.asarray(state.velocity).copy()
        for i, pid in enumerate(np.asarray(state.ids)):
            star = self[pid]
            if star.type >= 10:
                continue
            age = star.age+dt_myr
            value = engine.evolve(star.zams_mass, age)
            if value['mass'] < 0:
                raise RuntimeError("SSE returned negative mass")
            if value['type'] in (13, 14) and star.type < 13 and kicks == "hobbs":
                velocity[i] += natal_kick(pid, seed, value['type'], value['fallback'], value['ecsn'])
            mass[i] = value['mass']
            star.type, star.age, star.radius = value['type'], age, value['radius']
            star.core_mass, star.luminosity = value['core_mass'], value['luminosity']
        result = state._replace(mass=jnp.asarray(mass), velocity=jnp.asarray(velocity))
        books.change(before, diagnostics(result, G=G, softening=softening), 'sse')
        return result

    def merged(self, keep, gone, mass, kind, spin):
        a, b = self[keep], self[gone]
        remnant_type = max(a.type,b.type) if max(a.type,b.type)>=10 else a.type
        self.data[int(keep)] = replace(a,type=remnant_type,spin=list(spin),
            zams_mass=mass if remnant_type<10 else a.zams_mass,age=0. if remnant_type<10 else a.age)
        self.data.pop(int(gone))
