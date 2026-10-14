"""Task-owned Python interfaces to retained CPU physics; no Kokkos runtime or shared physics state."""

import ctypes as ct
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys

import numpy as np

PTR = ct.POINTER(ct.c_double)


def _array(value, shape=None):
    a = np.ascontiguousarray(value, dtype=np.float64)
    if (shape is not None and a.shape != shape) or not np.isfinite(a).all():
        raise ValueError(f"expected finite array of shape {shape}")
    return a


def _ptr(a):
    return a.ctypes.data_as(PTR)


class NativePhysics:
    def __init__(self, library=None):
        suffix = ".dylib" if sys.platform == "darwin" else ".so"
        path = library or os.environ.get("RUMI_CPU_LIBRARY", Path(__file__).resolve().parents[2]/"build"/("librumi_cpu"+suffix))
        self.lib = ct.CDLL(str(path))
        self.lib.rumi_kepler.argtypes = [PTR, PTR, ct.c_double, ct.c_double]
        self.lib.rumi_pn_pair.argtypes = [PTR, PTR, PTR, ct.c_double, ct.c_double, ct.c_int, PTR]
        self.lib.rumi_recoil.argtypes = [PTR, PTR, PTR, ct.c_double, ct.c_double, PTR]
        self.lib.rumi_bh_remnant.argtypes = [PTR, PTR, PTR, ct.c_double, ct.c_double, PTR]
        self.lib.rumi_pericentre.argtypes = [PTR, PTR, ct.c_double, PTR]
        self.lib.rumi_elements.argtypes = [PTR, ct.c_double, PTR, PTR]
        self.lib.rumi_secular_triple.argtypes = [PTR, PTR, PTR, ct.c_double, ct.c_double, ct.c_int, ct.c_double, PTR]
        self.lib.rumi_secular_binary.argtypes = [PTR, PTR, PTR, ct.c_double, ct.c_double, ct.c_int, ct.c_double]
        self.lib.rumi_full_encounter.argtypes = [ct.c_int, PTR, PTR, PTR, PTR, PTR, PTR, PTR, PTR, PTR, ct.c_char_p, ct.c_int]
        self.lib.rumi_sse_create.argtypes = [ct.c_double, ct.c_char_p, ct.c_char_p, ct.c_char_p, ct.c_char_p,
                                           ct.c_double, ct.c_double, ct.c_char_p, ct.c_int]
        self.lib.rumi_sse_create.restype = ct.c_void_p
        self.lib.rumi_sse_destroy.argtypes = [ct.c_void_p]
        self.lib.rumi_sse_evolve.argtypes = [ct.c_void_p, ct.c_double, ct.c_double, PTR, ct.c_char_p, ct.c_int]

    def kepler(self, position, velocity, mu, dt):
        r, v = _array(position, (3,)).copy(), _array(velocity, (3,)).copy()
        if not math.isfinite(mu) or mu <= 0 or not math.isfinite(dt):
            raise ValueError("finite dt and positive mu required")
        if self.lib.rumi_kepler(_ptr(r), _ptr(v), mu, dt):
            raise RuntimeError("Kepler drift failed")
        return r, v

    def pn_pair(self, position, velocity, mass, *, G=1.0, c=299792.458, mask=1):
        x, v, m = _array(position,(2,3)), _array(velocity,(2,3)), _array(mass,(2,))
        a = np.empty((2,3))
        if self.lib.rumi_pn_pair(_ptr(x),_ptr(v),_ptr(m),G,c,mask,_ptr(a)):
            raise RuntimeError("PN pair evaluation failed")
        return a

    def recoil(self, mass, spins, angular_direction, phi=0.3, phase=0.7):
        m, s, L = _array(mass,(2,)), _array(spins,(2,3)), _array(angular_direction,(3,))
        out = np.empty(3)
        if self.lib.rumi_recoil(_ptr(m),_ptr(s),_ptr(L),phi,phase,_ptr(out)):
            raise RuntimeError("recoil failed")
        return out

    def secular_binary(self, position, velocity, mass, dt, *, G=1.0, c=299792.458, mask=0):
        x, v, m = _array(position,(2,3)).copy(), _array(velocity,(2,3)).copy(), _array(mass,(2,))
        steps = self.lib.rumi_secular_binary(_ptr(x),_ptr(v),_ptr(m),G,c,mask,dt)
        if steps < 0:
            raise RuntimeError("secular binary evolution failed")
        return x, v

    def pericentre(self, position, velocity, mu):
        r, v = _array(position,(3,)), _array(velocity,(3,))
        result = np.empty(2)
        if self.lib.rumi_pericentre(_ptr(r),_ptr(v),mu,_ptr(result)):
            raise RuntimeError("pericentre evaluation failed")
        return tuple(result)

    def elements(self,a,e,inclination,node,periapsis,mean_anomaly,mu):
        values=_array([a,e,inclination,node,periapsis,mean_anomaly],(6,))
        r,v=np.empty(3),np.empty(3)
        if self.lib.rumi_elements(_ptr(values),mu,_ptr(r),_ptr(v)):
            raise RuntimeError('orbital reconstruction failed')
        return r,v

    def bh_remnant(self, mass, spins, angular_momentum, *, G=1., c=299792.458):
        m, s, L = _array(mass,(2,)), _array(spins,(2,3)), _array(angular_momentum,(3,))
        result = np.empty(7)
        if self.lib.rumi_bh_remnant(_ptr(m),_ptr(s),_ptr(L),G,c,_ptr(result)):
            raise RuntimeError("BH remnant evaluation failed")
        return result[0], result[1:4], result[4:7]

    def secular_triple(self, position, velocity, mass, dt, *, G=1., c=299792.458, mask=0):
        x, v, m = _array(position,(3,3)).copy(), _array(velocity,(3,3)).copy(), _array(mass,(3,))
        radiation = np.empty(1)
        status = self.lib.rumi_secular_triple(_ptr(x),_ptr(v),_ptr(m),G,c,mask,dt,_ptr(radiation))
        if status == 2:
            return None
        if status:
            raise RuntimeError("secular triple evolution failed")
        return x, v, float(radiation[0])


@dataclass(frozen=True)
class EncounterOptions:
    G: float = 1.0
    c: float = 299792.458
    pn_mask: int = 0
    sdar: bool = False
    k_ref: float = 1e-6
    kappa_cap: float = 0.1
    tolerance: float = 1e-10
    max_steps: int = 100000
    deuflhard: bool = False
    max_stages: int = 10
    time_tolerance: float = 1e-12


@dataclass
class EncounterResult:
    position: np.ndarray
    velocity: np.ndarray
    time_reached: float
    steps: int
    slowdown_energy_jump: float
    kappa_peak: float
    collision: tuple | None


class Encounter:
    """Persistent normalization, PN auxiliary velocities and spins belong to one identified group."""

    def __init__(self, position, velocity, mass, *, options=None, spins=None, native=None):
        self.native = native or NativePhysics()
        self.options = options or EncounterOptions()
        m = _array(mass)
        n = len(m)
        if not 2 <= n <= 8 or np.any(m <= 0):
            raise ValueError("retained LogH supports 2..8 positive masses")
        x, v = _array(position,(n,3)), _array(velocity,(n,3))
        self.total = float(m.sum())
        self.radius_scale = float(np.max(np.linalg.norm(x-np.average(x,weights=m,axis=0),axis=1)))
        if self.radius_scale <= 0:
            raise ValueError("singular encounter")
        self.velocity_scale = np.sqrt(self.options.G*self.total/self.radius_scale)
        self.time_scale = self.radius_scale/self.velocity_scale
        self.energy_scale = self.total*self.velocity_scale**2
        self.mass = m.copy()
        self.aux = v.copy()
        self.last_velocity = v.copy()
        self.spins = np.zeros((2,3)) if spins is None else _array(spins,(2,3)).copy()
        self.aux_spins = self.spins.copy()

    def advance(self, position, velocity, dt, *, collision_distances=None):
        o, m = self.options, self.mass
        n = len(m)
        x, v = _array(position,(n,3)), _array(velocity,(n,3))
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("finite positive encounter dt required")
        center, speed = np.average(x,weights=m,axis=0), np.average(v,weights=m,axis=0)
        xn = np.ascontiguousarray((x-center)/self.radius_scale)
        vn = np.ascontiguousarray((v-speed)/self.velocity_scale)
        # External kicks affect both physical and auxiliary velocities equally.
        aux = np.ascontiguousarray((self.aux+v-self.last_velocity-speed)/self.velocity_scale)
        mn = np.ascontiguousarray(m/self.total)
        spin, aux_spin = self.spins.copy(), self.aux_spins.copy()
        opts = np.array([dt/self.time_scale,1,o.c/self.velocity_scale,o.tolerance,o.max_steps,
                         o.pn_mask,o.sdar,o.k_ref,o.kappa_cap,o.deuflhard,o.max_stages,o.time_tolerance],float)
        coll = np.zeros((8,8))
        if collision_distances is not None:
            coll[:n,:n] = _array(collision_distances,(n,n))/self.radius_scale
        report, message = np.empty(9), ct.create_string_buffer(256)
        status = self.native.lib.rumi_full_encounter(n,_ptr(xn),_ptr(vn),_ptr(mn),_ptr(aux),_ptr(spin),
                      _ptr(aux_spin),_ptr(opts),_ptr(coll),_ptr(report),message,len(message))
        if status not in (0,2):
            raise RuntimeError(message.value.decode())
        reached = report[0]*self.time_scale
        ox, ov = xn*self.radius_scale+center+reached*speed, vn*self.velocity_scale+speed
        self.aux = aux*self.velocity_scale+speed
        self.last_velocity, self.spins, self.aux_spins = ov.copy(), spin, aux_spin
        event = (int(report[4]),int(report[5]),report[6]*self.radius_scale) if status==2 else None
        return EncounterResult(ox,ov,reached,int(report[1]),report[2]*self.energy_scale,float(report[3]),event)

    def carry(self):
        return dict(mass=self.mass.tolist(), aux=self.aux.tolist(), last_velocity=self.last_velocity.tolist(),
                    spins=self.spins.tolist(),aux_spins=self.aux_spins.tolist(),total=self.total,
                    radius_scale=self.radius_scale,velocity_scale=self.velocity_scale,
                    time_scale=self.time_scale,energy_scale=self.energy_scale)

    @classmethod
    def restore(cls, carry, *, options=None, native=None):
        task = cls.__new__(cls)
        task.native = native or NativePhysics()
        task.options = options or EncounterOptions()
        for name in ('mass','aux','last_velocity','spins','aux_spins'):
            setattr(task,name,_array(carry[name]).copy())
        for name in ('total','radius_scale','velocity_scale','time_scale','energy_scale'):
            value = float(carry[name])
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid encounter carry: {name}")
            setattr(task,name,value)
        return task


class StellarEvolution:
    def __init__(self, *, metallicity=0.02, wind="HURLEY", remnant="BELCZYNSKI02", ppisn="NONE", ecsn="OFF",
                 max_ns_mass=-1.0, lbv_factor=1.5, native=None):
        self.native = native or NativePhysics()
        message = ct.create_string_buffer(256)
        self.context = self.native.lib.rumi_sse_create(metallicity,wind.encode(),remnant.encode(),ppisn.encode(),
                                  ecsn.encode(),max_ns_mass,lbv_factor,message,len(message))
        if not self.context:
            raise RuntimeError(message.value.decode())

    def evolve(self, zams_mass, age_myr):
        result, message = np.empty(7), ct.create_string_buffer(256)
        status = self.native.lib.rumi_sse_evolve(self.context,zams_mass,age_myr,_ptr(result),message,len(message))
        if status:
            raise RuntimeError(message.value.decode())
        return dict(type=int(result[0]),mass=result[1],core_mass=result[2],radius=result[3],luminosity=result[4],
                    fallback=result[5],ecsn=bool(result[6]))

    def close(self):
        if self.context:
            self.native.lib.rumi_sse_destroy(self.context)
            self.context = None

    def __enter__(self):
        return self

    def __exit__(self,*_):
        self.close()
