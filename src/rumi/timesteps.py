"""Subsystem perturbation and tidal-sampling constraints."""

import numpy as np

from .subsystems import elements


def ratio(value, rate):
    return abs(value/rate) if abs(rate) > 1e-30 else np.inf


def perturbation_step(position, velocity, mass, acceleration, G, eta):
    if eta <= 0:
        return np.inf
    x, v, m, f = map(np.asarray, (position, velocity, mass, acceleration))
    if len(m) == 2:
        r, u, force = x[1]-x[0], v[1]-v[0], f[1]-f[0]
        distance, mu = np.linalg.norm(r), G*m.sum()
        if distance <= 1e-30 or np.linalg.norm(force) <= 1e-30:
            return np.inf
        orbit = elements(r, u, mu)
        h = np.cross(r, u)
        rhat = r/distance
        if orbit['energy'] >= 0:
            eccentricity = np.cross(u, h)/mu-rhat
            derivative = (np.cross(force, h)+np.cross(u, np.cross(r, force)))/mu
            return eta*min(ratio(orbit['energy'], u@force),
                           ratio(np.linalg.norm(eccentricity), np.linalg.norm(derivative)))
        a, e = orbit['a'], orbit['e']
        p = h@h/mu
        sin_nu = (r@u/distance)*np.sqrt(p)/(e*np.sqrt(mu)) if e > 1e-12 else 0.
        cos_nu = (p/distance-1)/e if e > 1e-12 else 1.
        cos_E = (e+cos_nu)/(1+e*cos_nu)
        hn = np.linalg.norm(h)
        tangent = np.cross(h/max(hn, 1e-300), rhat)
        radial, tangential = force@rhat, force@tangent
        mean_motion, factor = np.sqrt(mu/a**3), np.sqrt(max(0., 1-e*e))
        if mean_motion <= 1e-30 or factor <= 1e-30:
            return np.inf
        adot = 2/(mean_motion*factor)*(e*sin_nu*radial+(1+e*cos_nu)*tangential)
        edot = factor/(mean_motion*a)*(sin_nu*radial+(cos_nu+cos_E)*tangential)
        return eta*min(ratio(a, adot), ratio(1., edot))
    x = x-np.average(x, weights=m, axis=0)
    v = v-np.average(v, weights=m, axis=0)
    f = f-np.average(f, weights=m, axis=0)
    energy = .5*np.sum(m[:, None]*v*v)
    for i in range(len(m)):
        for j in range(i+1, len(m)):
            energy -= G*m[i]*m[j]/np.linalg.norm(x[j]-x[i])
    derivative = np.sum(m[:, None]*v*f)
    h = np.sum(m[:, None]*np.cross(x, v), axis=0)
    hdot = np.sum(m[:, None]*np.cross(x, f), axis=0)
    hn = np.linalg.norm(h)
    rate = h@hdot/hn if hn > 1e-30 else 0.
    return eta*min(ratio(energy, derivative), ratio(hn, rate))


def sampling_constants(position, velocity, mass, G, tolerance, resolution):
    x, v, m = map(np.asarray, (position, velocity, mass))
    x = x-np.average(x, weights=m, axis=0)
    v = v-np.average(v, weights=m, axis=0)
    floor = 0.
    if len(m) == 2:
        r, u, mu = x[1]-x[0], v[1]-v[0], G*m.sum()
        orbit = elements(r, u, mu)
        radius, speed = np.linalg.norm(r), np.linalg.norm(u)
        if orbit['energy'] < 0:
            radius = orbit['a']*(1+orbit['e'])
            speed = np.sqrt(mu/orbit['a']*(1+orbit['e'])/max(1-orbit['e'], 1e-300))
            floor = resolution*orbit['period']*(1-orbit['e'])**1.5
        momentum = m.prod()/m.sum()*speed
    else:
        radius = 2*np.max(np.linalg.norm(x, axis=1))
        momentum = np.sum(m*np.linalg.norm(v, axis=1))
    coefficient = tolerance/(2*G*radius*momentum) if radius*momentum > 0 else np.inf
    return coefficient, floor
