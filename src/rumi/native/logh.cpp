// Newtonian time-transformed leapfrog + polynomial GBS for independent 2..8 body CPU tasks.
// Mathematical conventions follow cpp-nbody 8b7729c's LogH/GBS implementation.
// This standalone pilot omits PN, slowdown, collision handling and cluster-global state.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

namespace {
constexpr int capacity = 8, dimension = 6 * capacity + 1;
using Vector = std::array<double, dimension>;

bool finite(const Vector& y, int n) {
    for (int d = 0; d < 6 * n + 1; ++d) if (!std::isfinite(y[d])) return false;
    return true;
}

double kinetic(const Vector& y, const double* m, int n) {
    double t = 0;
    for (int i = 0; i < n; ++i)
        for (int d = 0; d < 3; ++d) t += 0.5 * m[i] * y[3 * n + 3 * i + d] * y[3 * n + 3 * i + d];
    return t;
}

bool forces(const Vector& y, const double* m, int n, double G, Vector& a, double& potential) {
    a.fill(0);
    potential = 0;
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        double r[3], r2 = 0;
        for (int d = 0; d < 3; ++d) { r[d] = y[3*j+d] - y[3*i+d]; r2 += r[d]*r[d]; }
        if (!(r2 > 0) || !std::isfinite(r2)) return false;
        const double inv = 1 / std::sqrt(r2), f = G * inv * inv * inv;
        potential -= G * m[i] * m[j] * inv;
        for (int d = 0; d < 3; ++d) { a[3*i+d] += m[j]*f*r[d]; a[3*j+d] -= m[i]*f*r[d]; }
    }
    return std::isfinite(potential);
}

bool kick(Vector& y, const Vector& a, const double* m, int n, double binding, double ds) {
    const double o0 = kinetic(y, m, n) + binding;
    if (!(o0 > 0) || !std::isfinite(o0)) return false;
    double c = 0, d2 = 0;
    for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) {
        const double delta = ds * a[3*i+d];
        c += 0.5 * m[i] * y[3*n+3*i+d] * delta;
        d2 += 0.125 * m[i] * delta * delta;
    }
    double omega = o0;
    bool converged = false;
    for (int it = 0; it < 40; ++it) {
        const double f = omega*omega*omega - o0*omega*omega - c*omega - d2;
        const double fp = 3*omega*omega - 2*o0*omega - c;
        if (!std::isfinite(f) || !std::isfinite(fp) || fp == 0) return false;
        const double next = omega - f / fp;
        if (!(next > 0) || !std::isfinite(next)) return false;
        if (std::abs(next-omega) <= 8*std::numeric_limits<double>::epsilon()*omega) { omega = next; converged = true; break; }
        omega = next;
    }
    if (!converged) return false;
    for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) y[3*n+3*i+d] += ds*a[3*i+d]/omega;
    return finite(y, n);
}

bool midpoint(const Vector& initial, Vector& y, const double* m, int n, double G,
              double binding, double ds, int substeps) {
    y = initial;
    const double h = ds / substeps;
    Vector a;
    double potential;
    if (!forces(y, m, n, G, a, potential)) return false;
    for (int k = 0; k < substeps; ++k) {
        if (!kick(y, a, m, n, binding, h/2)) return false;
        const double omega = kinetic(y, m, n) + binding;
        if (!(omega > 0) || !std::isfinite(omega)) return false;
        const double dt = h / omega;
        for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) y[3*i+d] += dt*y[3*n+3*i+d];
        y[6*n] += dt;
        if (!forces(y, m, n, G, a, potential) || !kick(y, a, m, n, binding, h/2)) return false;
    }
    return finite(y, n);
}

bool step(const Vector& initial, Vector& result, const double* m, int n, double G,
          double binding, double ds, double tol, double& next) {
    constexpr int sequence[] = {2, 4, 6, 8, 12, 16, 24, 32};
    std::array<Vector, 8> previous{}, current{};
    Vector last;
    const int dim = 6*n+1;
    for (int k = 0; k < 8; ++k) {
        if (!midpoint(initial, current[0], m, n, G, binding, ds, sequence[k])) { next = ds/2; return false; }
        for (int j = 1; j <= k; ++j) {
            const double ratio = double(sequence[k]) / sequence[k-j];
            const double factor = 1 / (ratio*ratio - 1);
            for (int d = 0; d < dim; ++d)
                current[j][d] = current[j-1][d] + factor*(current[j-1][d] - previous[j-1][d]);
        }
        if (!finite(current[k], n)) { next = ds/2; return false; }
        if (k > 0) {
            double error = 0;
            for (int d = 0; d < dim; ++d) {
                const double e = std::abs(current[k][d]-last[d]) / std::max(1.0, std::abs(current[k][d]));
                if (!std::isfinite(e)) { next = ds/2; return false; }
                error = std::max(error, e);
            }
            if (error <= tol) {
                result = current[k];
                next = ds*std::clamp(0.8*std::pow(tol/std::max(error, 1e-300), 1.0/(2*(k+1))), 0.2, 4.0);
                return true;
            }
        }
        last = current[k];
        previous = current;
    }
    next = ds/2;
    return false;
}
}

extern "C" int rumi_logh(int n, const double* x, const double* v, const double* m,
                         double dt, double G, double tol, int max_steps,
                         double* out_x, double* out_v, int* steps, char* message, int message_size) {
    auto fail = [&](const char* why) { std::snprintf(message, message_size, "%s", why); return 1; };
    *steps = 0;
    if (n < 2 || n > capacity || !(dt > 0) || !std::isfinite(dt) || !(G > 0)
        || !std::isfinite(G) || !(tol > 0) || !std::isfinite(tol) || max_steps < 1)
        return fail("invalid LogH arguments");
    Vector y{};
    for (int i = 0; i < n; ++i) {
        if (!(m[i] > 0) || !std::isfinite(m[i])) return fail("invalid mass");
        for (int d = 0; d < 3; ++d) { y[3*i+d] = x[3*i+d]; y[3*n+3*i+d] = v[3*i+d]; }
    }
    if (!finite(y, n)) return fail("non-finite initial state");
    Vector a;
    double potential;
    if (!forces(y, m, n, G, a, potential)) return fail("singular initial configuration");
    const double binding = -kinetic(y, m, n) - potential;
    double ds = std::min(dt, 0.01) * (-potential);
    const double time_tol = 1e-12 * dt;
    for (int attempt = 0; attempt < max_steps; ++attempt) {
        *steps = attempt + 1;
        const double remaining = dt-y[6*n];
        if (std::abs(remaining) <= time_tol) {
            for (int i = 0; i < n; ++i) for (int d = 0; d < 3; ++d) { out_x[3*i+d] = y[3*i+d]; out_v[3*i+d] = y[3*n+3*i+d]; }
            return 0;
        }
        const double omega = kinetic(y, m, n) + binding;
        ds = std::min(ds, remaining*omega);
        if (!(ds > 0) || !std::isfinite(ds) || y[6*n] + ds/omega == y[6*n]) return fail("LogH time step stalled");
        Vector trial;
        double next = ds;
        if (!step(y, trial, m, n, G, binding, ds, tol, next)) { ds = next; continue; }
        const double advanced = trial[6*n]-y[6*n];
        if (!(advanced > 0) || !std::isfinite(advanced)) return fail("invalid physical time advance");
        if (trial[6*n] > dt+time_tol) { ds *= 0.99*remaining/advanced; continue; }
        y = trial;
        ds = next;
    }
    return fail("LogH did not reach target within max_steps");
}
