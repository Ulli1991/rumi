#include "gw_recoil.h"
#include <cmath>
#include <algorithm>
#include <cstdint>
#include <cstring>
double gw_coalescence_time(double a, double e, double m1, double m2,
                           double G, double c_code) {
    double c5 = std::pow(c_code, 5);
    double beta = 64.0/5.0 * G*G*G * m1 * m2 * (m1 + m2) / c5;

    if (e < 0.01) {
        // Circular approximation
        return a * a * a * a / (4.0 * beta);
    }

    // Full Peters integral approximation (Peters 1964 Eq. 5.14)
    double e04 = std::pow(e, 48.0/19.0);
    double c0 = a * (1.0 - e*e) / std::pow(e, 12.0/19.0) *
                std::pow(1.0 + 121.0/304.0 * e*e, -870.0/2299.0);
    // Approximate T_merge ~ (12/19) * c0^4 / (4*beta) * integral
    // Using the fit from Peters & Mathews:
    return a * a * a * a / (4.0 * beta) * std::pow(1.0 - e*e, 3.5) /
           (1.0 + 73.0/24.0 * e*e + 37.0/96.0 * e*e*e*e);
}

double tidal_disruption_radius(double m_star, double R_star, double M_bh) {
    // r_t = 1.3 (M_bh / m_star)^{1/3} R_star: FROST-CLUSTERS I (Rantala et al. 2024) App. B3.3 and FROST-CLUSTERS
    // III sec. 2.1.4, the cube-root scaling of Kochanek (1992). (Was 1.3 ((m + M) / m)^{1/3} R, BIFROST 2023.)
    // R_star in the same units as positions (pc)
    return 1.3 * std::cbrt(M_bh / m_star) * R_star;
}

void bh_bh_remnant(double m1, double m2, const double chi1[3], const double chi2[3],
                   const double L_orb[3], double& m_rem, double chi_rem[3],
                   double v_kick[3], double G, double c_code) {
    (void)G;
    (void)c_code;
    double M = m1 + m2;
    double eta = m1 * m2 / (M * M);
    double q = std::min(m1, m2) / std::max(m1, m2);
    double delta_m = std::fabs(m1 - m2) / M;

    // Radiated energy fraction (Barausse, Morozova & Rezzolla 2012, ApJ 758, 63)
    double E_rad_frac = 0.0547 * eta * (1.0 - 0.5 * delta_m * delta_m);
    m_rem = M * (1.0 - E_rad_frac);

    // Remnant spin (Barausse & Rezzolla 2009 simplified)
    // chi_f ~ eta * (3.464 - 3.162*eta - 2.484*eta^2)
    double chi_f_mag = eta * (3.464 - 3.162*eta - 2.484*eta*eta);
    chi_f_mag = std::min(chi_f_mag, 0.998);  // Kerr limit

    // Direction: along orbital angular momentum
    double L_mag = std::sqrt(L_orb[0]*L_orb[0] + L_orb[1]*L_orb[1] + L_orb[2]*L_orb[2]);
    if (L_mag > 1e-30) {
        chi_rem[0] = chi_f_mag * L_orb[0] / L_mag;
        chi_rem[1] = chi_f_mag * L_orb[1] / L_mag;
        chi_rem[2] = chi_f_mag * L_orb[2] / L_mag;
    } else {
        chi_rem[0] = chi_rem[1] = 0.0;
        chi_rem[2] = chi_f_mag;
    }

    // GW kick with spins (gw_recoil.h; Lousto et al. 2012). The phase angle and the in-plane orientation, unknown
    // from the orbit-averaged state, come from a hash of the masses, spins and L: deterministic on every MPI rank
    // and across restarts. (Before 2026-10-12: spins ignored, and the mass term used 1 + B delta_m instead of
    // 1 + B eta: 138 instead of 157 km/s at q = 0.5)
    std::uint64_t hsh = 1469598103934665603ULL;
    auto mix = [&](double x) { std::uint64_t u; std::memcpy(&u, &x, 8); hsh = (hsh ^ u) * 1099511628211ULL; };
    mix(m1); mix(m2);
    for (int k = 0; k < 3; k++) { mix(chi1[k]); mix(chi2[k]); mix(L_orb[k]); }
    const double u1 = (double)(hsh >> 11) * 0x1.0p-53;
    hsh = (hsh ^ 0x9e3779b97f4a7c15ULL) * 1099511628211ULL;
    const double u2 = (double)(hsh >> 11) * 0x1.0p-53;
    gw_recoil_kick(m1, m2, chi1, chi2, L_orb, u1, u2, v_kick);
}
