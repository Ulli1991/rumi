#pragma once

#include <cmath>

// Code units: pc, M_sun, km/s  (Gadget-style: time unit = 1 pc / 1 km/s)
// Length: 1 pc, Mass: 1 M_sun, Velocity: 1 km/s, Time: 1 pc/(1 km/s) ≈ 0.978 Myr
constexpr double G_CODE = 4.302194e-3;               // G in pc³ M_sun⁻¹ TU⁻²
constexpr double TIME_UNIT_MYR = 0.977792;           // 1 code TU in Myr (for config/display)
constexpr double TIME_UNIT_YR  = 977792.0;           // 1 code TU in years

// Conversion factors
constexpr double AU_TO_PC  = 4.84814e-6;             // 1 AU in pc (for outer planets init)
constexpr double PC_TO_CM  = 3.08567758e18;          // 1 parsec in cm (for SSE interface)
constexpr double MSUN_IN_G = 1.989e33;               // 1 M_sun in grams (for SSE interface)
constexpr double G_CGS     = 6.67430e-8;             // G in cm³ g⁻¹ s⁻² (for SSE interface)

// Numerical stability
constexpr double EPS = 1e-30;

// FSI coefficients
constexpr double FSI_KICK = 1.0 / 6.0;
constexpr double FSI_DRIFT = 0.5;
constexpr double FSI_MOD_KICK = 2.0 / 3.0;

// Gradient correction coefficient: G * dt² / 24 (from Eq. 29 of Rantala+2020: factor 2 in g_i × 1/48 = 1/24)
constexpr double GRAD_COEFF = 1.0 / 24.0;

