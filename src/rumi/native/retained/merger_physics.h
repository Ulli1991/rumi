#pragma once
double gw_coalescence_time(double a, double e, double m1, double m2, double G, double c);
double tidal_disruption_radius(double mass, double radius, double bh_mass);
void bh_bh_remnant(double m1, double m2, const double s1[3], const double s2[3],
                   const double L[3], double& mass, double spin[3], double kick[3], double G, double c);
