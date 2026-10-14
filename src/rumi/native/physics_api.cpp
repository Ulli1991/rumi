// Standalone interfaces to retained cpp-nbody CPU mathematics; task state is caller-owned.
#include "retained/logh.h"
#include "retained/gbs.h"
#include "retained/kepler.h"
#include "retained/secular.h"
#include "retained/post_newtonian.h"
#include "retained/gw_recoil.h"
#include "retained/sse.h"
#include "retained/merger_physics.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>

namespace {
bool finite(const double* a, int n) { for (int i=0;i<n;++i) if (!std::isfinite(a[i])) return false; return true; }
int error(char* message, int length, const char* what) { std::snprintf(message,length,"%s",what); return 1; }
}

// options: dt, G, c, tolerance, max_steps, pn_mask, sdar, k_ref, kappa_cap_c,
// deuflhard, kmax, time_tolerance. coll: 8x8 pair distances in current length units.
extern "C" int rumi_full_encounter(int n, double* x, double* v, const double* mass,
                                    double* aux, double* spins, double* aux_spins,
                                    const double* options, const double* coll,
                                    double* report, char* message, int length) {
    try {
        if (n<2 || n>8 || !finite(x,3*n) || !finite(v,3*n) || !finite(mass,n) || !finite(options,12))
            return error(message,length,"invalid encounter state or options");
        const double dt=options[0], G=options[1], c=options[2], tolerance=options[3];
        if (!(dt>0 && G>0 && tolerance>0) || options[4]<1 || options[11]<=0)
            return error(message,length,"invalid positive encounter parameters");
        LogHState state{};
        state.n_bodies=n; state.c_code=c;
        const int pn=int(options[5]);
        state.use_aux_vel=pn>0;
        state.has_spins=n==2 && (pn & PN_SPIN);
        if (pn>0 && (!(c>0) || !finite(aux,3*n))) return error(message,length,"PN requires finite auxiliary velocities and positive c");
        for (int i=0;i<n;++i) if (!(mass[i]>0)) return error(message,length,"nonpositive encounter mass");
        std::copy(x,x+3*n,state.pos); std::copy(v,v+3*n,state.vel); std::copy(mass,mass+n,state.mass);
        std::copy(aux,aux+3*n,state.aux_vel);
        if (state.has_spins) {
            if (!finite(spins,6) || !finite(aux_spins,6)) return error(message,length,"invalid spin state");
            std::copy(spins,spins+3,state.spin1); std::copy(spins+3,spins+6,state.spin2);
            std::copy(aux_spins,aux_spins+3,state.aux_spin1); std::copy(aux_spins+3,aux_spins+6,state.aux_spin2);
        }
        state.B=-logh_total_energy(state,G);
        if (!std::isfinite(state.B)) return error(message,length,"singular encounter configuration");
        SdarParams sd;
        sd.k_ref=options[7]; sd.c=options[8]; sd.coll_pair=coll;
        const bool sd_on=options[6]>0;
        if (sd_on) logh_sd_update(state,G,sd);
        GBSIntegrator solver(tolerance);
        solver.deuflhard=options[9]>0; solver.max_stages=int(options[10]); solver.time_tolerance=options[11];
        solver.sdar=sd_on ? &sd : nullptr;
        if (coll) solver.coll=coll;
        const double omega=logh_omega(state);
        if (!(omega>0) || !std::isfinite(omega)) return error(message,length,"invalid encounter time transformation");
        const int steps=solver.integrate_to_time(state,dt,G,dt*omega,c,pn,int(options[4]));
        if (!finite(state.pos,3*n) || !finite(state.vel,3*n) || !finite(state.aux_vel,3*n) || !std::isfinite(state.t))
            return error(message,length,"non-finite encounter result");
        if (!solver.reached && solver.coll_i<0) return error(message,length,"encounter did not reach target time");
        if (sd_on) logh_sd_update(state,G,sd,true);
        std::copy(state.pos,state.pos+3*n,x); std::copy(state.vel,state.vel+3*n,v);
        std::copy(state.aux_vel,state.aux_vel+3*n,aux);
        if (state.has_spins) {
            std::copy(state.spin1,state.spin1+3,spins); std::copy(state.spin2,state.spin2+3,spins+3);
            std::copy(state.aux_spin1,state.aux_spin1+3,aux_spins); std::copy(state.aux_spin2,state.aux_spin2+3,aux_spins+3);
        }
        report[0]=state.t; report[1]=steps; report[2]=state.sd_jump; report[3]=state.sd_kappa_peak;
        report[4]=solver.coll_i; report[5]=solver.coll_j; report[6]=solver.coll_rsep;
        report[7]=state.B; report[8]=logh_total_energy(state,G);
        return solver.coll_i>=0 ? 2 : 0;
    } catch (const std::exception& e) { return error(message,length,e.what()); }
}

extern "C" int rumi_kepler(double* r, double* v, double mu, double dt) {
    if (!finite(r,3)||!finite(v,3)||!(mu>0)||!std::isfinite(mu)||!std::isfinite(dt)) return 1;
    kepler_advance(r,v,mu,dt);
    return finite(r,3)&&finite(v,3) ? 0 : 1;
}

extern "C" int rumi_pn_pair(const double* x, const double* v, const double* m,
                            double G, double c, int mask, double* a) {
    if (!finite(x,6)||!finite(v,6)||!finite(m,2)||!(G>0&&c>0)) return 1;
    eih_pn_acceleration_pair(x,v,m[0],x+3,v+3,m[1],G,c,a,a+3,mask);
    return finite(a,6)?0:1;
}

extern "C" int rumi_secular_binary(double* x, double* v, const double* m,
                                   double G, double c, int mask, double dt) {
    if (!finite(x,6)||!finite(v,6)||!finite(m,2)||!(G>0&&c>0)||!std::isfinite(dt)) return -1;
    const int steps=integrate_secular_binary(x,v,m[0],x+3,v+3,m[1],G,c,mask,dt);
    return finite(x,6)&&finite(v,6)?steps:-1;
}

extern "C" int rumi_recoil(const double* m, const double* spin, const double* L, double phi, double phase, double* v) {
    if (!finite(m,2)||!finite(spin,6)||!finite(L,3)) return 1;
    gw_recoil_kick(m[0],m[1],spin,spin+3,L,phi,phase,v);
    return finite(v,3)?0:1;
}

extern "C" int rumi_bh_remnant(const double* m, const double* spin, const double* L,
                                double G, double c, double* result) {
    if (!finite(m,2)||!finite(spin,6)||!finite(L,3)||!(m[0]>0&&m[1]>0)) return 1;
    bh_bh_remnant(m[0],m[1],spin,spin+3,L,result[0],result+1,result+4,G,c);
    return finite(result,7)?0:1;
}

extern "C" int rumi_pericentre(const double* r, const double* v, double mu, double* result) {
    if (!finite(r,3)||!finite(v,3)||!(mu>0)) return 1;
    result[0]=kepler_time_to_pericentre(r,v,mu,result+1);
    return !std::isnan(result[0])&&std::isfinite(result[1])?0:1;
}

extern "C" int rumi_elements(const double* elements, double mu, double* r, double* v) {
    if (!finite(elements,6)||!(mu>0)||!(elements[0]>0)||elements[1]<0||elements[1]>=1) return 1;
    elements_to_cartesian(elements[0],elements[1],elements[2],elements[3],elements[4],elements[5],mu,r,v);
    return finite(r,3)&&finite(v,3)?0:1;
}

extern "C" int rumi_secular_triple(double* x, double* v, const double* m,
                                   double G, double c, int mask, double dt, double* radiation) {
    if (!finite(x,9)||!finite(v,9)||!finite(m,3)||!(G>0&&c>0)||!std::isfinite(dt)) return 1;
    const double M12=m[0]+m[1], M=M12+m[2];
    double r[3],u[3],R[3],U[3],center[3],speed[3],hi[3],ho[3];
    for (int d=0;d<3;++d) {
        r[d]=x[3+d]-x[d]; u[d]=v[3+d]-v[d];
        R[d]=x[6+d]-(m[0]*x[d]+m[1]*x[3+d])/M12;
        U[d]=v[6+d]-(m[0]*v[d]+m[1]*v[3+d])/M12;
        center[d]=(m[0]*x[d]+m[1]*x[3+d]+m[2]*x[6+d])/M;
        speed[d]=(m[0]*v[d]+m[1]*v[3+d]+m[2]*v[6+d])/M;
    }
    // Angular momenta require all components of the relative vectors above.
    for (int d=0;d<3;++d) {
        const int e=(d+1)%3,f=(d+2)%3;
        hi[d]=r[e]*u[f]-r[f]*u[e]; ho[d]=R[e]*U[f]-R[f]*U[e];
    }
    const auto inner=cartesian_to_elements(r,u,G*M12), outer=cartesian_to_elements(R,U,G*M);
    if (!(inner.a>0&&outer.a>0)) return 2;
    double hn=0,on=0,dot=0;
    for (int d=0;d<3;++d) { hn+=hi[d]*hi[d]; on+=ho[d]*ho[d]; dot+=hi[d]*ho[d]; }
    hn=std::sqrt(hn); on=std::sqrt(on);
    if (!(hn>0&&on>0)) return 2;
    SecularTriple tri{};
    tri.inner.a=inner.a; tri.inner.e=inner.e; tri.inner.inc=inner.inc;
    tri.inner.Omega=inner.Omega; tri.inner.omega=inner.omega; tri.inner.m1=m[0]; tri.inner.m2=m[1];
    tri.a_out=outer.a; tri.e_out=outer.e; tri.inc_out=outer.inc; tri.Omega_out=outer.Omega;
    tri.omega_out=outer.omega; tri.m3=m[2]; tri.cos_i_mut=dot/(hn*on);
    const double old_cos=tri.cos_i_mut;
    secular_evolve_triple(tri,dt,G,c,mask);
    if (!(tri.inner.a>0&&tri.a_out>0&&tri.inner.e>=0&&tri.inner.e<1)) return 1;
    double axis[3],axis_norm=0;
    for (int d=0;d<3;++d) {
        hi[d]/=hn; ho[d]/=on;
    }
    for (int d=0;d<3;++d) {
        const int e=(d+1)%3,f=(d+2)%3;
        axis[d]=ho[e]*hi[f]-ho[f]*hi[e]; axis_norm+=axis[d]*axis[d];
    }
    axis_norm=std::sqrt(axis_norm);
    if (axis_norm>1e-30) {
        const double angle=std::acos(std::clamp(tri.cos_i_mut,-1.0,1.0))-std::acos(std::clamp(old_cos,-1.0,1.0));
        for (int d=0;d<3;++d) axis[d]/=axis_norm;
        double rotated[3],projection=0;
        for (int d=0;d<3;++d) projection+=axis[d]*hi[d];
        for (int d=0;d<3;++d) {
            const int e=(d+1)%3,f=(d+2)%3;
            rotated[d]=hi[d]*std::cos(angle)+(axis[e]*hi[f]-axis[f]*hi[e])*std::sin(angle)
                      +axis[d]*projection*(1-std::cos(angle));
        }
        tri.inner.inc=std::acos(std::clamp(rotated[2],-1.0,1.0));
        tri.inner.Omega=std::atan2(rotated[0],-rotated[1]);
    }
    const double ni=std::sqrt(G*M12/std::pow(tri.inner.a,3)), no=std::sqrt(G*M/std::pow(tri.a_out,3));
    elements_to_cartesian(tri.inner.a,tri.inner.e,tri.inner.inc,tri.inner.Omega,tri.inner.omega,
                          inner.mean_anom+ni*dt,G*M12,r,u);
    elements_to_cartesian(tri.a_out,tri.e_out,tri.inc_out,tri.Omega_out,tri.omega_out,
                          outer.mean_anom+no*dt,G*M,R,U);
    for (int d=0;d<3;++d) {
        const double xc=center[d]+speed[d]*dt-m[2]/M*R[d], vc=speed[d]-m[2]/M*U[d];
        x[d]=xc-m[1]/M12*r[d]; x[3+d]=xc+m[0]/M12*r[d]; x[6+d]=center[d]+speed[d]*dt+M12/M*R[d];
        v[d]=vc-m[1]/M12*u[d]; v[3+d]=vc+m[0]/M12*u[d]; v[6+d]=speed[d]+M12/M*U[d];
    }
    *radiation=G*m[0]*m[1]/2*(1/tri.inner.a-1/inner.a);
    return finite(x,9)&&finite(v,9)&&std::isfinite(*radiation)?0:1;
}

extern "C" void* rumi_sse_create(double z, const char* wind, const char* remnant,
                                  const char* ppisn, const char* ecsn, double mxns, double flbv,
                                  char* message, int length) {
    try { return new SSEContext(z,make_sse_config(wind,remnant,ppisn,ecsn,mxns,flbv)); }
    catch (const std::exception& e) { error(message,length,e.what()); return nullptr; }
}
extern "C" void rumi_sse_destroy(void* context) { delete static_cast<SSEContext*>(context); }
extern "C" int rumi_sse_evolve(void* context, double mass, double age, double* result, char* message, int length) {
    if (!context||!(mass>0)||!std::isfinite(mass)||age<0||!std::isfinite(age)) return error(message,length,"invalid SSE arguments");
    try {
        const auto r=static_cast<SSEContext*>(context)->evolve(mass,age);
        result[0]=r.kw; result[1]=r.mass; result[2]=r.mass_core; result[3]=r.radius;
        result[4]=r.luminosity; result[5]=r.fbfac; result[6]=r.ecsn?1:0;
        return finite(result,7)?0:error(message,length,"non-finite SSE result");
    } catch (const std::exception& e) { return error(message,length,e.what()); }
}
