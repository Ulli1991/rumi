"""Reproducibly retain CPU numerical routines and inventory every upstream source change."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DIRECT = {
    "src/collisional/kepler.cpp": "kepler.cpp", "src/collisional/kepler.h": "kepler.h",
    "src/collisional/logh.cpp": "logh.cpp", "src/collisional/logh.h": "logh.h",
    "src/collisional/gw_recoil.h": "gw_recoil.h",
    "src/ic_gen/sse.cpp": "sse.cpp", "src/ic_gen/sse.h": "sse.h",
}


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def replace_function(text, marker, replacement):
    start = text.index(marker)
    masked = re.sub(r'//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'',
                    lambda m: " " * len(m[0]), text, flags=re.S)
    brace = masked.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (masked[end] == "{") - (masked[end] == "}")
        end += 1
    return text[:start] + replacement + text[end:]


def exports(source):
    result = {name: (source/path).read_text() for path, name in DIRECT.items()}
    pn = (source/"src/collisional/post_newtonian.cpp").read_text()
    pn = replace_function(pn, "void global_pn1_kick(", "")
    pn = pn.replace('#include "diagnostics.h"\n', "").replace("#include <Kokkos_Core.hpp>\n", "")
    result["post_newtonian.cpp"] = pn
    pnh = (source/"src/collisional/post_newtonian.h").read_text().replace('#include "particles.h"', "")
    pnh = re.sub(r"void global_pn1_kick\([^;]+;", "", pnh, flags=re.S)
    result["post_newtonian.h"] = pnh
    constants = (source/"src/core/constants.h").read_text().split("// Portable reciprocal square root:")[0]
    result["constants.h"] = constants.replace("#include <Kokkos_Core.hpp>\n", "")
    secular = (source/"src/collisional/secular.cpp").read_text()
    result["secular.cpp"] = replace_function(secular, "bool should_be_secular(", "")
    header = (source/"src/collisional/secular.h").read_text().replace('#include "subsystem.h"', '#include "kepler.h"')
    result["secular.h"] = re.sub(r"bool should_be_secular\([^;]+;", "", header, flags=re.S)
    gbs = (source/"src/collisional/gbs.cpp").read_text()
    gbs = gbs.replace('#include "collisional_config.h"\n', "").replace("#include <Kokkos_Core.hpp>\n", "")
    gbs = gbs.replace("GBSIntegrator::seq()", "GBSIntegrator::seq() const").replace("GBSIntegrator::kmax()", "GBSIntegrator::kmax() const")
    gbs = gbs.replace("g_collisional.gbs_deuflhard", "deuflhard").replace("g_collisional.gbs_kmax", "max_stages")
    gbs = replace_function(gbs, "void GBSIntegrator::compute_stages(", '''void GBSIntegrator::compute_stages(const LogHState& state, int k0, int k1, double ds, double G,
                                   double c_code, int pn_order, const LogHStepStart* start) {
    for (int k = k1; k >= k0; --k) {
        LogHState result;
        midpoint(state, result, seq()[k], ds, G, c_code, pn_order, start);
        int dim = 0;
        pack_state(result, table[k], dim);
    }
}''')
    # The retained native solver rejects invalid arithmetic instead of treating NaN as a small error.
    gbs = gbs.replace("if (err > max_err) max_err = err;", "if (!std::isfinite(err)) return std::numeric_limits<double>::infinity();\n                if (err > max_err) max_err = err;")
    gbs = re.sub(r"static const double time_tol = [^;]+;", "const double time_tol = time_tolerance;", gbs)
    result["gbs.cpp"] = gbs
    header = (source/"src/collisional/gbs.h").read_text()
    header = header.replace("static const int* seq();", "const int* seq() const;").replace("static int kmax();", "int kmax() const;")
    header = header.replace("double eta_gbs;", "bool deuflhard = false;\n    int max_stages = 10;\n    double time_tolerance = 1e-12;\n    double eta_gbs;")
    result["gbs.h"] = header
    mergers = (source/"src/collisional/mergers.cpp").read_text()
    functions = []
    for marker in ("double gw_coalescence_time(", "double tidal_disruption_radius(", "void bh_bh_remnant("):
        removed = replace_function(mergers, marker, "")
        start = mergers.index(marker)
        functions.append(mergers[start:start+len(mergers)-len(removed)])
    result["merger_physics.cpp"] = ('#include "gw_recoil.h"\n#include <cmath>\n#include <algorithm>\n'
                                    '#include <cstdint>\n#include <cstring>\n' + '\n\n'.join(functions)+'\n')
    result["merger_physics.h"] = '''#pragma once
double gw_coalescence_time(double a, double e, double m1, double m2, double G, double c);
double tidal_disruption_radius(double mass, double radius, double bh_mass);
void bh_bh_remnant(double m1, double m2, const double s1[3], const double s2[3],
                   const double L[3], double& mass, double spin[3], double kick[3], double G, double c);
'''
    return result


def parameter_table(source):
    text = (source/"src/core/params.cpp").read_text()
    pattern = r'\{"([A-Z][A-Z0-9_]*)",\s*([BIDS]),\s*"([^"\\]*(?:\\.[^"\\]*)*)",\s*"([^"\\]*)",\s*([A-Z]+),\s*"([^"\\]*)",\s*((?:"(?:\\.|[^"\\])*"\s*)+)\}'
    specs = {}
    for m in re.finditer(pattern, text):
        key, kind, default, category, level, choices, description = m.groups()
        description = "".join(json.loads(s) for s in re.findall(r'"(?:\\.|[^"\\])*"', description))
        specs[key] = dict(type=kind, default=default, category=category, level=level, choices=choices, description=description)
    if len(specs) < 100:
        raise RuntimeError(f"parameter extraction found only {len(specs)} entries; upstream format changed")
    return specs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT.parent/"cpp-nbody")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--apply", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    sha = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    sources = {str(p.relative_to(source)): digest(p.read_text()) for p in sorted((source/"src").rglob("*"))
               if p.is_file() and p.suffix in (".cpp", ".h")}
    generated = exports(source)
    native = ROOT/"src/rumi/native/retained"
    params = ROOT/"src/rumi/upstream_params.json"
    manifest_path = ROOT/"docs/upstream_manifest.json"
    manifest = dict(revision=sha, source_files=sources, retained_files={k: digest(v) for k, v in generated.items()})
    if args.check:
        if not manifest_path.exists():
            raise SystemExit("no upstream manifest; run --apply")
        old = json.loads(manifest_path.read_text())
        changed = [p for p in sorted(set(sources)|set(old["source_files"])) if sources.get(p) != old["source_files"].get(p)]
        altered = [p for p, content in generated.items() if not (native/p).exists() or (native/p).read_text() != content]
        if changed or altered:
            print(json.dumps(dict(upstream_changed=changed, retained_needing_update=altered), indent=2))
            raise SystemExit(1)
        print(f"upstream {sha[:7]}: {len(sources)} sources and {len(generated)} retained files match")
        return
    native.mkdir(parents=True, exist_ok=True)
    for name, content in generated.items():
        (native/name).write_text(content)
    params.write_text(json.dumps(parameter_table(source), indent=2)+"\n")
    manifest_path.write_text(json.dumps(manifest, indent=2)+"\n")
    print(f"retained {len(generated)} CPU source/header files and parameter table from {sha[:7]}")


if __name__ == "__main__":
    main()
