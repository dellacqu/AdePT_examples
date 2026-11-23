#!/usr/bin/env python3
"""Position scan across the middle-layer cells: FTFP_BERT, matched G4HepEm and AdePT.

Reads the output of runcards/validation/run_validation.sh with SCAN=1,
    <dir>/scan_eta<eta>_phi<phi>/<setup>/ATLLArBarrelout_Run0.root   (setup: cpu, matched, adept)
and computes, per point and setup, from the noise-free cells (needs the FrontHitsEdepNoNoise...
and EnergyCross* columns, added on 2026-11-23):
  response     mean EnergyLArNoNoise, relative to the matched mean over all points of the same scan
  fmax         fraction of the middle-layer energy in its hottest cell
  bias         energy-weighted barycentre of the 3x3 middle cells around the hottest one, minus the
               beam direction, in units of the cell (eta for the eta scan, phi for the phi scan)
  R_eta, R_phi the shower-shape variables of the middle layer around its hottest cell, as in
               analysis/v0.3: R_phi = E(3x3) / E(3x7), R_eta = E(3x7) / E(7x7) (eta x phi cells), noise-free
  cross-cell   fraction of EnergyLArNoNoise in steps whose post-step point is in another cell
  cross-sect.  the same for another section (front/middle/back)
Errors are standard errors of the mean. The position is given within the cell: 0 is a cell boundary,
0.5 the cell centre.

Usage:
    python3 scan.py <scan-dir> [--plot scan.png]

Needs uproot, numpy and (for --plot) matplotlib.
"""

import argparse
import glob
import math
import os
import re
import sys

import numpy as np
import uproot

SETUPS = [("cpu", "FTFP_BERT"), ("matched", "matched G4HepEm"), ("adept", "AdePT")]
D_ETA = 0.025  # middle cell
D_PHI = 2 * math.pi / 256
HALF_STAC_PHI = math.pi / 16
ETAS_PER_ROW = 32
N_MIDDLE = 512
STYLE = {"cpu": ("#2a78d6", "o"), "matched": ("#eb6834", "s"), "adept": ("#1baf7a", "^")}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
OBS = [("response", "response / matched mean", 4), ("fmax", "hottest middle cell fraction", 4),
       ("bias", "barycentre − true (cells)", 4), ("reta", "R_η = E(3×7) / E(7×7)", 4),
       ("rphi", "R_φ = E(3×3) / E(3×7)", 4), ("cross_cell", "energy in cell-crossing steps", 4),
       ("cross_section", "energy in section-crossing steps", 5)]


def per_event(path, eta_true, phi_true):
    t = uproot.open(path)["ATLLArBarrelout"]
    for col in ("MiddleHitsEdepNoNoise", "EnergyCrossCell"):
        if col not in t:
            sys.exit(f"{path}: no column {col}; this output is from before the noise-free cells were added")
    e = t["EnergyLArNoNoise"].array(library="np")
    mid = np.stack(t["MiddleHitsEdepNoNoise"].array(library="np")).reshape(-1, N_MIDDLE // ETAS_PER_ROW, ETAS_PER_ROW)
    n = len(e)
    flat = mid.reshape(n, -1)
    imax = np.argmax(flat, axis=1)
    pmax, emax = imax // ETAS_PER_ROW, imax % ETAS_PER_ROW
    msum = flat.sum(axis=1)
    fmax = flat[np.arange(n), imax] / msum
    # 3x3 barycentre and R_eta, R_phi around the hottest cell (the cells of this scan are interior)
    eta_b = np.empty(n)
    phi_b = np.empty(n)
    reta = np.empty(n)
    rphi = np.empty(n)

    def window(i, p0, e0, half_phi, half_eta):
        return mid[i, max(p0 - half_phi, 0):p0 + half_phi + 1, max(e0 - half_eta, 0):e0 + half_eta + 1].sum()

    for i in range(n):
        p0, e0 = pmax[i], emax[i]
        e33, e37, e77 = window(i, p0, e0, 1, 1), window(i, p0, e0, 3, 1), window(i, p0, e0, 3, 3)
        rphi[i] = e33 / e37
        reta[i] = e37 / e77
        win = mid[i, max(p0 - 1, 0):p0 + 2, max(e0 - 1, 0):e0 + 2]
        ps = np.arange(max(p0 - 1, 0), max(p0 - 1, 0) + win.shape[0])
        es = np.arange(max(e0 - 1, 0), max(e0 - 1, 0) + win.shape[1])
        w = win.sum()
        eta_b[i] = ((es + 0.5) * D_ETA * win.sum(axis=0)).sum() / w
        phi_b[i] = ((-HALF_STAC_PHI + (ps + 0.5) * D_PHI) * win.sum(axis=1)).sum() / w
    return {
        "e": e,
        "fmax": fmax,
        "bias_eta": (eta_b - eta_true) / D_ETA,
        "bias_phi": (phi_b - phi_true) / D_PHI,
        "reta": reta,
        "rphi": rphi,
        "cross_cell": t["EnergyCrossCell"].array(library="np") / e,
        "cross_section": t["EnergyCrossSection"].array(library="np") / e,
    }


def mean_sem(x):
    x = x[np.isfinite(x)]
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(len(x)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--plot")
    args = parser.parse_args()

    points = []
    for d in sorted(glob.glob(os.path.join(args.directory, "scan_eta*_phi*"))):
        m = re.search(r"scan_eta(\d+)p(\d+)_phi(\d+)p(\d+)$", d)
        if m:
            points.append((float(f"{m[1]}.{m[2]}"), float(f"{m[3]}.{m[4]}"), d))
    if not points:
        sys.exit("no scan_eta*_phi* directories found")
    setups = [(s, l) for s, l in SETUPS if all(os.path.isfile(os.path.join(d, s, "ATLLArBarrelout_Run0.root"))
                                                for _, _, d in points)]
    # eta scan: phi fixed at the cell centre; phi scan: eta fixed at 0.2125 and phi below the centre
    eta_scan = sorted([p for p in points if abs(p[1] - 0.01225) < 1e-4], key=lambda p: p[0])
    phi_scan = sorted([p for p in points if abs(p[0] - 0.2125) < 1e-6], key=lambda p: p[1])

    results = {}
    for name, scan, coord in (("eta", eta_scan, lambda p: (p[0] - 0.2) / D_ETA),
                              ("phi", phi_scan, lambda p: p[1] / D_PHI)):
        rows = []
        for p in scan:
            row = {"x": coord(p), "eta": p[0], "phi": p[1]}
            for s, _ in setups:
                ev = per_event(os.path.join(p[2], s, "ATLLArBarrelout_Run0.root"), p[0], p[1])
                row[s] = {"e": mean_sem(ev["e"]), "fmax": mean_sem(ev["fmax"]),
                          "bias": mean_sem(ev[f"bias_{name}"]), "reta": mean_sem(ev["reta"]),
                          "rphi": mean_sem(ev["rphi"]), "cross_cell": mean_sem(ev["cross_cell"]),
                          "cross_section": mean_sem(ev["cross_section"])}
            rows.append(row)
        ref = "matched" if any(s == "matched" for s, _ in setups) else setups[0][0]
        norm = np.mean([r[ref]["e"][0] for r in rows])
        for r in rows:
            for s, _ in setups:
                m, e = r[s]["e"]
                r[s]["response"] = (m / norm, e / norm)
        results[name] = rows

        print(f"## {name} scan (position within the middle cell: 0 = boundary, 0.5 = centre)\n")
        for key, label, dec in OBS:
            print(f"### {label}\n")
            head = "| position | eta | phi | " + " | ".join(l for _, l in setups)
            if ref == "matched" and any(s == "adept" for s, _ in setups):
                head += " | AdePT − matched"
            print(head + " |")
            print("|---" * (head.count("|")) + "|")
            for r in rows:
                cells = [f"{r[s][key][0]:.{dec}f} ± {r[s][key][1]:.{dec}f}" for s, _ in setups]
                if "| AdePT − matched" in head:
                    (a, da), (b, db) = r["adept"][key], r["matched"][key]
                    sig = (a - b) / math.hypot(da, db) if da or db else float("nan")
                    cells.append(f"{a - b:+.{dec}f} ({sig:+.1f}σ)")
                print(f"| {r['x']:.3f} | {r['eta']:.6f} | {r['phi']:.6f} | " + " | ".join(cells) + " |")
            print()

    if args.plot:
        plot(args.plot, results, setups)


def plot(path, results, setups):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    keys = [o for o in OBS if o[0] != "cross_section"]
    fig, axes = plt.subplots(len(keys), 2, figsize=(11, 2.3 * len(keys)), sharex="col", facecolor=SURFACE)
    offsets = dict(zip([s for s, _ in setups], np.linspace(-0.006, 0.006, len(setups))))
    for col, (name, title) in enumerate((("eta", "η scan (φ at cell centre)"), ("phi", "φ scan (η at cell centre)"))):
        rows = results[name]
        x = np.array([r["x"] for r in rows])
        for row_i, (key, label, _) in enumerate(keys):
            ax = axes[row_i, col]
            for s, l in setups:
                c, mk = STYLE[s]
                y = np.array([r[s][key][0] for r in rows])
                dy = np.array([r[s][key][1] for r in rows])
                ax.errorbar(x + offsets[s], y, yerr=dy, marker=mk, ms=6, color=c, mec=SURFACE, mew=1, lw=1.2,
                            elinewidth=1.5, label=l)
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if col == 0:
                ax.set_ylabel(label)
            if row_i == 0:
                ax.set_title(title, color=INK, fontsize=11, loc="left")
        axes[-1, col].set_xlabel("position in the middle cell (0 = boundary, 0.5 = centre)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="upper right", ncol=len(labels), labelcolor=INK2)
    fig.suptitle("ATLLArBarrel: 20 GeV e⁻ across a middle-layer cell, 1000 events per point", color=INK,
                 fontsize=12, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
