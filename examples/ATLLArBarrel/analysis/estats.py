#!/usr/bin/env python3
"""Energy statistics of ATLLArBarrel ROOT outputs, and differences between setups.

Computes the quantities of the comparison tables in Results.md (section 1a, appendix A3e) from the
ATLLArBarrelout ntuple: means with standard errors of the mean, rms, minimum, the
low-energy tail and the core mean, for the noise-free and noisy LAr energies, the three
compartments and the front fraction. Then it prints the differences between setups, in
MeV, percent and standard deviations.

Usage:
    python3 estats.py LABEL=FILE[,FILE...] [LABEL=FILE ...] [--diff A-B ...] [--tail 0.8]

    LABEL=FILE   one setup; several files of the same setup (repetitions) are separated by
                 commas and merged. A bare FILE uses the file name as label.
    --diff A-B   difference to print, setup A minus setup B (repeatable). Default: every
                 setup minus the first one.
    --tail X     events with EnergyLArNoNoise below X times the median are counted as the
                 low-energy tail and excluded from the core mean (default 0.8).

Example (section 3e, 50 GeV electrons, 10000 events, 8 threads):
    python3 estats.py FTFP_BERT=cpu/ATLLArBarrelout_Run0.root \\
                      matched=hepemem/ATLLArBarrelout_Run0.root \\
                      AdePT=adept/ATLLArBarrelout_Run0.root \\
                      --diff AdePT-matched --diff AdePT-FTFP_BERT

Needs uproot and numpy (pip install uproot numpy); ROOT itself is not needed.

Notes:
- Errors are standard errors of the mean, assuming independent events. The rms of the
  noise-free energy is printed without an uncertainty: the photonuclear low-energy tail
  makes a Gaussian error on the width misleading.
- The cell vectors (Front/Middle/BackHitsEdep) and EnergyLAr include the electronic noise
  (zero mean, ~169 MeV rms on the LAr sum). Only EnergyLArNoNoise and EnergyLArNoBirks are
  noise-free. The front fraction is computed per event from the noisy compartments, so it is
  not identical to a noise-free longitudinal fraction; events whose noisy total is <= 0
  (possible at low beam energy) are skipped for it and counted.
- This script was rewritten on 2026-11-23: the original estats.py of the review was not kept.
  It reproduces the definitions stated in Results.md, not necessarily every rounding of it.
"""

import argparse
import math
import os
import sys

import numpy as np
import uproot

TREE = "ATLLArBarrelout"
SCALARS = ["EnergyLArNoNoise", "EnergyLAr", "EnergyLArNoBirks"]
VECTORS = {"Front": "FrontHitsEdep", "Middle": "MiddleHitsEdep", "Back": "BackHitsEdep"}
# observable -> (decimals, show relative difference in percent)
FORMAT = {
    "EnergyLArNoNoise": (1, True),
    "EnergyLAr": (1, True),
    "EnergyLArNoBirks": (1, True),
    "Front": (1, False),
    "Middle": (1, False),
    "Back": (1, False),
    "Front fraction": (4, False),
}


def load(paths):
    """Read and concatenate the needed branches of one setup."""
    columns = {name: [] for name in SCALARS + list(VECTORS) + ["HadInteraction", "EventID"]}
    for path in paths:
        with uproot.open(path) as f:
            if TREE not in f:
                sys.exit(f"{path}: no '{TREE}' tree")
            tree = f[TREE]
            for name in SCALARS + ["HadInteraction", "EventID"]:
                if name not in tree:
                    sys.exit(f"{path}: branch '{name}' missing (output from before the noise-free columns?)")
                columns[name].append(tree[name].array(library="np"))
            for short, branch in VECTORS.items():
                cells = tree[branch].array(library="np")
                columns[short].append(np.array([np.sum(c) for c in cells], dtype=float))
    data = {k: np.concatenate(v) for k, v in columns.items()}
    total = data["Front"] + data["Middle"] + data["Back"]
    ok = total > 0
    data["Front fraction"] = np.where(ok, data["Front"] / np.where(ok, total, 1.0), np.nan)
    data["_skipped_fraction"] = int(np.count_nonzero(~ok))
    return data


def mean_sem(x):
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (float(x[0]) if n else math.nan), math.nan
    return float(np.mean(x)), float(np.std(x, ddof=1) / math.sqrt(n))


def fmt(value, error, decimals):
    if math.isnan(error):
        return f"{value:.{decimals}f}"
    return f"{value:.{decimals}f} ± {error:.{decimals}f}"


def fmt_diff(a, b, decimals, relative):
    (ma, ea), (mb, eb) = a, b
    d = ma - mb
    e = math.hypot(ea, eb)
    parts = []
    if relative and mb != 0:
        parts.append(f"{100 * d / mb:+.2f} %")
    if e > 0 and not math.isnan(e):
        parts.append(f"{d / e:+.1f} sigma")
    extra = f" ({', '.join(parts)})" if parts else ""
    return f"{d:+.{decimals}f} ± {e:.{decimals}f}{extra}"


def parse_setups(specs):
    setups = {}
    for spec in specs:
        label, sep, files = spec.partition("=")
        if not sep:
            label, files = os.path.basename(spec), spec
        paths = [p for p in files.split(",") if p]
        if label in setups:
            sys.exit(f"setup label '{label}' given twice; merge files with commas instead")
        for p in paths:
            if not os.path.isfile(p):
                sys.exit(f"{p}: file not found")
        setups[label] = paths
    return setups


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("setups", nargs="+", metavar="LABEL=FILE[,FILE...]")
    parser.add_argument("--diff", action="append", default=[], metavar="A-B")
    parser.add_argument("--tail", type=float, default=0.8)
    args = parser.parse_args()

    setups = parse_setups(args.setups)
    labels = list(setups)
    data = {label: load(paths) for label, paths in setups.items()}

    diffs = []
    for spec in args.diff:
        a, sep, b = spec.partition("-")
        if not sep or a not in data or b not in data:
            sys.exit(f"--diff {spec}: expected A-B with A and B among {labels}")
        diffs.append((a, b))
    if not args.diff:
        diffs = [(label, labels[0]) for label in labels[1:]]

    observables = list(FORMAT)
    stats = {label: {o: mean_sem(d[o]) for o in observables} for label, d in data.items()}

    # Checks: events, unique EventIDs, hadronic interactions
    print("## Samples\n")
    print("| Setup | Files | Events | Unique EventIDs | HadInteraction > 0 | Front fraction skipped |")
    print("|---|---|---|---|---|---|")
    for label in labels:
        d = data[label]
        n = len(d["EnergyLAr"])
        n_files = len(setups[label])
        unique = len(np.unique(d["EventID"])) if n_files == 1 else "n/a (several files)"
        print(f"| {label} | {n_files} | {n} | {unique} | {np.count_nonzero(d['HadInteraction'] > 0)} "
              f"| {d['_skipped_fraction']} |")
    print()

    header = "| Observable | " + " | ".join(labels) + "".join(f" | {a} − {b}" for a, b in diffs) + " |"
    print("## Means (MeV; errors are standard errors of the mean)\n")
    print(header)
    print("|---" * (1 + len(labels) + len(diffs)) + "|")
    for o in observables:
        decimals, relative = FORMAT[o]
        row = [fmt(*stats[label][o], decimals) for label in labels]
        row += [fmt_diff(stats[a][o], stats[b][o], decimals, relative) for a, b in diffs]
        print(f"| {o} | " + " | ".join(row) + " |")
    print()

    # Width, minimum and low-energy tail of the noise-free energy
    print(f"## EnergyLArNoNoise: width and tail (tail: below {args.tail:g} × median)\n")
    print("| Setup | rms | min | median | tail events | core mean |")
    print("|---|---|---|---|---|---|")
    core = {}
    for label in labels:
        e = data[label]["EnergyLArNoNoise"]
        median = float(np.median(e))
        in_core = e >= args.tail * median
        core[label] = mean_sem(e[in_core])
        rms = float(np.std(e, ddof=1)) if len(e) > 1 else math.nan
        print(f"| {label} | {rms:.1f} | {np.min(e):.1f} | {median:.1f} | {np.count_nonzero(~in_core)} "
              f"| {fmt(*core[label], 1)} |")
    if diffs:
        print()
        for a, b in diffs:
            print(f"- core mean {a} − {b}: {fmt_diff(core[a], core[b], 1, True)}")


if __name__ == "__main__":
    main()
