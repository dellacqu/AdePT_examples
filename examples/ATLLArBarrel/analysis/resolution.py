#!/usr/bin/env python3
"""LAr energy resolution curve of ATLLArBarrel, per setup, from a validation run.

Reads the output tree written by runcards/validation/run_validation.sh,
    <dir>/e<E>GeV/<setup>/ATLLArBarrelout_Run0.root   (setup: cpu, matched, adept)
and for every beam energy and setup computes the relative resolution sigma/mu of
  - EnergyLArNoNoise (Birks-corrected LAr energy, no electronic noise), and
  - EnergyLAr        (the same plus the electronic noise of all 1792 cells).
sigma/mu divides out the sampling fraction, so no calibration is needed.

Two estimators of sigma/mu are given:
  core  iterative Gaussian core: mean and standard deviation of the events within
        mu +- 2 sigma, iterated, with sigma corrected for the truncation of a Gaussian
        at +-2 sigma (factor 0.8796). Insensitive to the rare low-energy tail
        (photonuclear interactions, Results.md).
  rms   rms/mean of all events, which includes the tail.
Errors are bootstrap standard deviations (events resampled, 300 times).

The core resolution is fitted, as a weighted linear fit in (sigma/mu)^2, with
    (sigma/mu)^2 = a^2/E + b^2               (noise-free; b^2 may come out negative when b ~ 0)
    (sigma/mu)^2 = a^2/E + b^2 + c^2/E^2     (with noise; a and b fixed from the noise-free fit)
(E in GeV). The noise only adds in quadrature, so fixing a and b leaves c as a clean check.
Expected noise term: 4 MeV x sqrt(1792) = 169 MeV in the LAr, i.e. c ~ 0.169 GeV / (mu/E),
printed next to the fitted c as a check.

Usage:
    python3 resolution.py <validation-dir> [--energies 5 10 20 40 100] [--plot resolution.png]

Needs uproot, numpy and (for --plot) matplotlib.

Scope: this is the resolution of the sum of all LAr cells for a fixed beam direction,
with the simplified detector response of the example (constant field in Birks, independent
Gaussian noise on all cells, no clustering or calibration). It is a CPU/GPU comparison of
the simulated response, not the resolution of a reconstructed ATLAS electron.
"""

import argparse
import math
import os
import sys

import numpy as np
import uproot

SETUPS = [("cpu", "FTFP_BERT"), ("matched", "matched G4HepEm"), ("adept", "AdePT")]
OBSERVABLES = [("EnergyLArNoNoise", "noise-free"), ("EnergyLAr", "with electronic noise")]
TRUNC = 0.8796  # std of a unit Gaussian truncated at +-2 sigma
NOISE_LAR_GEV = 4e-3 * math.sqrt(1792)
N_BOOT = 300

# Validated categorical slots 1-3 (dataviz reference palette, light mode), with distinct markers
STYLE = {
    "cpu": dict(color="#2a78d6", marker="o"),
    "matched": dict(color="#eb6834", marker="s"),
    "adept": dict(color="#1baf7a", marker="^"),
}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def core(x, iterations=20):
    mu, sigma = np.median(x), np.std(x)
    for _ in range(iterations):
        sel = x[np.abs(x - mu) < 2 * sigma]
        new_mu, new_sigma = sel.mean(), sel.std(ddof=1) / TRUNC
        if abs(new_mu - mu) < 1e-6 * abs(mu) and abs(new_sigma - sigma) < 1e-6 * sigma:
            break
        mu, sigma = new_mu, new_sigma
    return mu, sigma


def estimators(x):
    mu_c, s_c = core(x)
    return s_c / mu_c, np.std(x, ddof=1) / np.mean(x), mu_c


def with_bootstrap(x, rng):
    value = estimators(x)
    boot = np.array([estimators(rng.choice(x, size=len(x)))[:2] for _ in range(N_BOOT)])
    return value, boot.std(axis=0, ddof=1)


def wls(X, y, dy):
    """Weighted linear least squares: returns parameters, covariance, chi2, ndf."""
    w = 1 / dy**2
    cov = np.linalg.inv(X.T @ (X * w[:, None]))
    p = cov @ (X.T @ (w * y))
    chi2 = float(np.sum(w * (y - X @ p) ** 2))
    return p, cov, chi2, len(y) - len(p)


def fit_noise_free(E, r, dr):
    """(sigma/mu)^2 = A/E + B, linear in A = a^2 and B = b^2 (B may come out negative)."""
    X = np.column_stack([1 / E, np.ones_like(E)])
    return wls(X, r**2, 2 * r * dr)


def fit_noise(E, r, dr, A, B):
    """(sigma/mu)^2 = A/E + B + C/E^2 with A, B fixed from the noise-free fit: only C = c^2 is free."""
    X = (1 / E**2)[:, None]
    return wls(X, r**2 - A / E - B, 2 * r * dr)


def curve(E, A, B, C=0.0):
    return np.sqrt(np.clip(A / E + B + C / E**2, 0, None))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory")
    parser.add_argument("--energies", type=float, nargs="+", default=[5, 10, 20, 40, 100])
    parser.add_argument("--plot")
    args = parser.parse_args()

    rng = np.random.default_rng(20261123)
    energies = np.array(args.energies, dtype=float)
    res = {}  # (setup, observable) -> arrays
    setups = []
    for setup, label in SETUPS:
        paths = [os.path.join(args.directory, f"e{E:g}GeV", setup, "ATLLArBarrelout_Run0.root") for E in energies]
        if not all(os.path.isfile(p) for p in paths):
            missing = [p for p in paths if not os.path.isfile(p)]
            print(f"skipping {label}: {len(missing)} file(s) missing, e.g. {missing[0]}", file=sys.stderr)
            continue
        setups.append((setup, label))
        for obs, _ in OBSERVABLES:
            rows = []
            for p in paths:
                with uproot.open(p) as f:
                    x = f["ATLLArBarrelout"][obs].array(library="np")
                (c, r, mu), (dc, dr) = with_bootstrap(x, rng)
                rows.append((c, dc, r, dr, mu, len(x)))
            res[setup, obs] = np.array(rows)
    if not setups:
        sys.exit("no complete setup found")

    for obs, obs_label in OBSERVABLES:
        print(f"## sigma/mu of {obs} ({obs_label}), %\n")
        print("| E (GeV) | " + " | ".join(f"{l} core | {l} rms" for _, l in setups) + " |")
        print("|---" * (1 + 2 * len(setups)) + "|")
        for i, E in enumerate(energies):
            cells = []
            for s, _ in setups:
                c, dc, r, dr, _, _ = res[s, obs][i]
                cells += [f"{100*c:.3f} ± {100*dc:.3f}", f"{100*r:.3f} ± {100*dr:.3f}"]
            print(f"| {E:g} | " + " | ".join(cells) + " |")
        print()

        if obs == "EnergyLArNoNoise":
            print("Fit of the core resolution: (sigma/mu)^2 = a^2/E + b^2, linear in a^2 and b^2\n")
            print("| Setup | a (% sqrt(GeV)) | b^2 (%^2) | b (%) | chi2/ndf |")
            print("|---|---|---|---|---|")
            for s_, label in setups:
                d = res[s_, obs]
                if len(energies) < 3:
                    print(f"| {label} | too few energies | | | |")
                    continue
                (A, B), cov, chi2, ndf = fit_noise_free(energies, d[:, 0], d[:, 1])
                dA, dB = np.sqrt(np.diag(cov))
                a, da = math.sqrt(max(A, 0)), dA / (2 * math.sqrt(A)) if A > 0 else math.nan
                if B > 2 * dB:
                    b_txt = f"{100*math.sqrt(B):.2f} ± {100*dB/(2*math.sqrt(B)):.2f}"
                else:  # compatible with 0: one-sided 95 % limit
                    b_txt = f"< {100*math.sqrt(max(B, 0) + 1.645 * dB):.2f} (95 % CL)"
                print(f"| {label} | {100*a:.2f} ± {100*da:.2f} | {1e4*B:+.3f} ± {1e4*dB:.3f} | {b_txt} | {chi2:.1f}/{ndf} |")
                res[s_, "fit"] = (A, B)
        else:
            print("Fit of the noise term: (sigma/mu)^2 = a^2/E + b^2 + c^2/E^2, a and b fixed from the noise-free fit\n")
            print("| Setup | c (GeV) | expected c (GeV) | chi2/ndf |")
            print("|---|---|---|---|")
            for s_, label in setups:
                if (s_, "fit") not in res:
                    continue
                A, B = res[s_, "fit"]
                d = res[s_, obs]
                (C,), cov, chi2, ndf = fit_noise(energies, d[:, 0], d[:, 1], A, B)
                c, dc = math.sqrt(max(C, 0)), math.sqrt(cov[0, 0]) / (2 * math.sqrt(C)) if C > 0 else math.nan
                # response mu/E from the noise-free core mean, averaged over the energies
                frac = np.mean(res[s_, "EnergyLArNoNoise"][:, 4] / (1000 * energies))
                print(f"| {label} | {c:.3f} ± {dc:.3f} | {NOISE_LAR_GEV / frac:.3f} | {chi2:.1f}/{ndf} |")
                res[s_, "fitC"] = C
        print()

    if args.plot:
        plot(args.plot, energies, setups, res)


def plot(path, energies, setups, res):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED,
                         "ytick.color": MUTED, "text.color": INK})
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1.3]),
                             facecolor=SURFACE)
    offsets = {s: f for (s, _), f in zip(setups, np.linspace(-0.04, 0.04, len(setups)))}
    reference = "matched" if any(s == "matched" for s, _ in setups) else setups[0][0]
    ref_label = dict(setups)[reference]
    Efine = np.geomspace(energies.min() * 0.85, energies.max() * 1.15, 200)
    for col, (obs, obs_label) in enumerate(OBSERVABLES):
        top, bottom = axes[0, col], axes[1, col]
        for s, label in setups:
            d = res[s, obs]
            x = energies * 10 ** offsets[s]
            st = STYLE[s]
            top.errorbar(x, 100 * d[:, 0], yerr=100 * d[:, 1], ls="none", marker=st["marker"], ms=7, color=st["color"],
                         mec=SURFACE, mew=1, elinewidth=1.5, capsize=0, label=label, zorder=3)
            if (s, "fit") in res:
                C = res.get((s, "fitC"), 0.0) if obs == "EnergyLAr" else 0.0
                top.plot(Efine, 100 * curve(Efine, *res[s, "fit"], C), color=st["color"], lw=1.5, alpha=0.8, zorder=2)
            ref = res[reference, obs]
            ratio = d[:, 0] / ref[:, 0]
            dratio = ratio * np.hypot(d[:, 1] / d[:, 0], ref[:, 1] / ref[:, 0]) if s != reference else d[:, 1] / d[:, 0]
            bottom.errorbar(x, ratio, yerr=dratio, ls="none", marker=st["marker"], ms=6, color=st["color"], mec=SURFACE,
                            mew=1, elinewidth=1.5, capsize=0, zorder=3)
        top.set_title(f"{obs} ({obs_label})", color=INK, fontsize=11, loc="left")
        top.set_ylabel("core σ/μ (%)")
        bottom.axhline(1, color=AXIS, lw=1, zorder=1)
        bottom.set_ylabel(f"ratio to {ref_label}" if col == 0 else "")
        bottom.set_xlabel("electron energy (GeV)")
        bottom.set_xscale("log")
        bottom.set_xticks(energies)
        bottom.set_xticklabels([f"{e:g}" for e in energies])
        bottom.minorticks_off()
        for ax in (top, bottom):
            ax.set_facecolor(SURFACE)
            ax.grid(True, color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
    axes[0, 0].legend(frameon=False, loc="upper right", labelcolor=INK2)
    fig.suptitle("ATLLArBarrel: LAr energy resolution, 1000 e⁻ per point, η = 0.2 (lines: fits, see text)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    print(f"plot: {path}")


if __name__ == "__main__":
    main()
