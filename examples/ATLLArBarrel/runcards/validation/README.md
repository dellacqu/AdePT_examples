# Electron validation: FTFP_BERT, matched G4HepEm and AdePT

1000 electrons at 5, 10, 20, 40 and 100 GeV, from the origin towards η = 0.2, φ = 0.01225, production cut
0.01 mm, in three setups:

| Setup | Build | Macro |
|---|---|---|
| FTFP_BERT (CPU) | default (`build-cpu`) | `electron<E>GeV.mac` |
| matched G4HepEm (CPU) | `-DWITH_G4HepEm=ON -DWITH_G4HepEmEM=ON` (`build-hepemem`) | `electron<E>GeV.mac` |
| AdePT (GPU) | `-DWITH_ADEPT=ON`, or the AdePT_examples top-level build | `electron<E>GeV_adept.mac` |

All builds with `-DWITH_GPS=OFF` and `CMAKE_BUILD_TYPE=Release` ([BUILD_OPTIONS.md](../../BUILD_OPTIONS.md)).
The matched setup has the same EM physics as AdePT on the CPU: it is the reference that tests the GPU transport.
FTFP_BERT differs from both by the G4HepEm vs Geant4 EM physics.

## Reproducibility

Every macro fixes the Geant4 seeds (`/random/setSeeds`, the same for the CPU and AdePT macros of one energy), and
the AdePT macros also fix the GPU seed (`/adept/setSeed`). With the same build and thread count, a rerun gives the
same events (with AdePT, up to the last bit of the per-event sums, `Results.md`). All `/adept/` commands and
`/run/setCut` come before `/run/initialize` ([AdePT#745](https://github.com/apt-sim/AdePT/issues/745)).

## Running

```sh
CPU_EXE=<build-cpu>/ATLLArBarrel MATCHED_EXE=<build-hepemem>/ATLLArBarrel ADEPT_EXE=<build-adept>/ATLLArBarrel \
  ./run_validation.sh <output-dir>
```

Optional: `THREADS` (default 8), `ENERGIES` (default `"5 10 20 40 100"`), `EVENTS` (changes `/run/beamOn` in the
copied macros, e.g. `EVENTS=10000 ENERGIES=40`), `SCAN=1` (the position scan below), `PYTHON` (a Python with `uproot`
and `numpy`, for the summary). The 15 runs are sequential, so the times are not disturbed by each other; on a Ryzen 9
3900 with an RTX 2070 SUPER they take about half an hour.

The output directory receives:

| File | Content |
|---|---|
| `environment.txt` | date, repository commit, executables and the Geant4/G4HepEm/AdePT libraries they load, CPU, GPU |
| `times.txt` | wall time, Geant4 event-loop time and number of `G4Exception` per run |
| `e<E>GeV/<setup>/` | the macro used, `run.log` and `ATLLArBarrelout_Run0.root` |
| `e<E>GeV/estats.md`, `summary.md` | means and differences from [`analysis/estats.py`](../../analysis/estats.py) |

The energy resolution curve, with fits and plot:

```sh
python3 ../../analysis/resolution.py <output-dir> --plot resolution.png
```

## Position scan across a middle-layer cell

`scan/` holds 9 points × 2 macros (CPU and AdePT) of 20 GeV electrons, 1000 events each, written by
`scan/make_scan_macros.py`. The beam direction goes from a cell boundary to the cell centre of the middle layer
(Δη = 0.025, Δφ = 2π/256) in steps of 1/8 cell: an η scan at the φ centre (η = 0.2 to 0.2125; η = 0.2 is the
direction of the energy points, which lies on an η boundary of all three layers) and a φ scan at the η centre
(φ = 0 to 3/8 cell). The analysis needs the noise-free cells and the boundary diagnostics of the ntuple
(`*HitsEdepNoNoise`, `EnergyCrossCell`, `EnergyCrossSection`).

```sh
SCAN=1 CPU_EXE=... MATCHED_EXE=... ADEPT_EXE=... ./run_validation.sh <scan-dir>
python3 ../../analysis/scan.py <scan-dir> --plot scan.png
```

The 27 runs take about 35 minutes on the machine above.
