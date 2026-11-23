#!/bin/bash
# Electron validation of ATLLArBarrel: FTFP_BERT (CPU), matched G4HepEm (CPU) and AdePT (GPU),
# 1000 electrons at 5, 10, 20, 40 and 100 GeV each. See README.md in this directory.
#
# Usage:
#   CPU_EXE=<build-cpu>/ATLLArBarrel MATCHED_EXE=<build-hepemem>/ATLLArBarrel \
#   ADEPT_EXE=<build-adept>/ATLLArBarrel ./run_validation.sh <output-dir>
#
# Optional variables: THREADS (default 8), ENERGIES (default "5 10 20 40 100"),
# SCAN=1 (run the position-scan macros of scan/ instead of the energy points),
# EVENTS (default: the 1000 of the macros; otherwise /run/beamOn is changed in the copied macro),
# PYTHON (a python with uproot and numpy, for the summary; default python3).
# A setup whose *_EXE is not set is skipped. The runs are sequential, so the times are not
# affected by other runs of this script.

set -euo pipefail

OUT=${1:?usage: run_validation.sh <output-dir>}
THREADS=${THREADS:-8}
ENERGIES=${ENERGIES:-"5 10 20 40 100"}
EVENTS=${EVENTS:-}
SCAN=${SCAN:-}
PYTHON=${PYTHON:-python3}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ESTATS="$HERE/../../analysis/estats.py"

mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)

declare -A EXE=([cpu]=${CPU_EXE:-} [matched]=${MATCHED_EXE:-} [adept]=${ADEPT_EXE:-})
for setup in cpu matched adept; do
  if [[ -n "${EXE[$setup]}" ]]; then
    EXE[$setup]=$(readlink -f "${EXE[$setup]}")
    [[ -x "${EXE[$setup]}" ]] || { echo "$setup: ${EXE[$setup]} is not an executable" >&2; exit 1; }
  fi
done

# Environment record
{
  echo "date: $(date -Iseconds)"
  echo "repository: $(git -C "$HERE" rev-parse HEAD 2>/dev/null)$(git -C "$HERE" diff --quiet HEAD -- . 2>/dev/null || echo ' (with local changes)')"
  echo "threads: $THREADS"
  if [[ -n "$SCAN" ]]; then echo "points: position scan (scan/*.mac)"; else echo "energies (GeV): $ENERGIES"; fi
  echo "events: ${EVENTS:-as in the macros (1000)}"
  for setup in cpu matched adept; do
    echo "$setup executable: ${EXE[$setup]:-skipped}"
    # libraries actually linked (geant4-config in PATH may belong to another installation)
    [[ -n "${EXE[$setup]}" ]] && ldd "${EXE[$setup]}" | grep -oE '/[^ ]*lib(G4run|g4HepEm|adept)[^ ]*\.so[^ ]*' | sed 's/^/  /' || true
  done
  echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ //'), $(nproc) logical cores"
  if command -v nvidia-smi >/dev/null; then
    echo "gpu: $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
  fi
  # the CUDA version of the AdePT build is in its CMakeCache.txt (CMAKE_CUDA_COMPILER), not necessarily the nvcc in PATH
} > "$OUT/environment.txt"
cat "$OUT/environment.txt"

# Points to run: macro base name (CPU macro; the AdePT one adds _adept) and output directory name
MACROS=() NAMES=()
if [[ -n "$SCAN" ]]; then
  for m in "$HERE"/scan/scan_*.mac; do
    [[ $m == *_adept.mac ]] && continue
    MACROS+=("${m%.mac}") NAMES+=("$(basename "${m%.mac}")")
  done
else
  for E in $ENERGIES; do MACROS+=("$HERE/electron${E}GeV") NAMES+=("e${E}GeV"); done
fi

: > "$OUT/times.txt"
for i in "${!MACROS[@]}"; do
  P=${NAMES[$i]}
  for setup in cpu matched adept; do
    [[ -n "${EXE[$setup]}" ]] || continue
    macro="${MACROS[$i]}.mac"
    [[ $setup == adept ]] && macro="${MACROS[$i]}_adept.mac"
    dir="$OUT/$P/$setup"
    mkdir -p "$dir"
    cp "$macro" "$dir/"
    [[ -n "$EVENTS" ]] && sed -i "s#^/run/beamOn .*#/run/beamOn $EVENTS#" "$dir/$(basename "$macro")"
    echo "== $P, $setup"
    start=$(date +%s)
    (cd "$dir" && "${EXE[$setup]}" -m "$(basename "$macro")" -t "$THREADS" > run.log 2>&1) || {
      echo "$P $setup failed, see $dir/run.log" >&2
      exit 1
    }
    wall=$(( $(date +%s) - start ))
    # event-loop time of the master (the last "Time:" line without a worker prefix)
    loop=$(grep -E '^ +Time: User=' "$dir/run.log" | tail -1 | grep -o 'Real=[0-9.]*s' || true)
    exceptions=$(grep -c 'G4Exception' "$dir/run.log" || true)
    echo "$P $setup wall=${wall}s eventloop_${loop:-unknown} G4Exception=$exceptions" | tee -a "$OUT/times.txt"
  done
done

# Statistics per point: AdePT against the matched reference and against FTFP_BERT
if "$PYTHON" -c 'import uproot, numpy' 2>/dev/null; then
  for P in "${NAMES[@]}"; do
    args=() diffs=()
    [[ -n "${EXE[cpu]}" ]] && args+=("FTFP_BERT=$OUT/$P/cpu/ATLLArBarrelout_Run0.root")
    [[ -n "${EXE[matched]}" ]] && args+=("matched=$OUT/$P/matched/ATLLArBarrelout_Run0.root")
    [[ -n "${EXE[adept]}" ]] && args+=("AdePT=$OUT/$P/adept/ATLLArBarrelout_Run0.root")
    [[ -n "${EXE[adept]}" && -n "${EXE[matched]}" ]] && diffs+=(--diff AdePT-matched)
    [[ -n "${EXE[adept]}" && -n "${EXE[cpu]}" ]] && diffs+=(--diff AdePT-FTFP_BERT)
    { echo "# $P"; echo; "$PYTHON" -I "$ESTATS" "${args[@]}" "${diffs[@]}"; echo; } > "$OUT/$P/estats.md"
  done
  for P in "${NAMES[@]}"; do cat "$OUT/$P/estats.md"; done > "$OUT/summary.md"
  echo "summary: $OUT/summary.md"
else
  echo "uproot/numpy not found for $PYTHON: run analysis/estats.py on the ROOT files by hand" >&2
fi
