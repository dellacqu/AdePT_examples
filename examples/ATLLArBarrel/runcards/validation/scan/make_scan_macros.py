#!/usr/bin/env python3
"""Write the position-scan macros of this directory (run once; the macros are kept in the repository).

20 GeV electrons from the origin, 1000 events per point, scanning the beam direction from a cell boundary
to the cell centre of the middle layer (Delta eta = 0.025, Delta phi = 2 pi / 256), in steps of 1/8 cell:
  eta scan: eta = 0.2 (boundary) ... 0.2125 (centre), phi = 0.01225 (centre)
  phi scan: phi = 0 (boundary) ... 3/8 cell, eta = 0.2125 (centre); its centre point is the last eta point
Each point has a CPU macro (FTFP_BERT and matched builds) and an AdePT macro with the same Geant4 seeds.
"""
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
D_ETA = 0.025
D_PHI = 2 * math.pi / 256
POINTS = [(0.2 + k * D_ETA / 8, 0.01225) for k in range(5)] + [(0.2125, k * D_PHI / 8) for k in range(4)]

ADEPT = """/adept/setSeed {aseed}
/adept/setVerbosity 0
/adept/setMillionsOfTrackSlots 1
/adept/setMillionsOfHitSlots 1
/adept/setCUDAStackLimit 8192
/adept/setTrackInAllRegions true
/adept/SetEnergyLossFluctuation true
"""

for i, (eta, phi) in enumerate(POINTS):
    theta = 2 * math.atan(math.exp(-eta))
    d = (math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta))
    name = f"scan_eta{eta:.6f}_phi{phi:.6f}".replace(".", "p")
    for adept in (False, True):
        lines = [f"#Position scan (see README.md), 20 GeV e-, eta {eta:.6f}, phi {phi:.6f}, 1000 events",
                 "#" + ("AdePT build (-DWITH_ADEPT=ON)" if adept else "CPU builds (FTFP_BERT and matched)") +
                 ", requires -DWITH_GPS=OFF",
                 f"/random/setSeeds 20261123 {3000 + i}"]
        text = "\n".join(lines) + "\n"
        if adept:
            text += ADEPT.format(aseed=8000 + i)
        text += ("#/run/setCut must precede /run/initialize (AdePT issue #745)\n/run/setCut 0.01 mm\n/run/initialize\n"
                 "/run/printProgress 100\n/gun/particle e-\n/gun/energy 20 GeV\n/gun/position 0. 0. 0. m\n"
                 f"/gun/direction {d[0]:.7f} {d[1]:.7f} {d[2]:.7f}\n/run/beamOn 1000\n")
        with open(os.path.join(HERE, name + ("_adept" if adept else "") + ".mac"), "w") as f:
            f.write(text)
    print(name)
