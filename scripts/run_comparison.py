"""
run_comparison.py
------------------
Simulation Monte Carlo comparant la capacité de Shannon moyenne
(sur canal à évanouissement de Rayleigh) des configurations
SISO / SIMO / MISO / MIMO en fonction du SNR, et de la capacité
MIMO en fonction du nombre d'antennes.

Génère deux figures dans figures/ :
    - capacity_vs_snr.png
    - mimo_capacity_vs_antennas.png
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channels import siso_channel, simo_channel, miso_channel, mimo_channel
from capacity import (
    capacity_siso, capacity_simo, capacity_miso,
    capacity_mimo_equal_power, capacity_mimo_waterfilling, snr_linear,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 42
N_TRIALS = 5000  # nombre de réalisations de canal par point (Monte Carlo)


def average_capacity_vs_snr(snr_db_range, nr=2, nt=2, n_trials=N_TRIALS, seed=SEED):
    rng = np.random.default_rng(seed)

    results = {
        "SISO": np.zeros_like(snr_db_range, dtype=float),
        f"SIMO ({nr}x1)": np.zeros_like(snr_db_range, dtype=float),
        f"MISO (1x{nt})": np.zeros_like(snr_db_range, dtype=float),
        f"MIMO ({nr}x{nt}, eq. power)": np.zeros_like(snr_db_range, dtype=float),
        f"MIMO ({nr}x{nt}, waterfilling)": np.zeros_like(snr_db_range, dtype=float),
    }

    for idx, snr_db in enumerate(snr_db_range):
        snr = snr_linear(snr_db)
        c_siso, c_simo, c_miso, c_mimo_eq, c_mimo_wf = 0.0, 0.0, 0.0, 0.0, 0.0

        for _ in range(n_trials):
            h = siso_channel(rng)[0, 0]
            c_siso += capacity_siso(h, snr)

            h_simo = simo_channel(nr, rng)
            c_simo += capacity_simo(h_simo, snr)

            h_miso = miso_channel(nt, rng)
            c_miso += capacity_miso(h_miso, snr)

            H = mimo_channel(nr, nt, rng)
            c_mimo_eq += capacity_mimo_equal_power(H, snr)
            c_mimo_wf += capacity_mimo_waterfilling(H, snr)

        results["SISO"][idx] = c_siso / n_trials
        results[f"SIMO ({nr}x1)"][idx] = c_simo / n_trials
        results[f"MISO (1x{nt})"][idx] = c_miso / n_trials
        results[f"MIMO ({nr}x{nt}, eq. power)"][idx] = c_mimo_eq / n_trials
        results[f"MIMO ({nr}x{nt}, waterfilling)"][idx] = c_mimo_wf / n_trials

    return results


def plot_capacity_vs_snr():
    snr_db_range = np.arange(-10, 21, 2)
    results = average_capacity_vs_snr(snr_db_range, nr=2, nt=2)

    plt.figure(figsize=(8, 6))
    markers = ["o", "s", "^", "D", "v"]
    for (label, curve), m in zip(results.items(), markers):
        plt.plot(snr_db_range, curve, marker=m, label=label)

    plt.xlabel("SNR (dB)")
    plt.ylabel("Capacité moyenne (bits/s/Hz)")
    plt.title("Capacité de Shannon moyenne — canal de Rayleigh i.i.d.\n"
              "(Monte Carlo, {} réalisations/point)".format(N_TRIALS))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "capacity_vs_snr.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure enregistrée : {out_path}")


def plot_mimo_capacity_vs_antennas():
    snr_db = 10
    snr = snr_linear(snr_db)
    antenna_configs = [1, 2, 4, 8, 16]
    rng = np.random.default_rng(SEED)
    n_trials = N_TRIALS

    cap_eq = []
    cap_wf = []
    for n in antenna_configs:
        c_eq, c_wf = 0.0, 0.0
        for _ in range(n_trials):
            H = mimo_channel(n, n, rng)
            c_eq += capacity_mimo_equal_power(H, snr)
            c_wf += capacity_mimo_waterfilling(H, snr)
        cap_eq.append(c_eq / n_trials)
        cap_wf.append(c_wf / n_trials)

    plt.figure(figsize=(8, 6))
    plt.plot(antenna_configs, cap_eq, marker="o", label="Répartition égale (pas de CSIT)")
    plt.plot(antenna_configs, cap_wf, marker="s", label="Waterfilling (CSIT parfaite)")
    plt.xlabel("Nombre d'antennes (Nr = Nt = N)")
    plt.ylabel("Capacité moyenne (bits/s/Hz)")
    plt.title(f"Croissance de la capacité MIMO N x N avec le nombre d'antennes\n"
              f"(SNR = {snr_db} dB, Rayleigh i.i.d.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "mimo_capacity_vs_antennas.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure enregistrée : {out_path}")


if __name__ == "__main__":
    print("Simulation 1/2 : capacité vs SNR (SISO/SIMO/MISO/MIMO)...")
    plot_capacity_vs_snr()
    print("Simulation 2/2 : capacité MIMO vs nombre d'antennes...")
    plot_mimo_capacity_vs_antennas()
    print("Terminé.")
