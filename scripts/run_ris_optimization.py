"""
run_ris_optimization.py
------------------------
Simulation de l'optimisation de phase d'une RIS pour un lien SISO
obstrué (lien direct faible), et comparaison :
    - sans RIS (lien direct seul)
    - RIS avec phases aléatoires (baseline non optimisée)
    - RIS avec phases optimisées (alignement de phase, solution fermée)

Génère deux figures dans figures/ :
    - ris_capacity_vs_elements.png : capacité vs nombre d'éléments RIS
    - ris_gradient_convergence.png : convergence du gradient ascent
      vers la solution fermée (validation numérique)
"""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from channels import ris_cascaded_channel
from capacity import capacity_siso, snr_linear
from ris import (
    combined_channel, optimal_phase_alignment, random_phase,
    gradient_ascent_phase,
)

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

SEED = 123
N_TRIALS = 3000
SNR_DB = 5  # SNR de référence (lien direct faible -> RIS utile)


def average_capacity_vs_n_elements():
    snr = snr_linear(SNR_DB)
    n_values = [0, 4, 8, 16, 32, 64, 128, 256]
    rng = np.random.default_rng(SEED)

    cap_no_ris = []
    cap_random = []
    cap_optimal = []

    for n in n_values:
        c_direct, c_rand, c_opt = 0.0, 0.0, 0.0
        for _ in range(N_TRIALS):
            if n == 0:
                # Pas de RIS : on prend juste un lien direct atténué
                h_d, _, _ = ris_cascaded_channel(1, rng, rician_k_direct=0.0)
                c_direct += capacity_siso(h_d, snr)
                c_rand += capacity_siso(h_d, snr)
                c_opt += capacity_siso(h_d, snr)
                continue

            h_d, h_1, h_2 = ris_cascaded_channel(n, rng, rician_k_direct=0.0)

            c_direct += capacity_siso(h_d, snr)

            theta_rand = random_phase(n, rng)
            h_eq_rand = combined_channel(h_d, h_1, h_2, theta_rand)
            c_rand += capacity_siso(h_eq_rand, snr)

            theta_opt = optimal_phase_alignment(h_d, h_1, h_2)
            h_eq_opt = combined_channel(h_d, h_1, h_2, theta_opt)
            c_opt += capacity_siso(h_eq_opt, snr)

        cap_no_ris.append(c_direct / N_TRIALS)
        cap_random.append(c_rand / N_TRIALS)
        cap_optimal.append(c_opt / N_TRIALS)

    return n_values, cap_no_ris, cap_random, cap_optimal


def plot_capacity_vs_n_elements():
    n_values, cap_no_ris, cap_random, cap_optimal = average_capacity_vs_n_elements()

    plt.figure(figsize=(8, 6))
    plt.plot(n_values, cap_no_ris, marker="o", label="Sans RIS (lien direct seul)")
    plt.plot(n_values, cap_random, marker="s", label="RIS, phases aléatoires")
    plt.plot(n_values, cap_optimal, marker="^", label="RIS, phases optimisées")
    plt.xlabel("Nombre d'éléments RIS (N)")
    plt.ylabel("Capacité moyenne (bits/s/Hz)")
    plt.title(f"Impact de l'optimisation de phase RIS sur la capacité\n"
              f"(SNR = {SNR_DB} dB, Monte Carlo {N_TRIALS} tirages/point)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "ris_capacity_vs_elements.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure enregistrée : {out_path}")

    # Gain de capacité optimal vs N=0 (sans RIS), pour discussion README
    gain = np.array(cap_optimal) - np.array(cap_no_ris)
    print("Gain de capacité (RIS optimisée - sans RIS), en bits/s/Hz :")
    for n, g in zip(n_values, gain):
        print(f"  N={n:>4d} : +{g:.3f} bits/s/Hz")


def plot_gradient_convergence():
    rng = np.random.default_rng(7)
    n = 64
    h_d, h_1, h_2 = ris_cascaded_channel(n, rng, rician_k_direct=0.0)

    theta_opt = optimal_phase_alignment(h_d, h_1, h_2)
    gain_opt = np.abs(combined_channel(h_d, h_1, h_2, theta_opt)) ** 2

    _, history = gradient_ascent_phase(h_d, h_1, h_2, n_iter=1500, lr=0.01, rng=rng)

    plt.figure(figsize=(8, 6))
    plt.plot(history, label="Gradient ascent (montée de gradient)")
    plt.axhline(gain_opt, color="red", linestyle="--",
                label="Optimum théorique (alignement de phase, solution fermée)")
    plt.xlabel("Itération")
    plt.ylabel(r"Gain du canal équivalent $|h_{eq}|^2$")
    plt.title(f"Convergence du gradient ascent vers l'optimum (N={n} éléments)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(FIG_DIR, "ris_gradient_convergence.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Figure enregistrée : {out_path}")


if __name__ == "__main__":
    print("Simulation 1/2 : capacité vs nombre d'éléments RIS...")
    plot_capacity_vs_n_elements()
    print("\nSimulation 2/2 : convergence du gradient ascent...")
    plot_gradient_convergence()
    print("\nTerminé.")
