"""
Tests unitaires basiques (pas de framework externe, exécutable avec
`python -m pytest tests/` ou directement `python tests/test_capacity_ris.py`).
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from capacity import (
    capacity_siso, capacity_simo, capacity_miso,
    capacity_mimo_equal_power, capacity_mimo_waterfilling, snr_linear,
)
from ris import (
    combined_channel, optimal_phase_alignment, gradient_ascent_phase,
)


def test_snr_linear():
    assert np.isclose(snr_linear(0), 1.0)
    assert np.isclose(snr_linear(10), 10.0)


def test_capacity_siso_zero_snr():
    # A SNR nul, la capacité doit être nulle quel que soit h
    assert np.isclose(capacity_siso(1 + 1j, 0.0), 0.0)


def test_siso_simo_miso_consistency():
    # Avec un seul lien (Nr=1 ou Nt=1), SIMO/MISO doivent redonner SISO
    h = 0.7 - 0.3j
    snr = 5.0
    c_siso = capacity_siso(h, snr)
    c_simo = capacity_simo(np.array([h]), snr)
    c_miso = capacity_miso(np.array([h]), snr)
    assert np.isclose(c_siso, c_simo)
    assert np.isclose(c_siso, c_miso)


def test_mimo_waterfilling_ge_equal_power():
    # Le waterfilling optimal doit toujours faire au moins aussi bien
    # que la répartition égale de puissance
    rng = np.random.default_rng(0)
    H = (rng.standard_normal((4, 4)) + 1j * rng.standard_normal((4, 4))) / np.sqrt(2)
    snr = 10.0
    c_eq = capacity_mimo_equal_power(H, snr)
    c_wf = capacity_mimo_waterfilling(H, snr)
    assert c_wf >= c_eq - 1e-9


def test_ris_optimal_phase_beats_direct_link():
    rng = np.random.default_rng(1)
    n = 32
    h_d = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    h_1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    h_2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

    theta_opt = optimal_phase_alignment(h_d, h_1, h_2)
    h_eq_opt = combined_channel(h_d, h_1, h_2, theta_opt)

    theta_rand = rng.uniform(0, 2 * np.pi, size=n)
    h_eq_rand = combined_channel(h_d, h_1, h_2, theta_rand)

    assert np.abs(h_eq_opt) ** 2 >= np.abs(h_eq_rand) ** 2
    assert np.abs(h_eq_opt) ** 2 >= np.abs(h_d) ** 2


def test_gradient_ascent_matches_closed_form():
    rng = np.random.default_rng(2)
    n = 16
    h_d = (rng.standard_normal() + 1j * rng.standard_normal()) / np.sqrt(2)
    h_1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    h_2 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)

    theta_opt = optimal_phase_alignment(h_d, h_1, h_2)
    gain_opt = np.abs(combined_channel(h_d, h_1, h_2, theta_opt)) ** 2

    theta_grad, history = gradient_ascent_phase(
        h_d, h_1, h_2, n_iter=2000, lr=0.01, rng=rng
    )
    gain_grad = history[-1]

    # Le gradient doit converger proche de l'optimum fermé (à 1% près)
    assert gain_grad >= 0.99 * gain_opt


if __name__ == "__main__":
    tests = [
        test_snr_linear,
        test_capacity_siso_zero_snr,
        test_siso_simo_miso_consistency,
        test_mimo_waterfilling_ge_equal_power,
        test_ris_optimal_phase_beats_direct_link,
        test_gradient_ascent_matches_closed_form,
    ]
    for t in tests:
        t()
        print(f"OK  {t.__name__}")
    print("\nTous les tests sont passés.")
