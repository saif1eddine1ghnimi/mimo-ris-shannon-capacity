"""
ris.py
------
Modélisation d'une RIS (Reconfigurable Intelligent Surface) à N
éléments réfléchissants passifs, insérée dans une liaison SISO, et
optimisation des phases pour maximiser le SNR reçu (donc la capacité).

Modèle de canal combiné (lien direct + lien réfléchi) :

    h_eq = h_d + h_2^T * Theta * h_1
         = h_d + sum_{i=1}^{N} h_2[i] * e^{j*theta_i} * h_1[i]

où Theta = diag(e^{j theta_1}, ..., e^{j theta_N}) est la matrice de
phase imposée par la RIS (gain d'amplitude = 1 par élément, cas idéal
souvent supposé dans la littérature RIS).

Le SNR reçu est proportionnel à |h_eq|^2. Le problème d'optimisation
de phase est donc :

    max_{theta_1,...,theta_N}  | h_d + sum_i h_2[i] h_1[i] e^{j theta_i} |^2

Solution optimale (alignement de phase) :
    Chaque terme h_1[i] h_2[i] e^{j theta_i} doit être mis en phase
    avec h_d, c'est-à-dire :
        theta_i* = arg(h_d) - arg(h_1[i] h_2[i])
    Cette solution est optimale car elle maximise chaque terme de la
    somme simultanément dans la direction de h_d (cf. Wu & Zhang, 2019,
    "Intelligent Reflecting Surface Enhanced Wireless Network").

On fournit également :
    - une méthode "phase aléatoire" (baseline, sans RIS optimisée)
    - une méthode d'optimisation itérative par montée de gradient
      projetée (pour illustrer/valider numériquement la solution fermée)
"""

from typing import Tuple

import numpy as np


def combined_channel(h_d: complex, h_1: np.ndarray, h_2: np.ndarray,
                      theta: np.ndarray) -> complex:
    """Calcule h_eq = h_d + sum_i h_1[i] h_2[i] e^{j theta_i}."""
    return h_d + np.sum(h_1 * h_2 * np.exp(1j * theta))


def optimal_phase_alignment(h_d: complex, h_1: np.ndarray,
                             h_2: np.ndarray) -> np.ndarray:
    """
    Solution optimale (fermée) d'alignement de phase :
        theta_i* = arg(h_d) - arg(h_1[i] * h_2[i])

    Returns
    -------
    theta : ndarray (N,) des phases optimales dans [-pi, pi)
    """
    cross = h_1 * h_2
    theta_star = np.angle(h_d) - np.angle(cross)
    return theta_star


def random_phase(n_ris: int, rng: np.random.Generator) -> np.ndarray:
    """Phases tirées uniformément dans [0, 2*pi) — baseline non optimisée."""
    return rng.uniform(0, 2 * np.pi, size=n_ris)


def gradient_ascent_phase(
    h_d: complex,
    h_1: np.ndarray,
    h_2: np.ndarray,
    n_iter: int = 2000,
    lr: float = 0.01,
    rng: np.random.Generator = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Optimisation itérative de theta par montée de gradient sur
    f(theta) = |h_eq(theta)|^2, utilisée comme validation numérique
    de la solution fermée `optimal_phase_alignment`.

    Le gradient de f par rapport à theta_i (à theta_j, j!=i fixés)
    est dérivé analytiquement :
        f(theta) = |h_d + sum_k a_k e^{j theta_k}|^2 ,  a_k = h_1[k] h_2[k]
        df/dtheta_i = -2 * Im( conj(h_eq) * a_i * e^{j theta_i} )

    Returns
    -------
    theta : phases finales
    history : valeurs de |h_eq|^2 au fil des itérations (pour tracer
              la convergence)
    """
    n = len(h_1)
    if rng is None:
        rng = np.random.default_rng()
    theta = rng.uniform(0, 2 * np.pi, size=n)
    a = h_1 * h_2
    history = np.zeros(n_iter)

    for it in range(n_iter):
        h_eq = combined_channel(h_d, h_1, h_2, theta)
        history[it] = np.abs(h_eq) ** 2
        grad = -2 * np.imag(np.conj(h_eq) * a * np.exp(1j * theta))
        theta = theta + lr * grad  # montée de gradient (maximisation)
        theta = np.mod(theta, 2 * np.pi)

    return theta, history


def snr_gain_db(h_d: complex, h_1: np.ndarray, h_2: np.ndarray) -> float:
    """
    Gain de SNR (en dB) apporté par l'optimisation de phase de la RIS
    par rapport au lien direct seul (RIS "éteinte" / absente).
    """
    theta_opt = optimal_phase_alignment(h_d, h_1, h_2)
    h_eq = combined_channel(h_d, h_1, h_2, theta_opt)
    gain_direct = np.abs(h_d) ** 2
    gain_ris = np.abs(h_eq) ** 2
    return 10 * np.log10(gain_ris / gain_direct)
