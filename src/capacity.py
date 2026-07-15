"""
capacity.py
-----------
Calcul de la capacité de Shannon pour les configurations
SISO / SIMO / MISO / MIMO, avec CSI parfaite au récepteur (et,
quand pertinent, à l'émetteur).

Hypothèses classiques (Telatar, 1999 ; Tse & Viswanath, 2005) :

- SISO : C = log2(1 + SNR * |h|^2)

- SIMO (combinaison à gain maximal, MRC) :
      C = log2(1 + SNR * ||h||^2)
  Le MRC est optimal en présence de bruit blanc et maximise le
  SNR de sortie -> gain de diversité de réception d'ordre Nr.

- MISO (précodage MRT, CSIT parfaite, contrainte de puissance totale) :
      C = log2(1 + SNR * ||h||^2)
  Même expression que SIMO par dualité (le MRT aligne la puissance
  d'émission sur la direction du canal).

- MIMO (répartition égale de puissance sur les antennes d'émission,
  pas de CSIT — cas le plus courant en pratique) :
      C = log2 det( I_Nr + (SNR/Nt) * H H^H )
  Avec waterfilling optimal (CSIT parfaite) on peut faire mieux,
  la fonction correspondante est fournie séparément.
"""

import numpy as np


def snr_linear(snr_db: np.ndarray) -> np.ndarray:
    """Convertit un SNR en dB vers l'échelle linéaire."""
    return 10.0 ** (np.asarray(snr_db) / 10.0)


def capacity_siso(h: complex, snr_lin: float) -> float:
    """Capacité SISO (bits/s/Hz)."""
    return np.log2(1 + snr_lin * np.abs(h) ** 2)


def capacity_simo(h: np.ndarray, snr_lin: float) -> float:
    """
    Capacité SIMO avec combinaison MRC.
    h : vecteur (Nr,) ou (Nr,1)
    """
    h = np.asarray(h).flatten()
    gain = np.sum(np.abs(h) ** 2)
    return np.log2(1 + snr_lin * gain)


def capacity_miso(h: np.ndarray, snr_lin: float) -> float:
    """
    Capacité MISO avec précodage MRT (CSIT parfaite).
    h : vecteur (1,Nt) ou (Nt,)
    """
    h = np.asarray(h).flatten()
    gain = np.sum(np.abs(h) ** 2)
    return np.log2(1 + snr_lin * gain)


def capacity_mimo_equal_power(H: np.ndarray, snr_lin: float) -> float:
    """
    Capacité MIMO à répartition égale de puissance sur les antennes
    d'émission (pas de CSIT) :
        C = log2 det( I_Nr + (SNR/Nt) * H H^H )
    """
    nr, nt = H.shape
    HHh = H @ H.conj().T
    M = np.eye(nr) + (snr_lin / nt) * HHh
    sign, logdet = np.linalg.slogdet(M)
    return logdet / np.log(2)


def capacity_mimo_waterfilling(H: np.ndarray, snr_lin: float) -> float:
    """
    Capacité MIMO avec allocation de puissance optimale (waterfilling)
    sur les valeurs propres de H^H H, sous contrainte de puissance
    totale = snr_lin (CSIT parfaite).
        C = sum_i log2(1 + mu * lambda_i)+   avec  sum_i p_i = snr_lin
    """
    # Valeurs singulières de H -> valeurs propres de H^H H
    s = np.linalg.svd(H, compute_uv=False)
    gains = s ** 2  # lambda_i
    gains = gains[gains > 1e-12]

    if len(gains) == 0:
        return 0.0

    # Recherche du niveau d'eau (water level) mu par bissection
    inv_gains = 1.0 / gains

    def total_power(mu):
        p = np.maximum(mu - inv_gains, 0.0)
        return np.sum(p)

    lo, hi = 0.0, (snr_lin + np.max(inv_gains)) * 10 + 10
    for _ in range(200):
        mid = (lo + hi) / 2
        if total_power(mid) < snr_lin:
            lo = mid
        else:
            hi = mid
    mu = (lo + hi) / 2

    p = np.maximum(mu - inv_gains, 0.0)
    active = p > 0
    C = np.sum(np.log2(1 + p[active] * gains[active]))
    return C
