"""
channels.py
-----------
Génération de canaux à évanouissement de Rayleigh (i.i.d. CN(0,1))
pour les configurations SISO / SIMO / MISO / MIMO, ainsi que le
modèle de canal en cascade pour un système assisté par RIS
(Reconfigurable Intelligent Surface).

Convention : y = H x + n, avec H de dimension (Nr x Nt).
    - SISO : Nr = 1, Nt = 1
    - SIMO : Nr > 1, Nt = 1   (diversité de réception)
    - MISO : Nr = 1, Nt > 1  (diversité d'émission)
    - MIMO : Nr > 1, Nt > 1

Toutes les fonctions renvoient des matrices numpy complexes.
"""

import numpy as np


def rayleigh_channel(nr: int, nt: int, rng: np.random.Generator) -> np.ndarray:
    """
    Génère une réalisation de canal MIMO à évanouissement de Rayleigh,
    i.i.d. CN(0,1) par coefficient (gain de puissance moyen = 1).

    Parameters
    ----------
    nr : nombre d'antennes de réception
    nt : nombre d'antennes d'émission
    rng : générateur numpy (pour reproductibilité)

    Returns
    -------
    H : ndarray complexe (nr, nt)
    """
    real = rng.standard_normal((nr, nt))
    imag = rng.standard_normal((nr, nt))
    return (real + 1j * imag) / np.sqrt(2.0)


def siso_channel(rng: np.random.Generator) -> np.ndarray:
    """Canal SISO : scalaire complexe CN(0,1)."""
    return rayleigh_channel(1, 1, rng)


def simo_channel(nr: int, rng: np.random.Generator) -> np.ndarray:
    """Canal SIMO : vecteur (nr, 1)."""
    return rayleigh_channel(nr, 1, rng)


def miso_channel(nt: int, rng: np.random.Generator) -> np.ndarray:
    """Canal MISO : vecteur (1, nt)."""
    return rayleigh_channel(1, nt, rng)


def mimo_channel(nr: int, nt: int, rng: np.random.Generator) -> np.ndarray:
    """Canal MIMO : matrice (nr, nt)."""
    return rayleigh_channel(nr, nt, rng)


def ris_cascaded_channel(
    n_ris: int,
    rng: np.random.Generator,
    rician_k_direct: float = 0.0,
):
    """
    Modèle de canal pour une liaison SISO assistée par une RIS à N
    éléments réfléchissants.

    Trois tronçons :
        h_d  : lien direct BS -> User          (scalaire, souvent obstrué)
        h_1  : lien BS -> RIS                  (vecteur N x 1)
        h_2  : lien RIS -> User                (vecteur N x 1)

    Le lien direct peut être modélisé avec un facteur de Rice K pour
    représenter une composante en visibilité directe partielle
    (K=0 -> Rayleigh pur / lien direct bloqué en moyenne).

    Returns
    -------
    h_d : complex scalar
    h_1 : ndarray (n_ris,)
    h_2 : ndarray (n_ris,)
    """
    # Lien BS->RIS et RIS->User : Rayleigh i.i.d.
    h_1 = rayleigh_channel(n_ris, 1, rng).flatten()
    h_2 = rayleigh_channel(n_ris, 1, rng).flatten()

    # Lien direct : Rice(K) pour permettre un mélange LOS/NLOS
    los = np.sqrt(rician_k_direct / (rician_k_direct + 1))
    nlos_std = np.sqrt(1 / (2 * (rician_k_direct + 1)))
    h_d = los + nlos_std * (rng.standard_normal() + 1j * rng.standard_normal())

    return h_d, h_1, h_2
