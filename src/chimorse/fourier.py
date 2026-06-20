"""
fourier.py
----------
Construction and evaluation of the chi-psi Fourier expansion basis
used for the angular dependence of the anisotropic Morse parameters.
"""

import numpy as np

# ----------------------------------------------------------------------

def count_fourier_coeffs(h_chi, h_psi, symm_chi, screw_step, after_symm=True):
    """Number of Fourier coefficients for the given harmonic orders, before or after screw-symmetrization."""
    if after_symm:
        if symm_chi:
            return 1 + h_chi + 2*h_psi + 2*h_chi*h_psi
        else:
            return 1 + 2*h_chi + 2*h_psi + 4*h_chi*h_psi
    else:
        k0 = int(round(360 / (2 * screw_step)))
        h_psi *= k0
        if symm_chi:
            return 1 + h_chi + 2*h_psi + 2*h_chi*h_psi
        else:
            return 1 + 2*h_chi + 2*h_psi + 4*h_chi*h_psi

# ----------------------------------------------------------------------

def create_matrix_lsqt_2d(h_chi, h_psi, chi_rad, psi_rad, symm_chi, screw_step):
    """Build the least-squares design matrix (and term labels) for the chi-psi Fourier expansion."""
    chi_rad = np.asarray(chi_rad, dtype=float)
    psi_rad = np.asarray(psi_rad, dtype=float)

    if chi_rad.shape != psi_rad.shape:
        raise ValueError("chi_rad and psi_rad must have same shape")

    n_samples = len(chi_rad)

    terms = []
    labels = []

    # CONSTANT TERM
    terms.append(np.ones(n_samples))
    labels.append("1")

    k0 = int(round(360 / (2 * screw_step)))

    # pure chi terms
    if symm_chi:

        for m in range(1, h_chi + 1):

            terms.append(np.cos(m * chi_rad))
            labels.append(f"cos({m}χ)")

    else:

        for m in range(1, h_chi + 1):

            terms.append(np.cos(m * chi_rad))
            labels.append(f"cos({m}χ)")

            terms.append(np.sin(m * chi_rad))
            labels.append(f"sin({m}χ)")

    # pure psi terms
    for j in range(1, h_psi + 1):

        n = k0 * j

        terms.append(np.cos(n * psi_rad))
        labels.append(f"cos({n}ψ)")

        terms.append(np.sin(n * psi_rad))
        labels.append(f"sin({n}ψ)")

    # coupled terms
    for m in range(1, h_chi + 1):

        cos_mchi = np.cos(m * chi_rad)

        if not symm_chi:
            sin_mchi = np.sin(m * chi_rad)

        for j in range(1, h_psi + 1):

            n = k0 * j

            cos_npsi = np.cos(n * psi_rad)
            sin_npsi = np.sin(n * psi_rad)

            if symm_chi:

                terms.append(cos_mchi * cos_npsi)
                labels.append(f"cos({m}χ)cos({n}ψ)")

                terms.append(cos_mchi * sin_npsi)
                labels.append(f"cos({m}χ)sin({n}ψ)")

            else:

                terms.append(cos_mchi * cos_npsi)
                labels.append(f"cos({m}χ)cos({n}ψ)")

                terms.append(cos_mchi * sin_npsi)
                labels.append(f"cos({m}χ)sin({n}ψ)")

                terms.append(sin_mchi * cos_npsi)
                labels.append(f"sin({m}χ)cos({n}ψ)")

                terms.append(sin_mchi * sin_npsi)
                labels.append(f"sin({m}χ)sin({n}ψ)")

    A = np.column_stack(terms)

    return A, labels

# ----------------------------------------------------------------------

def create_matrix_lsqt_2d_sum(h_chi, h_psi, chi, psi, symm_chi, screw_step):
    """Build a least-squares design matrix using the cos/sin(m*chi + n*psi) basis (sum-angle form)."""
    num_samples = len(chi)
    terms = [np.ones(num_samples)]

    for m in range(h_chi + 1):
        for n in range(-h_psi, h_psi + 1):
            if m == 0 and n <= 0:
                continue

            arg = m * chi + n * psi
            terms.append(np.cos(arg))
            terms.append(np.sin(arg))

    A = np.array(terms).T
    return A

# ----------------------------------------------------------------------

def fit_fourier_rmse(chi_rad, psi_rad, target, h_chi, h_psi, symm_chi, screw_step):
    """Fit target via least squares on the chi-psi Fourier basis and return the RMSE."""
    A, labels = create_matrix_lsqt_2d(
        h_chi=h_chi,
        h_psi=h_psi,
        chi_rad=chi_rad,
        psi_rad=psi_rad,
        symm_chi=symm_chi,
        screw_step=screw_step
    )
    coeff, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    pred = A @ coeff
    rmse = np.sqrt(np.mean((target - pred)**2))
    return rmse

# ----------------------------------------------------------------------

def compute_harmonic_rmse_grid(chi_rad, psi_rad, target, h_chi_range, h_psi_max,
                               symm_chi, screw_step):
    """Return the RMSE grid of fit_fourier_rmse over all (h_chi, h_psi) combinations in the given ranges."""
    h_chi_min, h_chi_max = h_chi_range
    rmse_grid = np.zeros((h_chi_max - h_chi_min + 1, h_psi_max))

    for i, h_chi in enumerate(range(h_chi_min, h_chi_max + 1)):
        for h_psi in range(1, h_psi_max + 1):
            rmse_grid[i, h_psi - 1] = fit_fourier_rmse(
                chi_rad, psi_rad, target,
                h_chi=h_chi,
                h_psi=h_psi,
                symm_chi=symm_chi,
                screw_step=screw_step
            )

    return rmse_grid
