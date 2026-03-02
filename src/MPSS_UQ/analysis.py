# -*- coding: utf-8 -*-
import numpy as np

from MPSS_UQ.aerosol import BOLTZMANN_CONSTANT



# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summarize_samples(samples, coverage=95, use_mean=False):
    """
    Summarize posterior samples of a derived quantity.

    The sample axis is always axis 1.  For a single measurement,
    the input shape should be (1, n_samples) or (1, n_samples, k).
    Leading dimensions of size 1 are squeezed from the output.

    Parameters
    ----------
    samples : ndarray
        Shape (n_meas, n_samples) or (n_meas, n_samples, k).
    coverage : float
        Credible mass in percent, in (0, 100).
    use_mean : bool
        If True, report the mean; otherwise the median.

    Returns
    -------
    center : ndarray
    ci_lower : ndarray
    ci_upper : ndarray
    """
    if not (0 < coverage < 100):
        raise ValueError("coverage must be in (0, 100)")

    if use_mean:
        center = np.mean(samples, axis=1)
    else:
        center = np.median(samples, axis=1)

    work = np.moveaxis(samples, 1, -1)
    shape = work.shape[:-1]
    flat = work.reshape(-1, work.shape[-1])

    ci_lower, ci_upper = highest_density_interval(flat, coverage / 100.0)
    ci_lower = ci_lower.reshape(shape)
    ci_upper = ci_upper.reshape(shape)

    # Squeeze leading dimension if single measurement
    if center.shape[0] == 1:
        center = center.squeeze(axis=0)
        ci_lower = ci_lower.squeeze(axis=0)
        ci_upper = ci_upper.squeeze(axis=0)

    return center, ci_lower, ci_upper



# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def highest_density_interval(samples, percentage):
    """Calculate the highest density interval (HDI) for a given percentage.

    Parameters
    ----------
    samples : ndarray, shape (n,) or (m, n)
        1-D array of samples, or 2-D array where each row is an independent
        set of samples.
    percentage : float
        Credible mass in [0, 1].

    Returns
    -------
    hdi_low : ndarray, shape () or (m,)
        Lower bound(s) of the HDI.
    hdi_high : ndarray, shape () or (m,)
        Upper bound(s) of the HDI.

    For 1-D input, returns two scalars (as 0-d arrays).
    For 2-D input, returns two arrays of length m.
    """
    if np.isclose(percentage, 1.0):
        if samples.ndim == 1:
            return samples.min(), samples.max()
        return samples.min(axis=1), samples.max(axis=1)

    sorted_ = np.sort(samples, axis=-1)
    n = sorted_.shape[-1]
    n_samples = int(percentage * n)

    widths = sorted_[..., n_samples:] - sorted_[..., :n - n_samples]
    idx = np.argmin(widths, axis=-1)

    if samples.ndim == 1:
        return np.array([sorted_[idx], sorted_[idx + n_samples]])

    rows = np.arange(sorted_.shape[0])
    return sorted_[rows, idx], sorted_[rows, idx + n_samples]



# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def relative_hdi_width(median, ci_lower, ci_upper, eps=1e-4):
    """
    Relative width of the highest density interval.

    Defined as (ci_upper - ci_lower) / (median + eps).  The small constant
    *eps* prevents division by zero in size bins where the posterior median
    is near zero.

    Parameters
    ----------
    median : ndarray
        Posterior median, shape (n_bins,) or (n_measurements, n_bins).
    ci_lower : ndarray
        Lower bound of the HDI, same shape as *median*.
    ci_upper : ndarray
        Upper bound of the HDI, same shape as *median*.
    eps : float, optional
        Regularisation constant (default 1e-4).

    Returns
    -------
    rel_width : ndarray
        Same shape as the inputs.
    """
    return (ci_upper - ci_lower) / (median + eps)


def total_concentration(psd):
    """
    Total particle number concentration, summed over all size bins.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
        Particle size distribution as number concentrations per bin.
        A single PSD vector or a batch of samples (e.g., as returned
        by :meth:`PSDPosterior.get_samples`).

    Returns
    -------
    Ntot : scalar or ndarray, shape (n_samples,)
        Total concentration.  Scalar when the input is 1-D, array
        when the input is 2-D.
    """
    return np.sum(psd, axis=-1)


def concentration_in_range(psd, d_m, d_lo, d_hi):
    """
    Number concentration in a diameter sub-range.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)
        Bin center diameters.
    d_lo, d_hi : float
        Lower and upper diameter bounds.

    Returns
    -------
    N : scalar or ndarray, shape (n_samples,)
    """
    mask = (d_m >= d_lo) & (d_m < d_hi)
    return np.sum(psd[..., mask], axis=-1)


def geometric_mean_diameter(psd, d_m):
    """
    Geometric mean diameter of the particle size distribution.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    d_g : scalar or ndarray, shape (n_samples,)
    """
    ln_d = np.log(d_m)
    Ntot = np.sum(psd, axis=-1)
    return np.exp(np.sum(psd * ln_d, axis=-1) / Ntot)


def geometric_std(psd, d_m):
    """
    Geometric standard deviation of the particle size distribution.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    sigma_g : scalar or ndarray, shape (n_samples,)
    """
    ln_d = np.log(d_m)
    Ntot = np.sum(psd, axis=-1)
    ln_dg = np.sum(psd * ln_d, axis=-1) / Ntot
    if psd.ndim > 1:
        ln_dg = ln_dg[..., np.newaxis]
    variance = np.sum(psd * (ln_d - ln_dg)**2, axis=-1) / Ntot
    return np.exp(np.sqrt(variance))


def mode_diameter(psd, d_m):
    """
    Mode diameter: diameter at peak concentration.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    d_mode : scalar or ndarray, shape (n_samples,)
    """
    idx = np.argmax(psd, axis=-1)
    return d_m[idx]


def median_diameter(psd, d_m):
    """
    Count median diameter: diameter below which 50 % of total
    number concentration lies.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    d_50 : scalar or ndarray, shape (n_samples,)
    """
    # cumulative sum along bins
    cumsum = np.cumsum(psd, axis=-1)

    # half concentration per sample
    half = np.sum(psd, axis=-1, keepdims=True) / 2.0

    # mask: where cumulative sum exceeds half
    mask = cumsum >= half

    # argmax along bins gives the first True (i.e. median bin)
    idx = np.argmax(mask, axis=-1)

    return d_m[idx]



def surface_area_concentration(psd, d_m):
    """
    Total surface area concentration assuming spherical particles.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    S : scalar or ndarray, shape (n_samples,)
        Surface area concentration (um^2 cm^-3).
    """
    # multiply by 1e12 to convert m^2 -> um^2
    return np.pi * np.sum(psd * d_m**2, axis=-1) * 1e12


def volume_concentration(psd, d_m):
    """
    Total volume concentration assuming spherical particles.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    V : scalar or ndarray, shape (n_samples,)
        Volume concentration (um^3 cm^-3).
    """
    # multiply by 1e18 to convert m^3 -> um^3
    return (np.pi / 6.0) * np.sum(psd * d_m**3, axis=-1) * 1e18


def effective_diameter(psd, d_m):
    """
    Effective diameter: ratio of third to second moment.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
    d_m : ndarray, shape (n_bins,)

    Returns
    -------
    d_eff : scalar or ndarray, shape (n_samples,)
    """
    return np.sum(psd * d_m**3, axis=-1) / np.sum(psd * d_m**2, axis=-1)


def condensation_sink(psd, d_m, temperature=293.15, pressure=101325.0, alpha=1.0):
    """
    An example calculation of condensation sink for sulfuric acid vapour.

    The input PSD is expected in cm^-3.  The conversion to SI is
    handled internally.

    Parameters
    ----------
    psd : ndarray, shape (n_bins,) or (n_samples, n_bins)
        Number concentration per size bin (cm^-3).
    d_m : ndarray, shape (n_bins,)
        Bin center diameters (m).
    temperature : float
        Ambient temperature (K).  Default 293.15 K (20 °C).
    pressure : float
        Ambient pressure (Pa).  Default 101325 Pa (1 atm).
    alpha : float
        Accommodation coefficient.  Default 1.0.

    Returns
    -------
    CS : scalar or ndarray, shape (n_samples,)
        Condensation sink (s^-1).
    """
    # Convert cm^-3 -> m^-3
    psd_si = psd * 1e6

    # Molecular properties of H2SO4
    M_h2so4 = 98.08e-3                     # molar mass (kg/mol)
    m_v = M_h2so4 / 6.02214076e23          # mass of one molecule (kg)

    # Diffusion coefficient: Hanson & Eisele (2000) scaling
    D_ref = 0.74e-5                         # m^2/s at 273.15 K, 101325 Pa
    D_v = D_ref * (temperature / 273.15)**1.75 * (101325.0 / pressure)

    # Mean thermal speed and mean free path
    c_v = np.sqrt(8.0 * BOLTZMANN_CONSTANT * temperature / (np.pi * m_v))
    lambda_v = 3.0 * D_v / c_v

    # Fuchs-Sutugin transition regime correction
    Kn = 2.0 * lambda_v / d_m
    beta = (1.0 + Kn) / (
        1.0 + (4.0 / (3.0 * alpha) + 0.377) * Kn
        + (4.0 / (3.0 * alpha)) * Kn**2
    )

    return 2.0 * np.pi * D_v * np.sum(psd_si * beta * d_m, axis=-1)
