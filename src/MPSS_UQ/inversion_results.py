# -*- coding: utf-8 -*-

"""
Classes for storing and processing inversion results from MPSS data.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from joblib import Parallel, delayed

from tqdm import tqdm

from MPSS_UQ.aerosol import BOLTZMANN_CONSTANT



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



class PSDPosterior:
    """
    Posterior distribution of a particle size distribution from a single measurement.

    Supports specifying the posterior with
        1) a log10-Gaussian approximation (MAP + covariance),
        2) samples from the posterior,
    but only one of them at a time.
    
    Computes the credible intervals on demand.
    
    Parameters
    ----------
    d_m_full : the diameters of the whole inverted PSD range
    sl_measured : slice, defaults to slice(None), the whole range, if not provided
        A slice indicating the start and stop indices of d_m_full that corresponds to the
        measured size range.
    post_mean_log10 : the posterior mean (in log10)
    post_covL_log10 : the Cholesky factor of the posterior covariance (in log10)
    post_samples : posterior samples
    ion_property_samples : array of floats or None, optional
        Values of the positive and negative ion mobility and ion ratio at each post_sample.
    reporting_range : 'measured' or 'full'
        The size range considered for posterior summaries, can be changed later.
        'measured' takes the shortest size interval in terms of inversion bins d_m_full such that
        the measured size range is covered. 'full' uses the whole inverted size range.
    """
    
    
    def __init__(self,
                 d_m_full,
                 sl_measured : slice = slice(None),
                 post_mean_log10=None,
                 post_covL_log10=None,
                 post_samples=None,
                 ion_property_samples=None,
                 reporting_range='measured',
                 ):
        
        # Particle size vector
        self._d_m_full = d_m_full
        
        # Helper for plotting the results (assume the bin widths stay constant)
        self.binwidth = np.log10(self._d_m_full[1]) - np.log10(self._d_m_full[0])
        
        # Check the inputs
        input1 = post_mean_log10 is not None and post_covL_log10 is not None
        input2 = post_samples is not None
        
        given_inputs = np.sum([input1, input2])
        
        if given_inputs == 0:
            raise ValueError('No valid input provided. Specify one of the two input types.')
        elif given_inputs > 1:
            raise ValueError('Multiple inputs provided. Specify only one input.')
        
        if input1:
            self.post_mean_log10 = post_mean_log10
            self.post_covL_log10 = post_covL_log10
            self.input_mode = 'gaussian-log10'
        
        elif input2:
            self.post_samples = post_samples.astype(np.float32, copy=False)
            self.input_mode = 'samples'
            if ion_property_samples is not None:
                self.ion_property_samples = ion_property_samples.astype(np.float32, copy=False)
        
        # Store the slice that was used to get the reported size range from the full inverted
        # size range
        self.sl_measured = sl_measured
        
        self.set_reporting_range(reporting_range)
        
        # Set up rng
        self.rng = np.random.default_rng()
    
    
    def set_reporting_range(self, reporting_range : str):
        
        if reporting_range == 'measured':
            self.sl = self.sl_measured
            self.d_m = self._d_m_full[self.sl]
            self.reporting_range = 'measured'
        
        elif reporting_range == 'full':
            self.sl = slice(0, len(self._d_m_full))
            self.d_m = self._d_m_full
            self.reporting_range = 'full'
        
        else:
            raise ValueError("Unknown reporting range. Use 'measured' or 'full'.")
    
    
    def _summary_from_covariance_log10(self, CI):
        """ Calculate the posterior median and highest density interval at credible level CI
        for the lognormal density in N.
        The HDI doesn't have an analytical expression, it must be solved numerically. For N,
        the HDI is multiplicatively symmetric around the mode (but not the median, which we
        report as the point estimate!). If 10^c is the posterior mode, then the lower HDI limit
        is 10^c / r and the upper limit is 10^c * r, where r = 10^\delta. We find \delta by
        solving a scalar equation, corresponding to finding the root of the function
        'mass_equation' below.
        """
        
        # Point estimate in N: posterior median
        post_median = 10**(self.post_mean_log10[self.sl])
        
        #   CI is the desired mass in percent, e.g. 95 -> alpha = 0.95
        alpha = CI / 100.0
        
        # Marginal stds in x-space (log10): s_i
        s = np.sqrt(np.sum(self.post_covL_log10[self.sl, :]**2, axis=1))
        mu10 = self.post_mean_log10[self.sl]
        ln10 = np.log(10.0)
        c = mu10 - ln10 * s**2  # 10^c is the mode for N
    
        CI_lower = np.zeros_like(mu10)
        CI_upper = np.zeros_like(mu10)

        def mass_equation(delta, si):
            # f(delta) = P(c-delta <= x <= c+delta) - alpha
            # where x ~ Normal(mu10, si^2)
            a1 = (delta - ln10 * si**2) / si
            a2 = (-delta - ln10 * si**2) / si
            return norm.cdf(a1) - norm.cdf(a2) - alpha
    
        for i, si in enumerate(s):
            # brentq needs a bracket [lo, hi] where f(lo) <= 0 and f(hi) >= 0.
            # At delta = 0, mass is 0, so mass_equation(0) = -alpha < 0.
            lo = 0.0
            
            # Pick a conservative upper bound for delta (in decades).
            # If too small, we expand it until the bracket is valid.
            hi = max(1.0, 10.0 * si)
    
            # Ensure f(hi) >= 0 so the root is bracketed.
            while mass_equation(hi, si) < 0:
                hi *= 2.0
            
            # Solve for delta such that the interval contains exactly alpha mass.
            delta = brentq(mass_equation, lo, hi, args=(si,))
            
            # Form the HDI endpoints in x-space
            xL = c[i] - delta
            xU = c[i] + delta
            
            # Map to N-space
            CI_lower[i] = 10.0**xL
            CI_upper[i] = 10.0**xU
        
        return post_median, CI_lower, CI_upper
    
    
    def _summary_from_samples(self, CI):
        """ Calculate the posterior median and highest density interval at credible level CI
        from the posterior samples of N.
        """
        samples = self.post_samples[:, self.sl]
        post_median = np.median(samples, axis=0)
        hdi_low, hdi_high = highest_density_interval(samples.T, CI / 100.0)
        return post_median, hdi_low, hdi_high
            
    
    def variance(self):
        """ Return the posterior variance.
        """
        if self.input_mode == 'gaussian-log10':
            return np.diag(self.post_covL_log10[self.sl, self.sl])**2
        elif self.input_mode == 'samples':
            return np.var(np.log10(self.post_samples[:, self.sl]), axis=0)
    
    
    def summary(self, coverage=95):
        """
        Returns the posterior median and shortest credible intervals.
        Input:
            coverage - the percentage of the posterior mass the credible interval should cover.
        """
        if coverage <= 0 or coverage >= 100:
            raise ValueError('Invalid value for posterior coverage. It should be in (0, 100).')
        
        if self.input_mode == 'gaussian-log10':
            return self._summary_from_covariance_log10(coverage)
        elif self.input_mode == 'samples':
            return self._summary_from_samples(coverage)
    
    
    def propagate_to(self, func, *args, num=None, **kwargs):
        """
        Propagate posterior uncertainty through a function. This is done by drawing samples from
        the posterior of the PSD, and putting those samples through *func*.
    
        The PSD samples are passed as the first argument to *func*.
        Any additional arguments are forwarded.
    
        Parameters
        ----------
        func : callable
            A function whose first argument is the PSD, shape
            (n_samples, n_bins).  Additional parameters (e.g.,
            diameter vector, physical constants) are passed via
            *args* and **kwargs.
        *args
            Extra positional arguments forwarded to *func*.
        num : int or None
            Number of posterior samples to use.
        **kwargs
            Extra keyword arguments forwarded to *func*.
    
        Returns
        -------
        result : ndarray, shape (1, n_samples, ...)
        """
        samples = self.get_samples(num=num)
        return func(samples, *args, **kwargs)[np.newaxis, ...]

    
    
    def get_samples(self, num=None):
        """
        Return posterior samples in linear (N) space, shape (num, n_bins).
    
        For input_mode='samples', returns stored samples (or a random
        subset if *num* < available samples).
        For input_mode='gaussian-log10', draws *num* samples from the
        Gaussian approximation.
    
        Parameters
        ----------
        num : int or None
            Number of samples.  For 'samples' mode, None returns all
            stored samples.  For 'gaussian-log10' mode, None defaults
            to 5000.
    
        Returns
        -------
        samples : ndarray, shape (num, n_bins)
            Posterior samples over the current reporting range.
        """
        if self.input_mode == 'samples':
            samples = self.post_samples[:, self.sl]
            if num is None or num == len(samples):
                return samples
            elif num < len(samples):
                idx = self.rng.choice(len(samples), num, replace=False)
                return samples[idx]
            else:
                raise ValueError(
                    f'Requested {num} samples but only {len(samples)} available.'
                    )
        
        elif self.input_mode == 'gaussian-log10':
            if num is None:
                num = 5000
             # Do 10^samples, but using np.exp (faster)
            return np.exp((self.post_mean_log10[self.sl, None]
                        + self.post_covL_log10[self.sl, self.sl]
                        @ self.rng.normal(loc=0.0, scale=1.0, size=(len(self.d_m), num))
                        ) * np.log(10)).T


class PSDPosteriorSeries:
    """
    Time series of posterior distributions of particle size distributions across
    multiple measurements.
    """

    def __init__(self, datetimes):
        self.datetimes = datetimes
        self._posteriors = [None] * len(datetimes)
    
    
    def assign_posterior(self, idx, posterior: PSDPosterior):
        self._posteriors[idx] = posterior
    
    
    def variance(self):
        """
        Returns an array of posterior variances for all posteriors.
        """
        num_posteriors = len(self._posteriors)
        num_d_m = self._posteriors[0].d_m.shape[0]
        variances = np.zeros((num_posteriors, num_d_m))
        
        for i in range(num_posteriors):
            variances[i] = self._posteriors[i].variance()
        
        return variances
    
    
    def summary(self, *args, n_jobs=-1, **kwargs):
        """
        Returns arrays of posterior median, lower CI, upper CI for all posteriors.
        
        Parameters
        ----------
        n_jobs : int
            Number of worker processes. -1 uses all cores.
        """
        num_posteriors = len(self._posteriors)
        num_d_m = self._posteriors[0].d_m.shape[0]
    
        posterior_medians = np.zeros((num_posteriors, num_d_m))
        CI_lower = np.zeros_like(posterior_medians)
        CI_upper = np.zeros_like(posterior_medians)
    
        def _process(i):
            med, lo, up = self._posteriors[i].summary(*args, **kwargs)
            posterior_medians[i] = med
            CI_lower[i] = lo
            CI_upper[i] = up
    
        Parallel(n_jobs=n_jobs, backend="threading", require="sharedmem")(
            delayed(_process)(i)
            for i in tqdm(range(num_posteriors), desc="Calculating posterior summaries")
        )

        return posterior_medians, CI_lower, CI_upper
    
    
    def propagate_to(self, func, *args, num=None, **kwargs):
        """
        Propagate posterior uncertainty through a function for each
        measurement.
    
        Parameters
        ----------
        func : callable
            A function whose first argument is the PSD.
        *args
            Extra positional arguments forwarded to *func*.
        num : int or None
            Number of posterior samples per measurement.
        **kwargs
            Extra keyword arguments forwarded to *func*.
    
        Returns
        -------
        result : ndarray, shape (n_measurements, n_samples, ...)
        """
        return np.array([
            posterior.propagate_to(func, *args, num=num, **kwargs).squeeze(axis=0)
            for posterior in self._posteriors
            ])
    
    
    def set_reporting_range(self, reporting_range : str):
        for i in range(len(self._posteriors)):
            self._posteriors[i].set_reporting_range(reporting_range)
    
    
    def _return_subset(self, indices):
            """
            Build a new PSDPosteriorSeries with a subset of rows (as a view when possible).
            `indices` can be a slice, a list/ndarray of ints, or a boolean mask.
            """
            new_datetimes = self.datetimes[indices]
            
            # Reuse the same PSDPosterior objects
            if isinstance(indices, (slice, list, np.ndarray)):
                new_posteriors = (np.array(self._posteriors, dtype=object)[indices]).tolist()
            else:
                # Fallback for other index types
                new_posteriors = [self._posteriors[indices]]
            
            new_series = PSDPosteriorSeries(new_datetimes)
            new_series._posteriors = new_posteriors
            return new_series
    
    
    def between_times(self, start, end, closed="both"):
        """
        Return a new PSDPosteriorSeries restricted to [start, end] using the chosen
        boundary convention.
        
        Parameters
        ----------
        start, end : datetime-like (np.datetime64, pandas Timestamp, str parsable to datetime64)
        closed : {'left', 'right', 'both', 'neither'}
            - 'both'   : start <= t <= end
            - 'left'   : start <= t <  end
            - 'right'  : start <  t <= end
            - 'neither': start <  t <  end
        """
        dt = self.datetimes
        # Ensure numpy datetime64 for consistent comparisons
        if not np.issubdtype(dt.dtype, np.datetime64):
            dt = dt.astype("datetime64[ns]")
        
        start = np.datetime64(start)
        end   = np.datetime64(end)
        
        mask_left  = (dt >= start) if closed in ("left", "both") else (dt > start)
        mask_right = (dt <= end)   if closed in ("right", "both") else (dt < end)
        mask = mask_left & mask_right
        
        return self._return_subset(mask)
    
    
    def __getitem__(self, idx):
        """
        - int -> PSDPosterior
        - slice / list / ndarray / boolean mask -> PSDPosteriorSeries
        """
        if isinstance(idx, (int, np.integer)):
            # Support negative indices (Python/NumPy semantics)
            if idx < 0:
                idx += len(self._posteriors)
            if idx < 0 or idx >= len(self._posteriors):
                raise IndexError("Index out of range.")
            return self._posteriors[idx]
        # For multi-select, return a sliced dataset
        return self._return_subset(idx)
    
    
    def __len__(self):
        return len(self._posteriors)
