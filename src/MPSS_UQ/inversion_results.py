# -*- coding: utf-8 -*-

"""
Classes for storing and processing inversion results from MPSS data.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from joblib import Parallel, delayed

from tqdm import tqdm



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


class InversionResult:
    """
    Represents the inversion result for a single measurement.

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
    
    
    def _postprocess_results_from_covariance_log10(self, CI):
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
    
    
    def _postprocess_results_from_samples(self, CI):
        """ Calculate the posterior median and highest density interval at credible level CI
        from the posterior samples of N.
        """
        samples = self.post_samples[:, self.sl]  # (5000, d_m)
        post_median = np.median(samples, axis=0)
        hdi_low, hdi_high = highest_density_interval(samples.T, CI / 100.0)  # (d_m, 2)
        return post_median, hdi_low, hdi_high
            
    
    def posterior_variance(self):
        """ Calculate (if needed) and return the posterior variance.
        """
        if self.input_mode == 'gaussian-log10':
            return np.diag(self.post_covL_log10[self.sl, self.sl])**2
        elif self.input_mode == 'samples':
            return np.var(np.log10(self.post_samples[:, self.sl]), axis=0)
    
    
    def posterior_summary(self, coverage=95):
        """
        Returns the posterior median and shortest credible intervals.
        Input:
            coverage - the percentage of the posterior mass the credible interval should cover.
        """
        if coverage <= 0 or coverage >= 100:
            raise ValueError('Invalid value for posterior coverage. It should be in (0, 100).')
        
        if self.input_mode == 'gaussian-log10':
            return self._postprocess_results_from_covariance_log10(coverage)
        elif self.input_mode == 'samples':
            return self._postprocess_results_from_samples(coverage)
    
    
    def Ntot_samples(self, n_samples=None):
        """
        Returns samples of total posterior particle count.
        Input:
            n_samples: number of Ntot samples to return
        """
        
        if self.input_mode == 'samples':
            if n_samples is None:
                return np.sum(self.post_samples[:, self.sl], axis=1)
            
            elif n_samples <= len(self.post_samples):
                sample_idxs = self.rng.choice(len(self.post_samples), n_samples, replace=False)
                return np.sum(self.post_samples[sample_idxs, self.sl], axis=1)
            
            else:
                raise ValueError(
                    f'Cannot request n={n_samples} Ntot samples because InversionResult has ' + 
                    f'only n={len(self.post_samples)} posterior samples.'
                    )
        
        elif self.input_mode == 'gaussian-log10':
            if n_samples is None:
                n_samples = 5000
            
            # We have to sample because of the nonlinear transformation
            post_samples_log10 = self.post_mean_log10[self.sl, None] \
                + self.post_covL_log10[self.sl, self.sl] @ self.rng.normal(
                loc=0.0, scale=1.0, size=(self.d_m.shape[0], n_samples)
                )
            # Do 10^samples, but using np.exp (faster)
            post_samples = np.exp(post_samples_log10 * np.log(10))
            Ntot_samples = np.sum(post_samples, axis=0)
            return Ntot_samples
    
    
    def draw_posterior_samples(self, num=1):
        ''' Draw posterior samples from the Gaussian approximation.
        Can only be used for the input mode 'gaussian-log10'.
        '''
        
        if self.input_mode == 'gaussian-log10':
            return np.exp((self.post_mean_log10[self.sl, None]
                        + self.post_covL_log10[self.sl, self.sl]
                        @ self.rng.normal(loc=0.0, scale=1.0, size=(len(self.d_m), num))
                        ) * np.log(10)).T
        
        else:
            raise ValueError("Posterior samples for input_mode=='samples' are found in " +
                             "result.samples")


class InversionDataset:
    """
    Container for inversion results across multiple measurements.
    """

    def __init__(self, datetimes):
        self.datetimes = datetimes
        self.results = [None] * len(datetimes)
    
    
    def assign_result(self, idx, result: InversionResult):
        self.results[idx] = result
    
    
    def posterior_variance(self):
        """
        Returns an array of posterior variances for all results.
        """
        num_results = len(self.results)
        num_d_m = self.results[0].d_m.shape[0]
        posterior_variances = np.zeros((num_results, num_d_m))
        
        for i in range(num_results):
            posterior_variances[i] = self.results[i].posterior_variance()
        
        return posterior_variances
    
    
    def posterior_summary(self, *args, n_jobs=-1, backend="loky", **kwargs):
        """
        Returns arrays of posterior median, lower CI, upper CI for all results.
        
        Parameters
        ----------
        n_jobs : int
            Number of worker processes. -1 uses all cores.
        backend : str
            "loky" uses processes (safe default). "threading" uses threads.
        """
        num_results = len(self.results)
        num_d_m = self.results[0].d_m.shape[0]
        
        def _summary_one(res, *args, **kwargs):
            # Helper so joblib can pickle the callable cleanly
            return res.posterior_summary(*args, **kwargs)
        
        iterator = tqdm(self.results, total=len(self.results),
                        desc='Calculating posterior summaries')
        # Run each result in parallel; order is preserved by joblib
        outs = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_summary_one)(res, *args, **kwargs) for res in iterator
            )
    
        # Unpack into preallocated arrays
        posterior_medians = np.zeros((num_results, num_d_m))
        CI_lower = np.zeros_like(posterior_medians)
        CI_upper = np.zeros_like(posterior_medians)
        for i, (med, lo, up) in enumerate(outs):
            posterior_medians[i] = med
            CI_lower[i] = lo
            CI_upper[i] = up
    
        return posterior_medians, CI_lower, CI_upper

    
    def Ntot_summary(self, coverage=95, n_samples=None, use_mean=False):
        """
        Compute per-measurement total particle count (Ntot) summaries across the dataset.
    
        For each InversionResult:
          1) draw Ntot samples via result.Ntot_samples(n_samples)
          2) report the center (median by default, or mean if use_mean=True)
          3) compute the highest-density interval (HDI) at the given coverage
    
        Parameters
        ----------
        coverage : float in (0, 100)
            Credible mass percentage for the interval (e.g. 95).
        n_samples : int or None
            Number of Ntot samples to draw per result.
            If None, uses the default behavior in InversionResult.Ntot_samples().
        use_mean : bool
            If True, report the mean instead of the median as the point estimate.
    
        Returns
        -------
        Ntots : (N,) ndarray
            Point estimates (median by default) of Ntot for each measurement.
        Ntots_CI_low : (N,) ndarray
            Lower HDI bound for each measurement.
        Ntots_CI_high : (N,) ndarray
            Upper HDI bound for each measurement.
        """
        if not (0 < coverage < 100):
            raise ValueError("coverage must be in (0, 100)")
        
        # Draw Ntot samples for each result
        all_Ntot = np.array([
            res.Ntot_samples(n_samples)
            for res in self.results
        ])
        
        Ntots = np.mean(all_Ntot, axis=1) if use_mean else np.median(all_Ntot, axis=1)
        hdi_low, hdi_high = highest_density_interval(all_Ntot, coverage / 100.0)
        
        return Ntots, hdi_low, hdi_high
    
    
    def set_reporting_range(self, reporting_range : str):
        for i in range(len(self.results)):
            self.results[i].set_reporting_range(reporting_range)
    
    
    def _return_subset(self, indices):
            """
            Build a new InversionDataset with a subset of rows (as a view when possible).
            `indices` can be a slice, a list/ndarray of ints, or a boolean mask.
            """
            new_datetimes = self.datetimes[indices]
            
            # Reuse the same InversionResult objects (no deep copy).
            # If you prefer isolation, you could clone here, but it's usually unnecessary.
            if isinstance(indices, (slice, list, np.ndarray)):
                new_results = (np.array(self.results, dtype=object)[indices]).tolist()
            else:
                # Fallback for other index types
                new_results = [self.results[indices]]
            
            new_ds = InversionDataset(new_datetimes)
            new_ds.results = new_results
            return new_ds
    
    
    def between_times(self, start, end, closed="both"):
        """
        Return a new InversionDataset restricted to [start, end] using the chosen
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
        - int -> InversionResult
        - slice / list / ndarray / boolean mask -> InversionDataset
        """
        if isinstance(idx, (int, np.integer)):
            # Support negative indices (Python/NumPy semantics)
            if idx < 0:
                idx += len(self.results)
            if idx < 0 or idx >= len(self.results):
                raise IndexError("Index out of range.")
            return self.results[idx]
        # For multi-select, return a sliced dataset
        return self._return_subset(idx)
