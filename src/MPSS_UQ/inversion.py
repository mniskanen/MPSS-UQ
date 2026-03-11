# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import toeplitz
from scipy.stats import norm
from typing import Literal, Tuple
import psutil
from joblib import Parallel, delayed

from tqdm import tqdm

from MPSS_UQ.particlesizers import MobilityParticleSizeSpectrometer
from MPSS_UQ.analysis import highest_density_interval


# Prevent the system from throttling down the CPU by giving any process that uses
# inversion methods a higher priority
import os
p = psutil.Process(os.getpid())
try:
    p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows constant
except AttributeError:
    # psutil.HIGH_PRIORITY_CLASS doesn't exist on non-Windows
    try:
        p.nice(-10)  # Linux/macOS: lower niceness = higher priority
    except Exception:
        pass  # silently ignore if not permitted



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
                 prior=None,
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
        
        self.prior = prior
        
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
        'mass_equation' below. The root-finding is vectorized over all components via bisection.
        """
		
        # Point estimate in N: posterior median
        post_median = 10.0 ** self.post_mean_log10[self.sl]
        
        alpha = CI / 100.0
        
        # Marginal stds in log10-space
        s = np.sqrt(self.variance())
        mu10 = self.post_mean_log10[self.sl]
        ln10 = np.log(10.0)
        c = mu10 - ln10 * s**2  # 10^c is the mode for N
        
        # --- Vectorized root-finding via bisection on delta (array) ---
        # mass_equation(delta, s) = Phi(a1) - Phi(a2) - alpha
        #   a1 = (delta - ln10*s^2) / s
        #   a2 = (-delta - ln10*s^2) / s
        
        # Initial bracket: lo = 0 everywhere, hi = max(1, 10*s)
        lo = np.zeros_like(s)
        hi = np.maximum(1.0, 10.0 * s)
        
        def mass_equation(delta):
            a1 = (delta - ln10 * s**2) / s
            a2 = (-delta - ln10 * s**2) / s
            return norm.cdf(a1) - norm.cdf(a2) - alpha
        
        # Widen hi where bracket is invalid (vectorized doubling)
        f_hi = mass_equation(hi)
        for _ in range(20):
            mask = f_hi < 0.0
            if not np.any(mask):
                break
            hi[mask] *= 2.0
            f_hi[mask] = mass_equation(hi)[mask]
        
        # Bisection: 60 iterations give ~18 digits of precision (2^-60 ≈ 1e-18)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = mass_equation(mid)
            neg = f_mid < 0.0
            lo = np.where(neg, mid, lo)
            hi = np.where(neg, hi, mid)
    
        delta = 0.5 * (lo + hi)
    
        # Map to N-space
        CI_lower = 10.0**(c - delta)
        CI_upper = 10.0**(c + delta)
    
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
            return np.sum(self.post_covL_log10[self.sl, :]**2, axis=1)
        elif self.input_mode == 'samples':
            return np.var(np.log10(self.post_samples[:, self.sl]), axis=0)
    
    
    def prior_to_posterior_ratio(self):
        """ Prior-to-posterior standard deviation ratio in log10(n) space.
        
        Returns values in [0, 1] where 0 = perfectly constrained
        and 1 = no information gained.
        """
        sigma_post = np.sqrt(self.variance())
        
        if self.prior is not None:
            # Assume here that the prior is the same for all bins
            sigma_prior = np.sqrt(self.prior['covariance'][0, 0])
        else:
            raise ValueError('No prior specified for this posterior.')
        
        return sigma_prior / sigma_post
    
    
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
                num = 500
            
            # Extract the rows of L corresponding to the reporting range.
            L_sub = self.post_covL_log10[self.sl, :]
            z = self.rng.normal(size=(L_sub.shape[1], num))
            
            # Compute samples in log10 space
            log10_samples = self.post_mean_log10[self.sl, None] + L_sub @ z
            
            # Convert to linear space using np.exp (faster)
            return np.exp(log10_samples * np.log(10)).T


class PSDPosteriorSeries:
    """
    Time series of posterior distributions of particle size distributions across
    multiple measurements.
    """

    def __init__(self, datetimes):
        self.datetimes = datetimes
        self._posteriors = [None] * len(datetimes)
        self.prior = None  # The same prior used for all measurements
    
    
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
    
    
    def prior_to_posterior_ratio(self):
        """ Returns arrays of the prior-to-posterior standard deviation ratios for all posteriors.
        """
        num_posteriors = len(self._posteriors)
        num_d_m = self._posteriors[0].d_m.shape[0]
        ratios = np.zeros((num_posteriors, num_d_m))
        
        for i in range(num_posteriors):
            ratios[i] = self._posteriors[i].prior_to_posterior_ratio()
        
        return ratios
    
    
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
        desc = f'Propagating posterior samples to {getattr(func, "__name__", repr(func))}'
        return np.array([
            posterior.propagate_to(func, *args, num=num, **kwargs).squeeze(axis=0)
            for posterior in tqdm(self._posteriors, desc=desc)
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



def invert_psd(
        sizer : MobilityParticleSizeSpectrometer,
        measurement,  # Measurement
        prior=None,   # If None, builds a default smoothness prior
        marginalize_ion_mobility=False,
        marginalize_ion_ratio=False,
        marginalization_grid : Literal['standard', 'fine'] = 'standard',
        num_samples=None,
        use_mcmc=False,
        ) -> PSDPosterior:
    ''' Estimate the particle size distribution (PSD) from an MPSS measurement.
    
    With no marginalization, the posterior is approximated by the Laplace approximation
    (MAP + covariance). Optionally, one can set use_mcmc to True which carries out
    MCMC sampling and returns the samples.
    
    Marginalization is carried over over a range of ion mobilities and ratios using
    the LYF model. When doing marginalization, the posterior is approximated as a mixture of
    Gaussians (Laplace approximations) and samples from the mixture are returned.
    
    marginalization_grid refers to how many mobility pairs are calculated when marginalizing.
    marginalization_grid='fine' gives better looking Ntot histogram plots.
    The posterior estimates and credible intervals are basically the same as with
    marginalization_grid='standard', but marginalization_grid='standard' is faster.
    
    num_samples is the number of posterior samples returned when using sampling methods.
    
    Returns a PSDPosterior containing either the Laplace approximation or posterior samples.
    '''
    
    # Calculate the measured reporting_range:
    eps = 1e-16
    lo = sizer.d_m_data.min() - eps
    hi = sizer.d_m_data.max() + eps
    
    i_start = np.searchsorted(sizer.d_m, lo, side='left')
    i_stop  = np.searchsorted(sizer.d_m, hi, side='right')
    
    # Take the bins just outside the range to make sure to show the whole measured range
    i_start = max(0, i_start - 1)
    i_stop = min(len(sizer.d_m), i_stop + 1)
        
    sl_measured = slice(i_start, i_stop)
    
    if prior is None:
        prior = smoothness_prior(sizer.d_m)
    
    if marginalize_ion_mobility is True or marginalize_ion_ratio is True:
        if use_mcmc is True:
            raise ValueError('Cannot carry out MCMC with posterior marginalization. ' +
                             'Set use_mcmc to False.')
        
        if num_samples is None:
            num_samples = 5000
        
        posterior_samples, ion_property_samples = Laplace_approximation_marginalize(
            sizer, prior, measurement,
            marginalize_ion_mobility,
            marginalize_ion_ratio,
            marginalization_grid=marginalization_grid,
            num_samples=num_samples,
            )
        return PSDPosterior(sizer.d_m,
                            sl_measured,
                            post_samples=posterior_samples,
                            ion_property_samples=ion_property_samples,
                            prior=prior,
                            )
    else:
        if use_mcmc:
            if num_samples is None:
                num_samples = 100000
                
            posterior_samples = run_mcmc(sizer, prior, measurement, num_samples=num_samples)
            return PSDPosterior(sizer.d_m,
                                sl_measured,
                                post_samples=posterior_samples,
                                prior=prior,
                                )
        
        else:
            # Laplace approximation
            MAP, post_cov_L = Laplace_approximation(sizer, prior, measurement)
            return PSDPosterior(sizer.d_m,
                                sl_measured,
                                post_mean_log10=MAP,
                                post_covL_log10=post_cov_L,
                                prior=prior,
                                )


def invert_dataset(
    sizer: MobilityParticleSizeSpectrometer,
    dataset,  # MeasurementDataset or a sequence of measurement-like objects
    *,
    prior=None,
    marginalize_ion_mobility: bool = False,
    marginalize_ion_ratio: bool = False,
    marginalization_grid: Literal['standard', 'fine'] = 'standard',
    use_mcmc: bool = False,
    num_samples: int | None = None,
    parallel: bool = False,
    backend: Literal['loky', 'multiprocessing', 'threading'] = 'loky',
    n_jobs: int | None = None,
    sort_for_cache: bool = True,
    progress: bool = True,
    ) -> PSDPosteriorSeries:
    """
    Invert an entire dataset of MPSS measurements (optionally in parallel)
    and return an PSDPosteriorSeries with per-measurement results.

    Parameters
    ----------
    sizer : MobilityParticleSizeSpectrometer
        A configured sizer instance. The function will set operating conditions
        per measurement; with backend='loky' each worker gets its own copy.
    dataset : MeasurementDataset | Sequence
        Iterable of measurements; must support __len__ and __getitem__ (0..n-1).
    prior : dict | None
        PSD prior. If None, uses the default smoothness_prior.
    marginalize_ion_mobility, marginalize_ion_ratio, marginalization_grid,
    use_mcmc, num_samples
        Passed through to invert_psd(...).
    parallel : bool
        If True, use joblib.Parallel with the given backend.
    backend : {'loky','multiprocessing','threading'}
        Default 'loky' copies objects (safe to mutate sizer in workers).
    n_jobs : int | None
        Number of worker processes/threads. Default = max(1, physical_cpus-1).
    sort_for_cache : bool
        If True, sort measurements by rounded (T, p) to improve cache locality
        in the sizer (see the @lru_cache usage in particlesizers.py). Results
        are reordered to match the input.
    progress : bool
        Show a progress bar.

    Returns
    -------
    PSDPosteriorSeries
        Holds results, datetimes, and supports downstream plotting/summary.
    """
    
    if prior is None:
        prior = smoothness_prior(sizer.d_m)
    
    # Build output container using input datetimes if available
    try:
        inv_dataset = PSDPosteriorSeries(dataset.datetimes)
    except AttributeError:
        inv_dataset = PSDPosteriorSeries(
            [getattr(dataset[i], "datetime", None) for i in range(len(dataset))]
            )
    
    # Store the prior for later access when analyzing results
    inv_dataset.prior = prior
    
    # Collect and optionally sort tasks
    measurements = [dataset[i] for i in range(len(dataset))]
    if sort_for_cache:
        # group by rounded temperature and pressure (mirrors your example)
        # rounding choices follow your current practice
        sortable = list(enumerate(measurements))
        sortable.sort(
            key=lambda x: (
                int(round(x[1].temperature / 2.0) * 2),       # 2 K buckets
                int(round(x[1].pressure * 1e2 / 25.0) * 25),  # 25 Pa buckets
            )
        )
    else:
        sortable = list(enumerate(measurements))

    # Worker
    def _solve_one(task: Tuple[int, object]):
        idx, meas = task
        # Set operating conditions per measurement
        sizer.set_operating_conditions(meas.temperature, meas.pressure)
        res = invert_psd(
            sizer, meas, prior=prior,
            marginalize_ion_mobility=marginalize_ion_mobility,
            marginalize_ion_ratio=marginalize_ion_ratio,
            marginalization_grid=marginalization_grid,
            num_samples=num_samples,
            use_mcmc=use_mcmc,
            )
        return idx, res

    # Decide concurrency
    if parallel:
        if n_jobs is None:
            n_cpus = psutil.cpu_count(logical=False) or 1
            n_jobs = max(1, n_cpus - 1)
        iterator = sortable
        if progress:
            iterator = tqdm(iterator, total=len(sortable), desc='Inverting dataset')
        
        # Important: the 'loky' backend makes copies (via pickling) of the objects each worker
        # needs, so each process receives its own copy of, for example, the DMPS and therefore
        # it is safe to mutate the MPSS inside 'invert_psd'. With a different backend this
        # may not be the case.
        # TODO: could do this in batches to help memory usage with huge datasets
        psd_posteriors = Parallel(n_jobs=n_jobs, backend=backend)(
            delayed(_solve_one)(task) for task in iterator
            )
        # Assign psd_posteriors to original indices
        for idx, post in psd_posteriors:
            inv_dataset.assign_posterior(idx, post)
    else:
        iterator = sortable
        if progress:
            iterator = tqdm(iterator, total=len(sortable), desc='Inverting dataset')
        for idx, meas in iterator:
            sizer.set_operating_conditions(meas.temperature, meas.pressure)
            res = invert_psd(
                sizer, meas, prior=prior,
                marginalize_ion_mobility=marginalize_ion_mobility,
                marginalize_ion_ratio=marginalize_ion_ratio,
                marginalization_grid=marginalization_grid,
                num_samples=num_samples,
                use_mcmc=use_mcmc,
            )
            inv_dataset.assign_posterior(idx, res)
    
    return inv_dataset


def log_post(vals, sizer, L_noise, prior, y_meas):
    ''' Compute the logarithm of the (non-normalized) posterior. '''
    
    return -0.5 * np.linalg.norm(L_noise * (y_meas - sizer.forward_model(vals)))**2 \
        - 0.5 * np.linalg.norm(prior['L'] @ (vals - prior['mean']))**2


def smoothness_prior(d_m, mean=0.0, standard_deviation=1.5, correlation_length=0.5):
    ''' Specify a Gaussian squared exponential smoothness prior with a correlation length.
    The input mean value is for the log10 of the concentration density dN/dlogdp (for
    discretization invariance), and it is converted to x internally. '''
    n_bins = d_m.shape[0]
    
    binwidth = np.log10(d_m[1]) - np.log10(d_m[0])
    
    prior = {}
    # Add the log10(binwidth) to convert from log10(dN/dlogdp) to log10(N)
    prior['mean'] = mean + np.log10(binwidth)
    
    # Correlation length == 1 corresponds to here to one order of magnitude
    # standard_deviation == standard deviation of the size distribution values
    
    a = standard_deviation**2
    distance_matrix = np.zeros((n_bins, n_bins))
    for i in range(n_bins):
        for j in range(n_bins):
            distance_matrix[i, j] = np.linalg.norm(
                np.log10(d_m[i]) - np.log10(d_m[j])
                )**2
    
    b = correlation_length / np.sqrt(2 * np.log(100))
    prior['covariance'] = a * np.exp(-0.5 * distance_matrix / b**2)
    
    # Add something small to the diagonal to make the matrix better invertible
    prior['covariance'] += 1e-12 * prior['covariance'][0, 0] * np.eye(n_bins)
    
    # Direct inverse
    # prior['inv_covariance'] = np.linalg.inv(prior['covariance'])
    # prior['L'] = np.linalg.cholesky(prior['inv_covariance'])
    
    # Inverse using Gohberg & Semencul formula for Toeplitz matrices (a bit better numerically)
    rhs = np.zeros(prior['covariance'].shape[0])
    rhs[0] = 1
    x = np.linalg.solve(prior['covariance'], rhs)
    B = toeplitz(x, np.zeros(x.shape[0]))
    C = toeplitz(np.concatenate(([0], np.flipud(x[1:]))), np.zeros(x.shape[0]))
    prior['inv_covariance'] = 1 / x[0] * (B @ B.T - C @ C.T)
    prior['L'] = np.linalg.cholesky(prior['inv_covariance']).T
    
    return prior


def Laplace_approximation(
        sizer : MobilityParticleSizeSpectrometer,
        prior,
        measurement,
        N_start=None
        ):
    ''' Compute the MAP estimate and Gaussian approximation to the posterior.
    This function assumes a positivity constraint in the form of a log10 transformation,
    and hence the returned values are given in log10-space.
    
    The initial guess N_start is an optional input.
    '''
    
    # Compare dimensions of measurement and prior covariances to decide if we should use the
    # Woodbury identity to compute the inverse in the formula for posterior covariance
    if prior['covariance'].shape[0] > measurement.noise_cov.shape[0]:
        use_inversion_lemma = True
    else:
        use_inversion_lemma = False
    
    if N_start is None:
        N_guess = np.ones(prior['inv_covariance'].shape[1]) * 0
    else:
        N_guess = N_start
    
    y_model = sizer.forward_model(N_guess)
    J = sizer.system_matrix * (10**N_guess * np.log(10))
    
    if use_inversion_lemma:
        JQ = J @ prior['covariance']
        S = JQ @ J.T
        S[np.diag_indices_from(S)] += measurement.noise_cov
        posterior_covariance = prior['covariance'] \
            - JQ.T @ np.linalg.inv(S) @ JQ
    else:
        posterior_covariance = np.linalg.inv(J.T @ (measurement.inv_noise_cov[:, None] * J)
                                             + prior['inv_covariance'])
    
    args = (sizer, measurement.inv_noise_L, prior, measurement.output)
    
    i = 0
    max_iter = 20
    min_step_reached = False
    enough_improvement = True
    required_improvement = 1e-6  # Minimum relative change in functional to keep iterating
    f_values = np.zeros(max_iter + 1)
    f_values[0] = -log_post(N_guess, sizer, measurement.inv_noise_L, prior, measurement.output)
    
    while (i < max_iter) and not min_step_reached and enough_improvement:
        gradient = (J.T * measurement.inv_noise_cov) @ (measurement.output - y_model) \
            - prior['inv_covariance'] @ (N_guess - prior['mean'])
        GN_dir = posterior_covariance @ gradient
        
        # Line search
        N_guess, f_values[i + 1], min_step_reached = linesearch(
            log_post, GN_dir, N_guess, f_values[i], *args
            )
        
        if (f_values[i] - f_values[i + 1]) / f_values[i] < required_improvement:
            enough_improvement = False
        
        y_model = sizer.forward_model(N_guess)
        J = sizer.system_matrix * (10**N_guess * np.log(10))
        
        if use_inversion_lemma:
            JQ = J @ prior['covariance']
            S = JQ @ J.T
            S[np.diag_indices_from(S)] += measurement.noise_cov
            posterior_covariance = prior['covariance'] \
                - JQ.T @ np.linalg.inv(S) @ JQ
        else:
            posterior_covariance = np.linalg.inv(J.T @ (measurement.inv_noise_cov[:, None] * J)
                                                 + prior['inv_covariance'])
        
        i += 1
    
    # Make sure that the posterior covariance is symmetric
    posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
        
    return N_guess, np.linalg.cholesky(posterior_covariance)


def Laplace_approximation_marginalize(sizer : MobilityParticleSizeSpectrometer,
                                      prior,
                                      measurement,
                                      marginalize_ion_mobility,
                                      marginalize_ion_ratio,
                                      marginalization_grid : Literal['standard', 'fine'] = 'standard',
                                      num_samples=5000,
                                      ):
    
    ''' Calculate the marginalized posterior of the PSD. Can marginalize over the ion mobilities
    and/or the ratio of positive to negative ions. Returns samples from the posterior mixture.
    
    Input:
        sizer : an initialized instance of the MobilityParticleSizeSpectrometer class
        prior : a dictionary with the prior specifications for the PSD
        measurement : a dictionary with data on the measurement
        marginalize_ion_mobility : True/False
        marginalize_ion_ratio : True/False
        marginalization_grid : 'standard' or 'fine', resolution of the marginalization grid
        num_samples : int
            Number of samples drawn from the posterior mixture.
    
    Output:
        posterior_mixture_samples : A vector consisting of draws from the marginalized posterior.
        ion_property_samples : array of floats,
            Values of the positive ion mobility, negative ion mobility, and ion ratio at each
            post_sample.
    '''
    
    # Set seed for reproducibility
    rng = np.random.default_rng()#seed=1)
    
    # Compute inversion and marginalize over ion mobility
    n_bins = sizer.d_m.shape[0]
    
    if marginalize_ion_mobility:
        if marginalization_grid == 'standard':
            step = 3.5e-6
        elif marginalization_grid == 'fine':
            step = 2.0e-6
        else:
            raise ValueError("marginalization_grid must be either 'standard' or 'fine'")
    else:
        # Set a large step, make_axis will reduce this to fit one grid point exactly in the middle
        # of the mobility trapezoid
        step = 1
    
    def make_axis(start, end, step):
        ''' Divide the range end - start into largest possible equal sized cells whose
        width <= step.
        Return the midpoints of those cells, and the final adjusted step size.
        '''
        span = end - start
        n = int(np.ceil(span / step))
        if n < 1:
            raise ValueError('Step is too large. ' +
                             f' Largest possible step is {np.floor(span / 2) : .3g}')
        step = span / n  # Adjust the stepsize to the nearest smaller one that fits
        
        return start + step / 2 + step * np.arange(n), step
    
    pos_ion_mobilities, step_pos = make_axis(1.05e-4, 1.70e-4, step)
    neg_ion_mobilities, step_neg = make_axis(1.10e-4, 2.10e-4, step)
    
    idxs = np.searchsorted(neg_ion_mobilities, pos_ion_mobilities, side='left')
    n_mobilities = np.sum(len(neg_ion_mobilities) - idxs)  # nbr of valid mobility pairs
    
    if marginalize_ion_ratio:
        if sizer.charging_model_name != 'LYF-interp-flux':
            raise ValueError("To marginalize ion ratio the charger has to be 'LYF-interp-flux'")
        n_ion_ratios = 10
        ion_ratio_std = 0.2 / 2
        
    else:
        if sizer.charging_model_name != 'LYF-interp':
            raise ValueError("Use charger 'LYF-interp' for marginalization of mobility")
        n_ion_ratios = 1
        ion_ratio = 1
    
    n_invert = n_mobilities * n_ion_ratios
    if n_invert == 1:
        print('Nothing to marginalize')
        return
    
    ion_properties = np.zeros((n_invert, 3))
    mixture_weights = np.zeros(n_invert)
    psd_posteriors = []
    
    # Starting guess for the Laplace approximation
    N_guess = np.ones(prior['inv_covariance'].shape[1]) * 0
    
    # Store the original charger properties so we can restore them
    # after they've been mutated by the marginalization loop below
    had_charger = getattr(sizer, 'charger_conditions_set', False)
    orig_charger_inputs = sizer.charger_inputs
    try:
        i = 0
        for p_idx, pos_ion_mobility in enumerate(pos_ion_mobilities):
            for n_idx, neg_ion_mobility in enumerate(neg_ion_mobilities):
                if pos_ion_mobility > neg_ion_mobility:
                    continue
                
                for _ in range(n_ion_ratios):
                    
                    if marginalize_ion_ratio:
                        ion_ratio = max(1e-6, rng.normal(loc=1.0, scale=ion_ratio_std))
                        sizer.set_charger_properties(pos_ion_mobility,
                                                     neg_ion_mobility,
                                                     ion_ratio
                                                     )
                    else:
                        sizer.set_charger_properties(pos_ion_mobility, neg_ion_mobility)
                    
                    # Calculate the Laplace approximation
                    MAP_estimate_log10, posterior_cov_L_log10 = Laplace_approximation(
                        sizer, prior, measurement, N_start=N_guess,
                        )
                    psd_posteriors.append(
                        PSDPosterior(sizer.d_m,
                                     post_mean_log10=MAP_estimate_log10,
                                     post_covL_log10=posterior_cov_L_log10,
                                     )
                        )
                    
                    # Calculate the (relative) weight of the current mixture component
                    mixture_weights[i] = calculate_area_of_cell(pos_ion_mobility,
                                                                neg_ion_mobility,
                                                                step_pos, step_neg)
                    
                    # Store ion properties
                    ion_properties[i] = pos_ion_mobility, neg_ion_mobility, ion_ratio
                    
                    # Use the current MAP estimate as a starting guess for the next one
                    # (it _probably_ is quite close to the truth)
                    N_guess = MAP_estimate_log10
                    
                    i += 1
    
    finally:
        # Restore the charger properties
        if had_charger and orig_charger_inputs is not None:
            sizer.set_charger_properties(*orig_charger_inputs)
        else:
            # No charger properties were set, restore to "not set" state
            sizer.charger_conditions_set = False
            sizer.system_matrix[:] = 0.0
    
    sum_w = np.sum(mixture_weights)
    if sum_w <= 0:
        raise RuntimeError('Mixture weights sum to zero. Check the grid construction.')
    mixture_weights /= sum_w
    
    # First calculate, proportional to mixture_probabilities, how many times
    # each mixture component should be sampled (counts)
    counts = mixture_weights * num_samples
    counts = np.floor(counts).astype(int)
    
    # The above rounds the number of counts down because they have to be integers.
    # Let's add in at random (but proportional to mixture_probabilities)
    # the missing numbers of counts using rng.choice()
    n_missing = num_samples - np.sum(counts)
    extra_components = rng.choice(len(mixture_weights),
                                  size=n_missing,
                                  p=mixture_weights
                                  )
    extra_counts = np.bincount(extra_components, minlength=len(mixture_weights))
    counts += extra_counts
    
    # Then sample each component in batches
    posterior_mixture_samples = np.zeros((num_samples, n_bins))
    ion_property_samples = np.zeros((num_samples, 3))
    start = 0
    for comp_idx, count in enumerate(counts):
        if count == 0:
            continue
        posterior_mixture_samples[start:start+count] = \
            psd_posteriors[comp_idx].get_samples(num=count)
        
        # Store the ion properties of each sample
        ion_property_samples[start:start+count] = ion_properties[comp_idx]
        
        start += count
    
    # Store as single precision samples to save space
    posterior_mixture_samples = posterior_mixture_samples.astype(np.float32, copy=False)
    ion_property_samples = ion_property_samples.astype(np.float32, copy=False)
    
    return posterior_mixture_samples, ion_property_samples


def calculate_area_of_cell(x, y, step_x, step_y):
    ''' Calculate the area of the midpoint cell inside the prior area, used in
    the marginalization of the mobilities.
    x = positive mobility
    y = negative mobility
    i.e., (x, y) is the cell midpoint
    step = cell width in that direction
    '''
    full_area = step_x * step_y
    
    # Bottom-right corner
    br_corner_x = x + step_x / 2
    br_corner_y = y - step_y / 2
    
    # First test if the lower right corner of the cell is to the right of the diagonal
    if br_corner_x > br_corner_y:
        # Calculate clipped area. We have 3 possible cases
        
        bl_corner_x = x - step_x / 2
        bl_corner_y = y - step_y / 2
        # bottom left corner past the diagonal
        if bl_corner_x > bl_corner_y:
            a = br_corner_x - bl_corner_x
            b = bl_corner_x - br_corner_y
            c = br_corner_x - bl_corner_y
            area_outside = a * b + 0.5 * a * c
            return full_area - area_outside
        
        tr_corner_x = x + step_x / 2
        tr_corner_y = y + step_y / 2
        # top left corner past the diagonal
        if tr_corner_x > tr_corner_y:
            a = tr_corner_y - br_corner_y
            b = br_corner_x - tr_corner_y
            c = tr_corner_y - br_corner_y
            area_outside = a * b + 0.5 * a * c
            return full_area - area_outside
        
        # only bottom right past the diagonal
        a = br_corner_x - br_corner_y
        area_outside = 0.5 * a ** 2
        return full_area - area_outside
    
    else:
        # whole cell inside
        return full_area


def linesearch(fn, direction, N_0, previous_best_f_value, *args):
    ''' Do simple linesearch.
    fn : function to be maximized (in our case the log posterior)
    '''
    
    min_stepl = 1e-3
    
    # Brute force, backtrack until the functional value increases, then choose previous step
    stepl = 1
    reduce = 0.7
    dN_old = stepl * direction
    post_old = -fn(N_0 + dN_old, *args)
    
    found_best_value = False
    while not found_best_value:
        stepl *= reduce
        dN_new = stepl * direction
        post_new = -fn(N_0 + dN_new, *args)
        
        if((post_new < previous_best_f_value and post_new > post_old)
            or stepl < (min_stepl * reduce)
            ):
            # Output the second to last iteration values (which was the best value)
            found_best_value = True
            post_new = post_old
            dN_new = dN_old
            stepl /= reduce  # undo the last reduction in step length
        
        else:
            # carry forward the previous iteration values
            dN_old = dN_new
            post_old = post_new
    
    # # For debug
    # print(f'Step length: {stepl:.2g}')
    
    if stepl < min_stepl:
        min_step_reached = True
    else:
        min_step_reached = False
    
    return N_0 + dN_new, post_new, min_step_reached


def rw_metropolis_preconditioned(
    log_post_fn,
    x_start,
    L_cov,                # Cholesky of the proposal covariance
    n_samples,
    burn_in,
    step_scale=1.0,       # Proposal scale factor
    adapt_scale=True,     # Robbins–Monro adaptation of step_scale
    target_acc=0.234,
    ):
    """
    Random-Walk Metropolis in log10-space with proposal covariance proportional to Laplace covariance.

    Returns:
        samples      : (n_samples, p) array of draws from the posterior in x-space
        acc_rate     : overall acceptance rate over the run (including burn-in)
        logp_trace   : (n_samples,) log posterior values at saved samples
        step_scale   : final step scale used
    """
    rng = np.random.default_rng()
    p = x_start.shape[0]
    L = np.array(L_cov, copy=True)

    # Storage
    samples = np.zeros((n_samples, p), dtype=float)
    logp_trace = np.zeros(n_samples, dtype=float)

    # Initialize
    x_curr = np.array(x_start, copy=True)
    logp_curr = float(log_post_fn(x_curr))
    accepted = 0

    if adapt_scale:
        log_step = np.log(step_scale)
    
    total_iters = burn_in + n_samples
    for it in range(total_iters):
        # Proposal
        z = rng.normal(size=p)
        x_prop = x_curr + np.exp(log_step) * (L @ z)
        logp_prop = float(log_post_fn(x_prop))
        
        log_alpha = logp_prop - logp_curr
        if np.log(rng.uniform()) < log_alpha:
            x_curr = x_prop
            logp_curr = logp_prop
            accepted += 1
        
        if adapt_scale:
            a_t = min(1.0, float(np.exp(log_alpha)))
            gamma = 1.0 / np.sqrt(it + 1.0)
            log_step += gamma * (a_t - target_acc)
        
        # Save after burn-in
        if it >= burn_in:
            idx = it - burn_in
            samples[idx] = x_curr
            logp_trace[idx] = logp_curr
    
    acc_rate = accepted / total_iters
    step_scale = np.exp(log_step)
    
    return samples, acc_rate, logp_trace, step_scale


def run_mcmc(
        sizer : MobilityParticleSizeSpectrometer,
        prior,
        measurement,
        num_samples=10000,
        ):
    ''' Sample the posterior with MCMC. Sampling is initialized from a Laplace approximation.
    This function assumes a positivity constraint in the form of a log10 transformation,
    and hence the returned values are given in log10-space.
    '''
    
    # Initialize
    MAP_estimate, posterior_cov_L = Laplace_approximation(sizer, prior, measurement)
    
    def log_post_fn(x):
        return log_post(x, sizer, measurement.inv_noise_L, prior, measurement.output)
    
    # Run random walk Metropolis
    samples_x, acc_rate, logp_vals, final_scale = rw_metropolis_preconditioned(
        log_post_fn,
        x_start=MAP_estimate,
        L_cov=posterior_cov_L,
        n_samples=num_samples,
        burn_in=1000,
        step_scale=1.0,
        adapt_scale=True,
        target_acc=0.234,
    )
    
    return np.power(10, samples_x)
