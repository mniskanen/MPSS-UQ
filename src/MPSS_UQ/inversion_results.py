# -*- coding: utf-8 -*-

"""
Classes for storing and processing inversion results from MPSS data.
"""

import numpy as np
from scipy.stats import norm
# import scipy.stats


def highest_density_interval(samples, percentage):
    """ Calculate the "percentage" highest probability density (HPD) region
    from a set of samples. """
    
    # A corner case
    if np.isclose(percentage, 1.0):
        return np.array([np.min(samples), np.max(samples)])
    
    samples_sorted = np.sort(np.copy(samples))
    n_tot = samples.shape[0]
    
    # Number of samples needed for the required percentage
    n_samples = int(np.floor(percentage * n_tot))
    
    # Width of all intervals with the right number of samples
    widths = samples_sorted[n_samples:] - samples_sorted[:-n_samples]
    min_width_idx = np.argmin(widths)
    
    hdi_start = samples_sorted[min_width_idx]
    hdi_end = samples_sorted[min_width_idx + n_samples]
    
    return np.array([hdi_start, hdi_end])


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
    d_m_full : 
    sl_measured : slice
        A slice indicating the start and stop indices of d_m_full that corresponds to the
        measured size range.
    post_mean_log10 : None,
    post_cov_log10 : None,
    post_samples : None,
    ion_property_samples : array of floats or None, optional
        Values of the positive and negative ion mobility and ion ratio at each post_sample.
    reporting_range : 'measured' or 'full'
        The size range considered for posterior summaries, can be changed later.
        'measured' takes the shortest size interval in terms of inversion bins d_m_full such that
        the measured size range is covered. 'full' uses the whole inverted size range.
    """
    
    
    def __init__(self,
                 d_m_full,
                 sl_measured,
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
            self.post_samples = post_samples
            self.input_mode = 'samples'
            if ion_property_samples is not None:
                self.ion_property_samples = ion_property_samples
        
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
        
        # Reset the Cholesky decompositions of the post covariance
        if self.input_mode == 'gaussian-log10':
            self.cov_log10_cholesky = None
    
    
    def _postprocess_results_from_covariance_log10(self, CI):
        
        # Calculate the requested credible interval estimates
        # Marginal stds: sigma_i = sqrt(sum_j C[i,j]^2)
        sigma = np.sqrt(np.sum(self.post_covL_log10[self.sl, :]**2, axis=1))
        k = norm.ppf(0.5 + CI / 100 / 2)
        
        CI_lower = 10**(self.post_mean_log10[self.sl] - k * sigma)
        CI_upper = 10**(self.post_mean_log10[self.sl] + k * sigma)
        
        return 10**self.post_mean_log10[self.sl], CI_lower, CI_upper
    
    
    def _postprocess_results_from_samples(self, CI):
        
        post_mean = np.mean(self.post_samples[:, self.sl], axis=0)
        
        # Highest density intervals
        CI_lower = np.zeros(self.d_m.shape[0])
        CI_upper = np.zeros_like(CI_lower)
        for j, i in enumerate(range(self.sl.start, self.sl.stop)):
            CI_lower[j], CI_upper[j] = highest_density_interval(self.post_samples[:, i], CI / 100)
            
        return post_mean, CI_lower, CI_upper
    
    
    def posterior_variance(self):
        """ Calculate (if needed) and return the posterior variance.
        """
        if self.input_mode == 'gaussian-log10':
            return np.diag(self.post_covL_log10[self.sl, self.sl])**2
        elif self.input_mode == 'samples':
            return np.var(np.log10(self.post_samples[:, self.sl]), axis=0)
    
    
    def posterior_summary(self, coverage=95):
        """
        Returns the posterior mean and credible intervals.
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
    
    
    def get_posterior_sample(self):
        ''' Return one sample from the posterior.
        Can only be used for the input mode 'gaussian-log10'.
        '''
        
        if self.input_mode == 'gaussian-log10':
            return 10**(self.post_mean_log10[self.sl] 
                        + self.post_covL_log10[self.sl, self.sl]
                        @ self.rng.normal(loc=0.0, scale=1.0, size=len(self.d_m))
                        )
        
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
    
    
    def posterior_summary(self, *args, **kwargs):
        """
        Returns arrays of posterior mean, lower CI, upper CI for all results.
        """
        num_results = len(self.results)
        num_d_m = self.results[0].d_m.shape[0]
        posterior_means = np.zeros((num_results, num_d_m))
        CI_lower = np.zeros_like(posterior_means)
        CI_upper = np.zeros_like(posterior_means)
        
        for i in range(num_results):
            posterior_means[i], CI_lower[i], CI_upper[i] = self.results[i].posterior_summary(
                *args, **kwargs
                )
        
        return posterior_means, CI_lower, CI_upper
    
    
    def set_reporting_range(self, reporting_range : str):
        for i in range(len(self.results)):
            self.results[i].set_reporting_range(reporting_range)
