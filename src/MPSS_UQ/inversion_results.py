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

    Supports a Gaussian posterior approximation (MAP + covariance) in the linear and
    log10 versions, and posterior samples, but only one of them at a time.
    Computes the credible intervals on demand.
    """
    
    def __init__(self,
                 d_m,
                 post_mean=None,
                 post_cov=None,
                 post_mean_log10=None,
                 post_cov_log10=None,
                 post_samples=None,
                 ):
        
        # Particle size vector
        self.d_m = d_m
        
        # Helper for plotting the results
        self.binwidth = np.log10(self.d_m[1]) - np.log10(self.d_m[0])
        
        # Check the inputs
        input1 = post_mean is not None and post_cov is not None
        input2 = post_mean_log10 is not None and post_cov_log10 is not None
        input3 = post_samples is not None
        
        given_inputs = np.sum([input1, input2, input3])
        
        if given_inputs == 0:
            raise ValueError('No valid input provided. Specify one of the three input types.')
        elif given_inputs > 1:
            raise ValueError('Multiple inputs provided. Specify only one input.')
        
        if input1:
            self.post_mean = post_mean
            self.post_cov = post_cov
            self.input_mode = 'gaussian-linear'
        
        elif input2:
            self.post_mean_log10 = post_mean_log10
            self.post_cov_log10 = post_cov_log10
            self.input_mode = 'gaussian-log10'
        
        elif input3:
            self.post_samples = post_samples
            self.input_mode = 'samples'
    
    
    def _postprocess_resuts_from_covariance_linear(self, CI):
        
        # Calculate the requested credible interval estimates
        sigma = np.sqrt(np.diag(self.post_cov))
        k = norm.ppf(0.5 + CI / 100 / 2)
        
        CI_lower = self.post_mean - k * sigma
        CI_upper = self.post_mean + k * sigma
        
        return self.post_mean, CI_lower, CI_upper
    
    
    def _postprocess_resuts_from_covariance_log10(self, CI):
        
        # Calculate the requested credible interval estimates
        sigma = np.sqrt(np.diag(self.post_cov_log10))
        k = norm.ppf(0.5 + CI / 100 / 2)
        
        CI_lower = 10**(self.post_mean_log10 - k * sigma)
        CI_upper = 10**(self.post_mean_log10 + k * sigma)
        
        return 10**self.post_mean_log10, CI_lower, CI_upper
    
    
    def _postprocess_resuts_from_samples(self, CI):
        
        post_mean = np.mean(self.post_samples, axis=0)
        
        # Highest density intervals
        CI_lower = np.zeros(self.d_m.shape[0])
        CI_upper = np.zeros_like(CI_lower)
        for i in range(self.d_m.shape[0]):
            CI_lower[i], CI_upper[i] = highest_density_interval(self.post_samples[:, i], CI / 100)
            
        return post_mean, CI_lower, CI_upper
    
    
    def posterior_summary(self, coverage=95):
        """
        Returns the posterior mean and credible intervals.
        Input:
            coverage - the percentage of the posterior mass the credible interval should cover.
        """
        if coverage <= 0 or coverage >= 100:
            raise ValueError('Invalid value for posterior coverage. It should be in (0, 100).')
        
        if self.input_mode == 'gaussian-linear':
            return self._postprocess_resuts_from_covariance_linear(coverage)
        elif self.input_mode == 'gaussian-log10':
            return self._postprocess_resuts_from_covariance_log10(coverage)
        elif self.input_mode == 'samples':
            return self._postprocess_resuts_from_samples(coverage)
    
    
    def compute_total_particle_count(self, method="montecarlo", n_samples=10000):
        
        # TODO: jatka tästä, lasketaan kokonaishiukkasmäärät ja niille epävarmuudet
        """
        Computes total particle count and uncertainty.

        Parameters:
            method: "montecarlo" or "linearized"
            n_samples: number of samples for Monte Carlo

        Returns:
            dict with total MAP, mean, and CI
        """
        if self.samples_log10 is not None:
            samples_linear = 10 ** self.samples_log10
            total_samples = np.sum(samples_linear, axis=1)
        elif self.map_log10 is not None and self.cov_log10 is not None:
            if method == "montecarlo":
                samples = np.random.multivariate_normal(self.map_log10, self.cov_log10, size=n_samples)
                samples_linear = 10 ** samples
                total_samples = np.sum(samples_linear, axis=1)
            else:
                raise NotImplementedError("Only Monte Carlo method is implemented for Gaussian posterior.")
        else:
            raise ValueError("No posterior information available.")

        return {
            "Total MAP": np.sum(10**self.map_log10),
            "Total Mean": np.mean(total_samples),
            "Total CI": tuple(np.percentile(total_samples, [2.5, 97.5]))
        }


class InversionDataset:
    """
    Container for inversion results across multiple measurements.
    """

    def __init__(self, datetimes):
        self.datetimes = datetimes
        self.results = [None] * len(datetimes)

    def assign_result(self, idx, result: InversionResult):
        self.results[idx] = result

    def posterior_summary(self, *args, **kwargs):
        """
        Returns arrays of posterior mean, lower CI, upper CI for all results.
        """
        # return [r.posterior_summary(*args, **kwargs) for r in self.results if r is not None]
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
    
    
    def get_total_counts(self):
        """
        Returns list of total particle count statistics for all results.
        """
        return [r.compute_total_particle_count() for r in self.results if r is not None]
