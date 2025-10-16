# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import toeplitz
from scipy.stats import norm
from tqdm import tqdm

from MPSS_UQ.inversion_results import InversionResult

# Prevent the system from throttling down the CPU by giving any process that uses
# inversion methods a higher priority
import psutil, os
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)


def invert_psd(
        DMPS,         # DMPS or SMPS
        measurement,  # Measurement
        prior=None,   # If None, builds a default smoothness prior from DMPS.d_m TODO
        marginalize_ion_mobility=False,
        marginalize_ion_ratio=False,
        method='sampling',
        num_samples=5000,
        ):
    ''' Estimate the particle size distribution (PSD) from an MPSS measurement and
    return an InversionResult containing posterior summaries or samples.
    Marginalization can be carried over over a range of ion mobilities and ratios using
    the LYF model. Method ('sampling' or 'gaussian-approximation') refers to the way
    the marginalized posterior is represented. Sampling is more accurate (given enough
    samples) but can be a bit slower and takes much more memory than Gaussian
    approximation.
    '''
    # Calculate the measured reporting_range:
    eps = 1e-16
    lo = DMPS.d_m_data.min() - eps
    hi = DMPS.d_m_data.max() + eps
    
    i_start = np.searchsorted(DMPS.d_m, lo, side='left')
    i_stop  = np.searchsorted(DMPS.d_m, hi, side='right')
    
    # Take the bins just outside the range to make sure to show the whole measured range
    i_start = max(0, i_start - 1)
    i_stop = min(len(DMPS.d_m), i_stop + 1)
        
    sl_measured = slice(i_start, i_stop)
    
    if marginalize_ion_mobility is False and marginalize_ion_ratio is False:
        MAP, post_cov = Laplace_approximation(DMPS, prior, measurement)
        return InversionResult(DMPS.d_m,
                               post_mean_log10=MAP,
                               post_cov_log10=post_cov,
                               sl_measured=sl_measured,
                               )
    
    else:
        
        if method == 'sampling':
            posterior_samples = Laplace_approximation_marginalize(DMPS,
                                                                  prior,
                                                                  measurement,
                                                                  marginalize_ion_mobility,
                                                                  marginalize_ion_ratio,
                                                                  method,
                                                                  num_samples=num_samples,
                                                                  )
            
            return InversionResult(DMPS.d_m,
                                   post_samples=posterior_samples,
                                   sl_measured=sl_measured,
                                   )
            
        elif method == 'gaussian-approximation':
            posterior_mean, posterior_covariance = Laplace_approximation_marginalize(
                DMPS,
                prior,
                measurement,
                marginalize_ion_mobility,
                marginalize_ion_ratio,
                method,
                )
            
            return InversionResult(DMPS.d_m,
                                   post_mean=posterior_mean,
                                   post_cov=posterior_covariance,
                                   sl_measured=sl_measured,
                                   )
        
        else:
            raise ValueError('Unknkown marginalization method')


def log_post(vals, DMPS, L_noise, prior, y_meas):
    ''' Compute the logarithm of the (non-normalized) posterior. '''
    
    return -0.5 * np.linalg.norm(L_noise @ (y_meas - DMPS.forward_model(vals)))**2 \
        - 0.5 * np.linalg.norm(prior['L'] @ (vals - prior['mean']))**2


def smoothness_prior(d_m, mean, correlation_length, standard_deviation):
    ''' Specify a Gaussian smoothness prior with a correlation length. '''
    
    n_bins = d_m.shape[0]
    
    prior = {}
    prior['mean'] = mean
    
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
    prior['covariance'] += 1e-6 * prior['covariance'][0, 0] * np.eye(n_bins)
    
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


def Laplace_approximation(DMPS, prior, measurement, N_start=None):
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
    
    y_model = DMPS.forward_model(N_guess)
    J = DMPS.system_matrix @ np.diag(10**N_guess) * np.log(10)
    
    if use_inversion_lemma:
        posterior_covariance = prior['covariance'] - prior['covariance'] @ J.T @ np.linalg.inv(
            measurement.noise_cov + J @ prior['covariance'] @ J.T
            ) @ J @ prior['covariance']
    else:
        posterior_covariance = np.linalg.inv(J.T @ measurement.inv_noise_cov @ J
                                             + prior['inv_covariance'])
    
    args = (DMPS, measurement.noise_L, prior, measurement.output)
    
    i = 0
    max_iter = 20
    min_step_reached = False
    enough_improvement = True
    required_improvement = 1e-6  # Minimum relative change in functional to keep iterating
    f_values = np.zeros(max_iter + 1)
    f_values[0] = -log_post(N_guess, DMPS, measurement.noise_L, prior, measurement.output)
    
    while (i < max_iter) and not min_step_reached and enough_improvement:
        gradient = (J.T @ measurement.inv_noise_cov) @ (measurement.output - y_model) \
            - prior['inv_covariance'] @ (N_guess - prior['mean'])
        GN_dir = posterior_covariance @ gradient
        
        # Line search
        N_guess, f_values[i + 1], min_step_reached = linesearch(
            log_post, GN_dir, N_guess, f_values[i], *args
            )
        
        if (f_values[i] - f_values[i + 1]) / f_values[i] < required_improvement:
            enough_improvement = False
        
        y_model = DMPS.forward_model(N_guess)
        J = DMPS.system_matrix @ np.diag(10**N_guess) * np.log(10)
        
        if use_inversion_lemma:
            posterior_covariance = prior['covariance'] - prior['covariance'] @ J.T @ np.linalg.inv(
                measurement.noise_cov + J @ prior['covariance'] @ J.T
                ) @ J @ prior['covariance']
        else:
            posterior_covariance = np.linalg.inv(J.T @ measurement.inv_noise_cov @ J
                                                 + prior['inv_covariance'])
        
        i += 1
    
    # Make sure that the posterior covariance is symmetric
    if np.all(np.abs(posterior_covariance - posterior_covariance.T) >= 1e-15):
        posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
        
    return N_guess, posterior_covariance


def Laplace_approximation_marginalize(DMPS,
                                      prior,
                                      measurement,
                                      marginalize_ion_mobility,
                                      marginalize_ion_ratio,
                                      method='sampling',
                                      num_samples=5000,
                                      ):
    
    ''' Calculate the marginalized posterior of the PSD. Can marginalize over the ion mobilities
    and/or the ratio of positive to negative ions.
    
    Input:
        DMPS - an initialized instance of the DifferentialMobilityParticleSizer class
        prior - a dictionary with the prior specifications for the PSD
        measurement - a dictionary with data on the measurement
        marginalize_ion_mobility - True/False
        marginalize_ion_ratio - True/False
        method - 'sampling' or 'gaussian-approximation'
                 'sampling' returns samples from the posterior mixture.
                 'gaussian-approximation' returns the mean and covariance of the
                 posterior calculated with an analytical formula from the means 
                 and covariances of the individual mixtures.
        num_samples - (if using sampling) number of samples drawn from the posterior mixture.
    
    Output:
        posterior_mixture_samples - a vector consisting of draws from the marginalized posterior
        OR
        posterior_mean, posterior_covariance - of the mixture density.
    '''
    
    # Set seed for reproducibility
    rng = np.random.default_rng(seed=1)
    
    # Compute inversion and marginalize over ion mobility
    n_bins = DMPS.d_m.shape[0]
    
    if marginalize_ion_mobility:
        n_gridpoints_pos = 25
        n_gridpoints_neg = int(n_gridpoints_pos * 1.05 / 0.65)
        n_invert = n_gridpoints_pos * n_gridpoints_neg
        pos_ion_mobilities = np.linspace(1.05e-4, 1.70e-4, n_gridpoints_pos + 1)
        neg_ion_mobilities = np.linspace(1.05e-4, 2.10e-4, n_gridpoints_neg + 1)
    
        # Midpoints
        pos_ion_mobilities = pos_ion_mobilities[0:-1] \
            + 0.5 * (pos_ion_mobilities[1] - pos_ion_mobilities[0])
        neg_ion_mobilities = neg_ion_mobilities[0:-1] \
            + 0.5 * (neg_ion_mobilities[1] - neg_ion_mobilities[0])
        
        PP, NN = np.meshgrid(pos_ion_mobilities, neg_ion_mobilities, indexing='ij')
        n_mobilities = np.sum(NN >= PP)
    
    else:
        pos_ion_mobilities = np.array([1.35e-4])
        neg_ion_mobilities = np.array([1.60e-4])
        n_mobilities = 1
    
    if marginalize_ion_ratio:
        n_ion_ratios = 10
        ion_ratio_std = 0.2 / 2
        
    else:
        n_ion_ratios = 1
    
    n_invert = n_mobilities * n_ion_ratios
    if n_invert == 1:
        print('Nothing to marginalize')
        return
    
    MAP_estimates_log10 = np.zeros((n_invert, n_bins))
    posterior_covs_log10 = np.zeros((n_invert, n_bins, n_bins))
    log_posts = np.zeros((n_gridpoints_pos, n_gridpoints_neg)) * np.nan
    if method == 'sampling':
        posterior_cov_Ls_log10 = np.zeros((n_invert, n_bins, n_bins)) * np.nan
    
    # Starting guess for the Laplace approximation
    N_guess = np.ones(prior['inv_covariance'].shape[1]) * 0
    
    i = 0
    for p_idx, pos_ion_mobility in enumerate(pos_ion_mobilities):
        for n_idx, neg_ion_mobility in enumerate(neg_ion_mobilities):
            if pos_ion_mobility > neg_ion_mobility:
                continue
            
            for _ in range(n_ion_ratios):
                
                if marginalize_ion_ratio:
                    ion_ratio = np.random.normal(loc=1.0, scale=ion_ratio_std)
                    DMPS.set_charger_properties(pos_ion_mobility,
                                                neg_ion_mobility,
                                                ion_ratio
                                                )
                else:
                    DMPS.set_charger_properties(pos_ion_mobility, neg_ion_mobility)
                
                # Calculate the Laplace approximation
                MAP_estimates_log10[i], posterior_covs_log10[i] = Laplace_approximation(
                    DMPS,
                    prior,
                    measurement,
                    N_start=N_guess,
                    )
                
                # Calculate the Cholesky factor
                if method == 'sampling':
                    posterior_cov_Ls_log10[i] = np.linalg.cholesky(posterior_covs_log10[i])
                
                # Use the current MAP estimate as a starting guess for the next one
                # (it _probably_ is quite close to the truth)
                N_guess = MAP_estimates_log10[i]
                
                i += 1
    
    # Possibly different probability for each mixture
    mixtures = np.arange(n_invert)
    log_posts_notnan = log_posts.flatten()
    log_posts_notnan = np.delete(log_posts_notnan, np.isnan(log_posts_notnan))
    
    mixture_probabilities = np.ones(mixtures.shape[0])
    mixture_probabilities /= np.sum(mixture_probabilities)
    
    if method == 'sampling':
        # First calculate, proportional to mixture_probabilities, how many times
        # each mixture component should be sampled (counts)
        counts = mixture_probabilities * num_samples
        counts = np.floor(counts).astype(int)
        
        # The above rounds the number of counts down because they have to be integers.
        # Let's add in at random (but proportional to mixture_probabilities)
        # the missing numbers of counts using rng.choice()
        n_missing = num_samples - np.sum(counts)
        extra_components = rng.choice(len(mixture_probabilities),
                                      size=n_missing,
                                      p=mixture_probabilities
                                      )
        extra_counts = np.bincount(extra_components, minlength=len(mixture_probabilities))
        counts += extra_counts
        
        # Then sample each component in batches
        posterior_mixture_samples = np.zeros((num_samples, n_bins),
                                             dtype=np.float32
                                             )
        start = 0
        for comp_idx, count in enumerate(counts):
            if count == 0:
                continue
            posterior_mixture_samples[start:start+count] = np.power(10,
                MAP_estimates_log10[comp_idx][:, None]
                + posterior_cov_Ls_log10[comp_idx] @ rng.normal(loc=0.0,
                                                                scale=1.0,
                                                                size=(n_bins, count)
                                                                )
                ).T
            start += count
        
        return posterior_mixture_samples
    
    elif method == 'gaussian-approximation':
        
        # Calculate the mean and covariance of the mixture analytically in the linear space.
        # We have to first transform each mixture component from log10 to linear space, and only
        # then can we compute the mean and covariance.
        posterior_means = np.zeros_like(MAP_estimates_log10)
        for i in range(n_invert):
            posterior_means[i] = 10**MAP_estimates_log10[i] * np.exp(
                1 / 2 * np.diag(posterior_covs_log10[i]) * np.log(10)**2
                )
        posterior_mean = mixture_probabilities @ posterior_means
        posterior_covariances = np.zeros_like(posterior_covs_log10)
        I = np.eye(len(posterior_mean))
        # Only consider the diagonal of the covariance here
        for i in range(n_invert):
            posterior_covariances[i] = np.outer(posterior_means[i], posterior_means[i]) * (
                np.diag(np.exp(np.diag(posterior_covs_log10[i]) * np.log(10)**2)) - I
                )
        posterior_covariance = np.zeros_like(posterior_covs_log10[0])
        for i in range(n_invert):
            mean_diff = posterior_means[i] - posterior_mean
            posterior_covariance += mixture_probabilities[i] * (
                posterior_covariances[i] + np.outer(mean_diff, mean_diff)
                )
        
        return posterior_mean, posterior_covariance
    
    else:
        raise ValueError('Uknown mixing method')


def linesearch(fn, direction, N_0, previous_best_f_value, *args):
    ''' Do simple linesearch.
    fn : function to be maximized (in our case the log posterior)
    '''
    
    min_stepl = 1e-3
    
    # Brute force, backtrack until the functional value increases, then choose previous step
    stepl = 1
    reduce = 0.7
    dN_old = stepl * direction
    while any((N_0 + dN_old) > 10):
        stepl *= reduce
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
