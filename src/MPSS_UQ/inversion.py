# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import toeplitz
from tqdm import tqdm
# from time import perf_counter

# Prevent the system from throttling down the CPU by giving any process that uses
# inversion methods a higher priority
import psutil, os
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)


def log_post(vals, DMPS, L_noise, prior, measurement):
    ''' Compute the logarithm of the (non-normalized) posterior. '''
    
    return -0.5 * np.linalg.norm(L_noise @ (measurement - DMPS.forward_model(vals)))**2 \
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
    This function assumes a positivity constraint in the form of a log10 transformation.
    
    N_start, an initial guess for the inversion of log10(N), is an optional input.
    
    '''
    
    if N_start is None:
        N_guess = np.ones(prior['inv_covariance'].shape[1]) * 0
    else:
        N_guess = N_start
    y_model = DMPS.forward_model(N_guess)
    J = DMPS.system_matrix @ np.diag(10**N_guess) * np.log(10)
    
    # posterior_covariance = np.linalg.inv(J.T @ inv_noise_cov @ J + prior['inv_covariance'])
    
    posterior_covariance = prior['covariance'] - prior['covariance'] @ J.T @ np.linalg.inv(
        measurement['noise_cov'] + J @ prior['covariance'] @ J.T
        ) @ J @ prior['covariance']
    
    args = (DMPS, measurement['noise_L'], prior, measurement['output'])
    
    i = 0
    max_iter = 20
    min_step_reached = False
    enough_improvement = True
    required_improvement = 1e-6  # Minimum relative change in functional to keep iterating
    f_values = np.zeros(max_iter + 1)
    f_values[0] = -log_post(N_guess, DMPS, measurement['noise_L'], prior, measurement['output'])
    
    while (i < max_iter) and not min_step_reached and enough_improvement:
        gradient = (J.T @ measurement['inv_noise_cov']) @ (measurement['output'] - y_model) \
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
        posterior_covariance = prior['covariance'] - prior['covariance'] @ J.T @ np.linalg.inv(
            measurement['noise_cov'] + J @ prior['covariance'] @ J.T
            ) @ J @ prior['covariance']
        
        i += 1
    
    # Make sure that the posterior covariance is symmetric
    if np.all(np.abs(posterior_covariance - posterior_covariance.T) >= 1e-15):
        posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
        
    return N_guess, posterior_covariance


def Laplace_approximation_marginalize(DMPS, prior, measurement,
                                      marginalize_ion_mobility,
                                      marginalize_ion_ratio):
    
    ''' Calculate the marginalized posterior of the PSD. Can marginalize over the ion mobilities
    and/or the ratio of positive to negative ions.
    
    Input:
        DMPS - an initialized instance of the DifferentialMobilityParticleSizer class
        prior - a dictionary with the prior specifications for the PSD
        measurement - a dictionary with data on the measurement
        ax - axes of where we want to plot the results
        marginalize_ion_mobility - True or False
        marginalize_ion_ratio - True or False
    
    Output:
        posterior_mixture_samples - a vector consisting of draws from the marginalized posterior.
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
        ion_ratio_std = 0.0
    
    
    n_invert = n_mobilities * n_ion_ratios
    if n_invert == 1:
        print('Nothing to marginalize')
        return
    
    # total_time_mobility = 0
    # total_time_laplace = 0
    
    MAP_estimates = np.zeros((n_invert, n_bins))
    posterior_covs = np.zeros((n_invert, n_bins, n_bins))
    posterior_cov_Ls = np.zeros((n_invert, n_bins, n_bins)) * np.nan
    log_posts = np.zeros((n_gridpoints_pos, n_gridpoints_neg)) * np.nan
    
    pbar = tqdm(position=0, desc='Marginalizing')
    pbar.reset(total = n_invert)
    i = 0
    for p_idx, pos_ion_mobility in enumerate(pos_ion_mobilities):
        for n_idx, neg_ion_mobility in enumerate(neg_ion_mobilities):
            if pos_ion_mobility > neg_ion_mobility:
                continue
            
            for _ in range(n_ion_ratios):
                
                ion_ratio = np.random.normal(loc=1.0, scale=ion_ratio_std)
                
                # Update the charging model and DMPS system matrix
                # t1 = perf_counter()
                DMPS.update_charger_ion_properties(pos_ion_mobility, neg_ion_mobility, ion_ratio)
                # t2 = perf_counter()
                # total_time_mobility += t2 - t1
                
                # Calculate the Laplace approximation
                MAP_estimates[i], posterior_covs[i] = Laplace_approximation(DMPS,
                                                                            prior,
                                                                            measurement
                                                                            )
                
                # Calculate the Cholesky factor
                posterior_cov_Ls[i] = np.linalg.cholesky(posterior_covs[i])
                # t3 = perf_counter()
                # total_time_laplace += t3 - t2
                
                i += 1
                pbar.update(1)
    
    # print(f'\nTotal time for mobility calculations: {total_time_mobility : .3f} sec')
    # print(f'Total time for Laplace calculations: {total_time_laplace : .3f} sec')
    
    # Possibly different probability for each mixture
    mixtures = np.arange(n_invert)
    n_posterior_mixture_samples = 100000
    log_posts_notnan = log_posts.flatten()
    log_posts_notnan = np.delete(log_posts_notnan, np.isnan(log_posts_notnan))
    
    mixture_probabilities = np.ones(mixtures.shape[0])
    mixture_probabilities /= np.sum(mixture_probabilities)
    
    posterior_mixture_samples = np.zeros((n_posterior_mixture_samples, n_bins))
    zeros = np.zeros(n_bins)
    ones = np.ones(n_bins)
    pbar = tqdm(position=0, desc='Sampling from the posterior mixture')
    pbar.reset(total = n_posterior_mixture_samples)
    for i in range(n_posterior_mixture_samples):
        # First choose the component
        component = rng.choice(mixtures, p=mixture_probabilities)
        
        # Then sample from that Gaussian
        posterior_mixture_samples[i] = 10**(MAP_estimates[component] +
                    posterior_cov_Ls[component] @ rng.normal(loc=zeros, scale=ones))
        
        pbar.update(1)
    
    return posterior_mixture_samples


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
