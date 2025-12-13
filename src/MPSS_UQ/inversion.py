# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import toeplitz
from scipy.stats import norm
from tqdm import tqdm

from MPSS_UQ.inversion_results import InversionResult
from MPSS_UQ.particlesizers import DifferentialMobilityParticleSizer

# Prevent the system from throttling down the CPU by giving any process that uses
# inversion methods a higher priority
import psutil, os
p = psutil.Process(os.getpid())
try:
    p.nice(psutil.HIGH_PRIORITY_CLASS)  # Windows constant
except AttributeError:
    # psutil.HIGH_PRIORITY_CLASS doesn't exist on non-Windows
    try:
        p.nice(-10)  # Linux/macOS: lower niceness = higher priority
    except Exception:
        pass  # silently ignore if not permitted



def invert_psd(
        sizer : DifferentialMobilityParticleSizer,  # Only DMPS implemented for now
        measurement,  # Measurement
        prior=None,   # If None, builds a default smoothness prior
        marginalize_ion_mobility=False,
        marginalize_ion_ratio=False,
        num_samples=5000,
        ):
    ''' Estimate the particle size distribution (PSD) from an MPSS measurement and
    return an InversionResult containing posterior summaries or samples.
    Marginalization can be carried over over a range of ion mobilities and ratios using
    the LYF model.
    num_samples is the number of posterior samples returned when doing marginalization.
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
        prior = smoothness_prior(sizer.d_m, 0, 0.5, 1.5)
    
    if marginalize_ion_mobility is False and marginalize_ion_ratio is False:
        MAP, post_cov_L = Laplace_approximation(sizer, prior, measurement)
        return InversionResult(sizer.d_m,
                               sl_measured,
                               post_mean_log10=MAP,
                               post_covL_log10=post_cov_L,
                               )
    
    else:
        posterior_samples, ion_property_samples = Laplace_approximation_marginalize(
            sizer, prior, measurement,
            marginalize_ion_mobility,
            marginalize_ion_ratio,
            num_samples=num_samples,
            )
        return InversionResult(sizer.d_m,
                               sl_measured,
                               post_samples=posterior_samples,
                               ion_property_samples=ion_property_samples,
                               )


def log_post(vals, sizer, L_noise, prior, y_meas):
    ''' Compute the logarithm of the (non-normalized) posterior. '''
    
    return -0.5 * np.linalg.norm(L_noise * (y_meas - sizer.forward_model(vals)))**2 \
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


def Laplace_approximation(
        sizer : DifferentialMobilityParticleSizer,
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
        S[np.diag_indices_from((S))] += measurement.noise_cov
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
            S[np.diag_indices_from((S))] += measurement.noise_cov
            posterior_covariance = prior['covariance'] \
                - JQ.T @ np.linalg.inv(S) @ JQ
        else:
            posterior_covariance = np.linalg.inv(J.T @ (measurement.inv_noise_cov[:, None] * J)
                                                 + prior['inv_covariance'])
        
        i += 1
    
    # Make sure that the posterior covariance is symmetric
    posterior_covariance = (posterior_covariance + posterior_covariance.T) / 2
        
    return N_guess, np.linalg.cholesky(posterior_covariance)


def Laplace_approximation_marginalize(sizer : DifferentialMobilityParticleSizer,
                                      prior,
                                      measurement,
                                      marginalize_ion_mobility,
                                      marginalize_ion_ratio,
                                      num_samples=5000,
                                      ):
    
    ''' Calculate the marginalized posterior of the PSD. Can marginalize over the ion mobilities
    and/or the ratio of positive to negative ions. Returns samples from the posterior mixture.
    
    Input:
        sizer : an initialized instance of the DifferentialMobilityParticleSizer class
        prior : a dictionary with the prior specifications for the PSD
        measurement : a dictionary with data on the measurement
        marginalize_ion_mobility : True/False
        marginalize_ion_ratio : True/False
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
        n_gridpoints_pos = 25
        n_gridpoints_neg = int(n_gridpoints_pos * 1.05 / 0.65)
        n_invert = n_gridpoints_pos * n_gridpoints_neg
        pos_ion_mobilities = np.linspace(1.05e-4, 1.70e-4, n_gridpoints_pos + 1)
        neg_ion_mobilities = np.linspace(1.10e-4, 2.10e-4, n_gridpoints_neg + 1)
    
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
        ion_ratio = 1
    
    n_invert = n_mobilities * n_ion_ratios
    if n_invert == 1:
        print('Nothing to marginalize')
        return
    
    MAP_estimates_log10 = np.zeros((n_invert, n_bins))
    posterior_cov_Ls_log10 = np.zeros((n_invert, n_bins, n_bins))
    log_posts = np.zeros((n_gridpoints_pos, n_gridpoints_neg))
    ion_properties = np.zeros((n_invert, 3))
    
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
                    sizer.set_charger_properties(pos_ion_mobility,
                                                neg_ion_mobility,
                                                ion_ratio
                                                )
                else:
                    sizer.set_charger_properties(pos_ion_mobility, neg_ion_mobility)
                
                # Calculate the Laplace approximation
                MAP_estimates_log10[i], posterior_cov_Ls_log10[i] = Laplace_approximation(
                    sizer,
                    prior,
                    measurement,
                    N_start=N_guess,
                    )
                
                # Store ion properties
                ion_properties[i] = pos_ion_mobility, neg_ion_mobility, ion_ratio
                
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
    ion_property_samples = np.zeros((num_samples, 3))
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
        
        # Store the ion properties of each sample
        ion_property_samples[start:start+count] = ion_properties[comp_idx]
        
        start += count
    
    return posterior_mixture_samples, ion_property_samples


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
