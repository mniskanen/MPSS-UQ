# -*- coding: utf-8 -*-

import numpy as np

from scipy.linalg import toeplitz
from typing import Literal

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
        marginalization_grid : Literal['standard', 'fine'] = 'standard',
        num_samples=None,
        sample_posterior=False,
        ):
    ''' Estimate the particle size distribution (PSD) from an MPSS measurement.
    
    With no marginalization, the posterior is approximated by the Laplace approximation
    (MAP + covariance). Optionally, one can set sample_posterior to True which carries out
    MCMC sampling and returns the samples.
    
    Marginalization is carried over over a range of ion mobilities and ratios using
    the LYF model. When doing marginalization, the posterior is approximated as a mixture of
    Gaussians (Laplace approximations) and samples from the mixture are returned.
    
    marginalization_grid refers to how many mobility pairs are calculated when marginalizing.
    marginalization_grid='fine' gives better looking Ntot histogram plots.
    The posterior estimates and credible intervals are basically the same as with
    marginalization_grid='standard', but marginalization_grid='standard' is faster.
    
    num_samples is the number of posterior samples returned when using sampling methods.
    
    Returns an InversionResult containing either the Laplace approximation or posterior samples.
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
    
    if marginalize_ion_mobility is True or marginalize_ion_ratio is True:
        if sample_posterior is True:
            raise ValueError('Cannot carry out MCMC with posterior marginalization. ' +
                             'Set sample_posterior to False.')
        
        if num_samples is None:
            num_samples = 5000
        
        posterior_samples, ion_property_samples = Laplace_approximation_marginalize(
            sizer, prior, measurement,
            marginalize_ion_mobility,
            marginalize_ion_ratio,
            marginalization_grid=marginalization_grid,
            num_samples=num_samples,
            )
        return InversionResult(sizer.d_m,
                               sl_measured,
                               post_samples=posterior_samples,
                               ion_property_samples=ion_property_samples,
                               )
    else:
        if sample_posterior:
            # MCMC
            if num_samples is None:
                num_samples = 100000
                
            posterior_samples = run_mcmc(sizer, prior, measurement, num_samples=num_samples)
            return InversionResult(sizer.d_m, sl_measured, post_samples=posterior_samples)
        
        else:
            # Laplace approximation
            MAP, post_cov_L = Laplace_approximation(sizer, prior, measurement)
            return InversionResult(sizer.d_m,
                                   sl_measured,
                                   post_mean_log10=MAP,
                                   post_covL_log10=post_cov_L,
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


def Laplace_approximation_marginalize(sizer : DifferentialMobilityParticleSizer,
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
        sizer : an initialized instance of the DifferentialMobilityParticleSizer class
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
        
        def make_axis(start, end, step):
            span = end - start
            n = int(span / step)  # base number of steps
            # Add one more point if leftover > half step
            if span - n * step > step / 2:
                n += 1
            return np.linspace(start, end, n + 1)
        
        pos_ion_mobilities = make_axis(1.05e-4, 1.70e-4, step)
        neg_ion_mobilities = make_axis(1.10e-4, 2.10e-4, step)
        
        idxs = np.searchsorted(neg_ion_mobilities, pos_ion_mobilities, side='left')
        n_mobilities = np.sum(len(neg_ion_mobilities) - idxs)  # nbr of valid mobility pairs
    
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
    
    ion_properties = np.zeros((n_invert, 3))
    inversion_results = []
    
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
                MAP_estimate_log10, posterior_cov_L_log10 = Laplace_approximation(
                    sizer, prior, measurement, N_start=N_guess,
                    )
                inversion_results.append(
                    InversionResult(sizer.d_m,
                                    post_mean_log10=MAP_estimate_log10,
                                    post_covL_log10=posterior_cov_L_log10,
                                    )
                    )
                
                # Store ion properties
                ion_properties[i] = pos_ion_mobility, neg_ion_mobility, ion_ratio
                
                # Use the current MAP estimate as a starting guess for the next one
                # (it _probably_ is quite close to the truth)
                N_guess = MAP_estimate_log10
                
                i += 1
    
    # The same probability for each mixture
    mixtures = np.arange(n_invert)
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
    posterior_mixture_samples = np.zeros((num_samples, n_bins))
    ion_property_samples = np.zeros((num_samples, 3))
    start = 0
    for comp_idx, count in enumerate(counts):
        if count == 0:
            continue
        posterior_mixture_samples[start:start+count] = \
            inversion_results[comp_idx].draw_posterior_samples(num=count)
        
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
        sizer : DifferentialMobilityParticleSizer,
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
