# -*- coding: utf-8 -*-

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tck

from MPSS_UQ.inversion_results import highest_density_interval, InversionResult


def plot_psd(ax, d_m, N, *args, **kwargs):
    ''' A helper function to plot a particle size distribution, as dN/dlogdm,
    on a supplied axis ax. The input should be "N", the concentration of particles in each bin. '''
    
    binwidth = np.diff(np.log10(d_m))
    binwidth = np.concatenate((binwidth, binwidth[-1, np.newaxis]))
    
    ax.loglog(d_m * 1e9, N / binwidth, *args, **kwargs)
    
    ax.set_xlabel('Particle mobility diameter (nm)')
    ax.set_ylabel(r'dN / d$\log$d$_m$')
    ax.legend()


def plot_posterior_summary(ax, result, CI_coverage=95):
    ''' Plot a simple posterior summary (mean value and credible intervals).
    Input:
        result - instance of an InversionResult
        CI_coverage - percentage of posterior the credible intervals should cover
    '''
    
    N_median, CI_lower, CI_upper = result.posterior_summary(coverage=CI_coverage)
    
    ax.fill_between(result.d_m * 1e9,
                    CI_upper / result.binwidth,
                    CI_lower / result.binwidth,
                    alpha=0.25,
                    facecolor='C0',
                    label=f'{CI_coverage} % credible interval'
                    )
    
    label = 'Posterior median'
    
    plot_psd(ax, result.d_m, N=N_median, linestyle='-', color='C0', label=label)
    
    ax.legend()


def plot_timeseries(ax, datetimes, d_m, Z,
                    cmap='viridis',
                    log_color_scale=True,
                    vmin=None, vmax=None,  # manual limits
                    cbar_label=None,
                    ):
    '''
    Plot Z(time, size) as pcolormesh with gap masking and optional manual color limits.
    
    Parameters
    ----------
    vmin, vmax : float or None
        Manual color limits. If None, computed from data.
    log_color_scale : bool
        If True (and norm is None), use LogNorm; else linear Normalize.
    '''
    
    # Compute time diffs in minutes
    dt = np.diff(datetimes)
    
    # Estimate typical measurement length
    typical_dt = np.median(dt)
    gap_thresh = 1.5 * typical_dt
    
    # Alternatively give it direcetly in minutes
    # gap_thresh = np.timedelta64(9, 'm')
    
    # Give the last datapoint a width so it is correctly visualized
    date_edges = np.empty(len(datetimes) + 1, dtype=datetimes.dtype)
    date_edges[:-1] = datetimes
    date_edges[-1] = datetimes[-1] + typical_dt
    
    binwidth = np.log10(d_m[1]) - np.log10(d_m[0])
    d_m_edges = np.empty(len(d_m) + 1, dtype=d_m.dtype)
    d_m_edges[:-1] = 10**(np.log10(d_m) - 0.5 * binwidth)
    d_m_edges[-1] = 10**(np.log10(d_m[-1]) + 0.5 * binwidth)
    
    # To nm
    d_m_edges *= 1e9
    
    # Identify the intervals (between t[i] and t[i+1]) that are too wide
    gap_intervals = dt > gap_thresh
    
    # # Create masked array and mask the column whose cell spans the gap interval
    gap_cols = np.zeros(Z.shape[1], dtype=bool)
    gap_cols[:-1] = gap_intervals  # column i corresponds to [t[i], t[i+1])
    Z_masked = ma.array(Z, copy=True)
    Z_masked[:, gap_cols] = ma.masked
    
    # Determine color scaling
    data_for_limits = Z_masked.compressed() if ma.isMaskedArray(Z_masked) else Z_masked.ravel()
    if vmin is None:
        vmin = np.quantile(data_for_limits, 0.001)
    if vmax is None:
        vmax = np.max(data_for_limits)
    
    if log_color_scale:
        _norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        _norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    im = ax.pcolormesh(date_edges, d_m_edges, Z_masked, cmap=cmap, norm=_norm, shading='auto')
    cbar = ax.figure.colorbar(im, ax=ax, label=cbar_label)
    ax._colorbar = cbar  # attach the colorbar for easy reference later
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
    ax.set_yticks([d_m[0] * 1e9, 20, 50, 100, 250, 500, d_m[-1] * 1e9])
    ax.set_ylabel('Particle diameter (nm)')
    ax.set_xlabel('Time')
    
    return im


def plot_system_matrix(ax, DMPS, title=None):
    ''' Plot the system matrix of the DMPS.
    Input:
        ax - figure axis
        DMPS - The DMPS object
        title - additional title for the figure
    '''
    if title is None:
        title = ''
    else:
        title = f', {title}'  # add a comma
    
    dmx = DMPS.d_m * 1e9          # x centers
    dmy = DMPS.d_m_data * 1e9
    
    # Build log-spaced edges from centers
    def edges_from_centers_log(c):
        c = np.asarray(c)
        # interior edges as geometric mean of neighbors
        inner = np.sqrt(c[1:] * c[:-1])
        # extrapolate endpoints (geometric ratio)
        r0 = c[1] / c[0]
        r1 = c[-1] / c[-2]
        e0 = c[0] / np.sqrt(r0)
        e1 = c[-1] * np.sqrt(r1)
        return np.concatenate(([e0], inner, [e1]))
    
    xe = edges_from_centers_log(dmx)
    ye = edges_from_centers_log(dmy)
    Xedges, Yedges = np.meshgrid(xe, ye)
    
    im = ax.pcolormesh(Xedges, Yedges, DMPS.system_matrix)
    
    ax.invert_yaxis()
    ax.set_title(f'DMPS system matrix{title}')
    ax.set_xlabel('Modelled diameter (nm)')
    ax.set_ylabel('Nominal measured diameter (nm)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal', adjustable='box')
    # cbar = ax.figure.colorbar(im, ax=ax, label='Matrix values')


def plot_datafit(ax, DMPS, output_measured, result : InversionResult, CI_coverage=95):
    # Data prediction
    n_samples = 5000
    output_predicted_samples = np.zeros((n_samples, len(output_measured)))
    
    rng = np.random.default_rng()
    
    # The forward model can only be run for the full length PSD
    reporting_range = result.reporting_range
    if reporting_range == 'measured':
        result.set_reporting_range('full')
    
    if result.input_mode == 'samples':
        # Limit the max number of samples to the max number available
        n_samples = min(len(result.post_samples), 5000)
        
        def _update_ion_props(ion_properties):
            if DMPS.charging_model_name == 'LYF-interp':
                DMPS.set_charger_properties(ion_properties[0], ion_properties[1])
            elif DMPS.charging_model_name == 'LYF-interp-flux':
                DMPS.set_charger_properties(ion_properties[0], ion_properties[1],
                                            ion_properties[2])
        
        sample_idxs = rng.choice(len(result.post_samples), size=n_samples, replace=False)
        # Evaluate the posterior samples model output in order so that we minimize the number
        # of times the charger properties need to be reset
        sample_idxs.sort()
        if hasattr(result, 'ion_property_samples'):
            ion_properties = result.ion_property_samples[sample_idxs[0]]
            _update_ion_props(ion_properties)
        
        for i, sample_idx in enumerate(sample_idxs):
            # Check if ion properties changed, do we need to update DMPS
            if hasattr(result, 'ion_property_samples'):
                if np.any(ion_properties != result.ion_property_samples[sample_idx]):
                    ion_properties = result.ion_property_samples[sample_idx]
                    _update_ion_props(ion_properties)
            
            output_predicted_samples[i] = DMPS.forward_model(
                np.log10(result.post_samples[sample_idx])
                )
    
    elif result.input_mode == 'gaussian-log10':
        for i in range(n_samples):
            output_predicted_samples[i] = DMPS.forward_model(
                np.log10(result.draw_posterior_samples(num=1).squeeze())
                )
    else:
        raise ValueError('Unknown result input mode.')
    
    # Put the reporting range back to what it was when calling this function
    if reporting_range == 'measured':
        result.set_reporting_range('measured')
    
    output_predicted_median_noiseless = np.median(output_predicted_samples, axis=0)
    
    # Add counting noise
    output_predicted_samples_noisy = rng.poisson(lam=output_predicted_samples).astype(np.float64)
    # Add additive noise (electronic/environmental)
    noise_std = 2 + 0.01 * output_predicted_samples
    output_predicted_samples_noisy += noise_std * rng.normal(loc=0.0, scale=1.0,
                                                             size=output_predicted_samples.shape)
    
    # Highest density intervals
    CI_lower = np.zeros(len(output_measured))
    CI_upper = np.zeros_like(CI_lower)
    for i in range(len(output_measured)):
        CI_lower[i], CI_upper[i] = highest_density_interval(output_predicted_samples_noisy[:, i],
                                                            CI_coverage / 100
                                                            )
    
    ax.semilogx(DMPS.d_m_data * 1e9, output_measured, 'kx', label='Observed counts')
    ax.fill_between(DMPS.d_m_data * 1e9,
                    CI_upper,
                    CI_lower,
                    alpha=0.25,
                    facecolor='C1',
                    label=f'{CI_coverage} % posterior predictive interval'
                    )
    ax.semilogx(DMPS.d_m_data * 1e9, output_predicted_median_noiseless, 'C1-',
                label='Median predicted signal (noiseless)')
    ax.legend()
    ax.grid('on')
    ax.set_title('Data fit')
    ax.set_xlabel('DMA output nominal diameter (nm)')
    ax.set_ylabel('Counts (#)')


def plot_Ntot_histogram(ax, Ntots, Ntot_true=None, xlimits=None):
    ''' Plot a histogram and some credible intervals of the sampled Ntot. '''
    
    [Ntot_low95, Ntot_high95] = highest_density_interval(Ntots, 0.95)
    # [Ntot_low50, Ntot_high50] = highest_density_interval(Ntots, 0.50)
    
    if xlimits is None:
        [plt_lo, plt_hi] = highest_density_interval(Ntots, 0.997)
    else:
        plt_lo, plt_hi = xlimits
    
    counts, bins = np.histogram(Ntots, bins=100, range=(plt_lo, plt_hi), density=True)
    width = bins[1] - bins[0]
    ax.bar(bins[1:] - width, counts, width=width, edgecolor="white", color='C0', alpha=0.8)
    ymin, ymax = ax.get_ylim()
    
    ax.vlines(x=[Ntot_low95, Ntot_high95],
               ymin=ymin, ymax=ymax * 1.15, colors='k', linestyle='--')
    # ax.vlines(x=[Ntot_low50, Ntot_high50],
    #            ymin=ymin, ymax=ymax * 1.00, colors='k', linestyle='--')
    
    ax.set_ylim((ymin, 1.3 * ymax))
    
    # Plot the CI ranges
    # N50_diff = Ntot_high50 - Ntot_low50
    N95_diff = Ntot_high95 - Ntot_low95
    ax.plot(
        np.array([Ntot_low95 + 0.1 * N95_diff, Ntot_high95 - 0.1 * N95_diff]),
        1.2 * np.array([ymax, ymax]),
        'k--',
        linewidth=1.5
        )
    # ax.plot(
    #     np.array([Ntot_low50 + 0.1 * N50_diff, Ntot_high50 - 0.1 * N50_diff]),
    #     1.05 * np.array([ymax, ymax]),
    #     'k--',
    #     linewidth=1.5
    #     )
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 10
    }
    ax.annotate('95 % CI',
                 xy=(0.5 * (Ntot_low95 + Ntot_high95), ymax * 1.23), **anno_args)
    # ax.annotate('50 % CI',
    #              xy=(0.5 * (Ntot_low50 + Ntot_high50), ymax * 1.08), **anno_args)
    anno_args['size'] = 16
    ax.annotate('[', xy=(Ntot_low95, 1.2 * ymax), **anno_args)
    ax.annotate(']', xy=(Ntot_high95, 1.2 * ymax), **anno_args)
    # ax.annotate('[', xy=(Ntot_low50, 1.05 * ymax), **anno_args)
    # ax.annotate(']', xy=(Ntot_high50, 1.05 * ymax), **anno_args)
    
    ax.set_title(r'Sampled $N_{tot}$ values and credible intervals')# + '\n' +
              # f'50 % CI: [{Ntot_low50 : .3g}, {Ntot_high50 : .3g}]' + '\n' +
              # f'95 % CI: [{Ntot_low95 : .3g}, {Ntot_high95 : .3g}]')
    ax.set_xlabel(r'$N_{tot}$')
    ax.set_ylabel(r'$N_{tot}$ density')
    
    if Ntot_true is not None:
        low_y, high_y = ax.get_ylim()
        ax.plot([Ntot_true, Ntot_true], [0, 0.2 * high_y], 'k-', linewidth=5)
        ax.annotate(r'True $N_\mathrm{tot}$',
                     xy=(Ntot_true, 0.22 * high_y), ha='center', size=10)