# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def plot_psd(ax, d_m, *args, n=None, N=None, **kwargs):
    ''' A function to plot a particle size distribution on a supplied axis.
    If the input given is "n", we don't need to normalize by bin width, and
    if the input given is "N", we do. '''
    
    if n is not None:
        assert N is None, 'Cannot input both n and N.'
        ax.loglog(d_m * 1e9, n, *args, **kwargs)
        
    elif N is not None:
        binwidth = np.diff(np.log10(d_m))
        binwidth = np.concatenate((binwidth, binwidth[-1, np.newaxis]))
        ax.loglog(d_m * 1e9, N / binwidth, *args, **kwargs)
    
    ax.set_xlabel('Particle mobility diameter (nm)')
    ax.set_ylabel(r'dN / d$\log$d$_m$')


def highest_density_interval(samples, percentage) :
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


def plot_system_matrix(DMPS, num=None, title=None):
    ''' Plot the system matrix of the DMPS.
    Input:
        DMPS - The DMPS object
        num - figure number
        title - additional title for the figure
    '''
    
    if num is None:
        num = 10
    
    if title is None:
        title = ''
    else:
        title = f', {title}'  # add a comma
    
    plt.figure(num=num), plt.clf()
    X, Y = np.meshgrid(DMPS.d_m * 1e9, DMPS.d_m_data * 1e9)
    
    plt.pcolormesh(X, Y, DMPS.system_matrix)
    
    # Z = np.clip(dma.system_matrix, 1e-4, np.inf)
    # norm = colors.LogNorm(vmin=np.nanmin(Z), vmax=np.nanmax(Z), clip=False)
    # plt.pcolormesh(X, Y, Z, norm=norm)
    
    # plt.pcolormesh(X, Y, np.log10(np.clip(dma.system_matrix, 1e-6, np.inf)))
    
    plt.gca().invert_yaxis()
    plt.title(f'DMPS system matrix{title}')
    plt.xlabel('Modelled particle diameters (nm)')
    plt.ylabel('Mmeasured particle diameters (nm)')
    plt.xscale('log')
    plt.yscale('log')
    plt.axis('equal')
    # plt.colorbar()


def plot_MAP(DMPS, log10_N, log10_postcov, ax):
    
    N_inverted = 10**log10_N
    binwidth = np.log10(DMPS.d_m[1]) - np.log10(DMPS.d_m[0])
    
    ax.fill_between(DMPS.d_m * 1e9,
                        10**(log10_N + 1.96 * np.sqrt(np.diag(log10_postcov))) / binwidth,
                        10**(log10_N - 1.96 * np.sqrt(np.diag(log10_postcov))) / binwidth,
                        alpha=0.25, facecolor='C0', label='95 % credible interval')
    
    plot_psd(ax, DMPS.d_m, N=N_inverted, label='MAP estimate', color='k', linestyle='--')
    
    ax.set_yscale('linear')
    ax.legend()


def plot_datafit(DMPS, output_measured, log10_N, ax):
    
    # Data prediction
    output_predicted = DMPS.forward_model(log10_N)
    ax.semilogx(DMPS.d_m_data * 1e9, output_measured, label='Observed output')
    ax.semilogx(DMPS.d_m_data * 1e9, output_predicted, label='Predicted output (from inversion)')
    ax.legend()
    ax.grid('on')
    ax.set_title('output')
    ax.set_xlabel('Selected DMA output diameter (nm)')
    ax.set_ylabel('Counts (#)')


def plot_marginalized_psd(DMPS, posterior_samples, ax, CI=95):
    ''' A basic plot of the marginalized posterior of the PSD.
    
    Input:
        CI - percentage of the credible interval (0-100)
    '''
    
    if CI < 0 or CI > 100:
        raise ValueError('Invalid credible interval percentage.')
    
    # Divide by bin width (dN / dlogdp)
    binwidth = np.diff(np.log10(DMPS.d_m))
    binwidth = np.concatenate((binwidth, binwidth[-1, np.newaxis]))
    samples_dNdlogdp = posterior_samples / binwidth
    
    # Highest density intervals
    pr_plot = np.zeros((2, DMPS.d_m.shape[0]))
    for i in range(DMPS.d_m.shape[0]):
        # breakpoint()
        # pr_plot[:, i] = highest_density_interval(samples_dNdlogdp[:, i], CI / 100)
        pr_plot[:, i] = highest_density_interval(np.log10(posterior_samples[:, i]), CI / 100)
    
    pr_plot = 10**pr_plot / binwidth
    
    ax.fill_between(DMPS.d_m * 1e9, pr_plot[1], pr_plot[0], alpha=0.25,
                          facecolor='C0', label=f'{CI} % credible interval')
    
    # Plot samples
    n_jump = int(np.ceil(posterior_samples.shape[0] / 50))
    ax.plot(DMPS.d_m * 1e9, samples_dNdlogdp[::n_jump, :].T,
                         color='C2',
                         linewidth=1,
                         alpha=0.2
                         )
    ax.plot(DMPS.d_m * 1e9, samples_dNdlogdp[-1, :], color='C2', linewidth=1, alpha=0.2,
                         label='Posterior samples'
                         )
    
    # TODO Note for the paper: All estimates have to be calculated in the log-space and only then
    # transformed to the linear space. At least is looks wrong otherwise...
    mean_estimate = 10**np.mean(np.log10(posterior_samples), axis=0)
    # mean_estimate = np.mean(posterior_samples, axis=0)
    ax.plot(DMPS.d_m * 1e9, mean_estimate / binwidth, 'k--', label='Mean of posterior samples')
    
    ax.set_xscale('log')
    
    ax.legend()
    ax.set_title('Particle size distribution')
    ax.set_xlabel('Particle mobility diameter (nm)')
    ax.set_ylabel(r'dN / d$\log$d$_m$')


def plot_Ntot_histogram(Ntots, ax):
    ''' Plot a histogram and some credible intervals of the sampled Ntot. '''
    
    [Ntot_low95, Ntot_high95] = highest_density_interval(Ntots, 0.95)
    [Ntot_low50, Ntot_high50] = highest_density_interval(Ntots, 0.50)
    [plt_lo, plt_hi] = highest_density_interval(Ntots, 0.9999)
    counts, bins = np.histogram(Ntots, bins=50, range=(plt_lo, plt_hi))
    width = bins[1] - bins[0]
    ax.bar(bins[1:] - width, counts, width=width, edgecolor="white", color='C0', alpha=0.8)
    ymin, ymax = ax.get_ylim()
    
    ax.vlines(x=[Ntot_low95, Ntot_high95],
               ymin=ymin, ymax=ymax * 1.15, colors='k', linestyle='--')
    ax.vlines(x=[Ntot_low50, Ntot_high50],
               ymin=ymin, ymax=ymax * 1.00, colors='k', linestyle='--')
    
    ax.set_ylim((ymin, 1.3 * ymax))
    
    # Plot the CI ranges
    ax.plot(np.array([1.02 * Ntot_low95, 0.99 * Ntot_high95]), 1.2 * np.array([ymax, ymax]),
            'k--', linewidth=1.5)
    ax.plot(np.array([1.02 * Ntot_low50, 0.99 * Ntot_high50]), 1.05 * np.array([ymax, ymax]),
            'k--', linewidth=1.5)
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 10
    }
    ax.annotate('95 % CI',
                 xy=(0.5 * (Ntot_low95 + Ntot_high95), ymax * 1.23), **anno_args)
    ax.annotate('50 % CI',
                 xy=(0.5 * (Ntot_low50 + Ntot_high50), ymax * 1.08), **anno_args)
    anno_args['size'] = 16
    ax.annotate('[', xy=(Ntot_low95, 1.2 * ymax), **anno_args)
    ax.annotate(']', xy=(Ntot_high95, 1.2 * ymax), **anno_args)
    ax.annotate('[', xy=(Ntot_low50, 1.05 * ymax), **anno_args)
    ax.annotate(']', xy=(Ntot_high50, 1.05 * ymax), **anno_args)
    
    ax.set_title(r'Sampled $N_{tot}$ values and credible intervals')# + '\n' +
              # f'50 % CI: [{Ntot_low50 : .3g}, {Ntot_high50 : .3g}]' + '\n' +
              # f'95 % CI: [{Ntot_low95 : .3g}, {Ntot_high95 : .3g}]')
    ax.set_xlabel(r'$N_{tot}$')
    ax.set_ylabel('Number of samples')