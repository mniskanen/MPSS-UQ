# -*- coding: utf-8 -*-

import math
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.ticker as tck

from MPSS_UQ.analysis import highest_density_interval, summarize_samples
from MPSS_UQ.inversion import PSDPosterior
from MPSS_UQ.particlesizers import MobilityParticleSizeSpectrometer


def plot_psd(ax, d_m, N, *args, **kwargs):
    ''' A helper function to plot a particle size distribution, as dN/dlogdm,
    on a supplied axis ax. The input should be "N", the concentration of particles in each bin. '''
    
    binwidth = np.diff(np.log10(d_m))
    binwidth = np.concatenate((binwidth, binwidth[-1, np.newaxis]))
    
    ax.loglog(d_m * 1e9, N / binwidth, *args, **kwargs)
    
    ax.set_xlabel('Particle mobility diameter (nm)')
    ax.set_ylabel(r'dN / d$\log$d$_m$')
    ax.legend()


def plot_posterior_summary(ax, psd_posterior : PSDPosterior, CI_coverage=95, color='C0'):
    ''' Plot a simple posterior summary (mean value and credible intervals).
    Input:
        psd_posterior - instance of PSDPosterior
        CI_coverage - percentage of posterior the credible intervals should cover
    '''
    
    N_median, CI_lower, CI_upper = psd_posterior.summary(coverage=CI_coverage)
    
    ax.fill_between(psd_posterior.d_m * 1e9,
                    CI_upper / psd_posterior.binwidth,
                    CI_lower / psd_posterior.binwidth,
                    alpha=0.25,
                    facecolor=color,
                    label=f'{CI_coverage} % credible interval'
                    )
    
    label = 'Posterior median'
    
    plot_psd(ax, psd_posterior.d_m, N=N_median, linestyle='-', color=color, label=label)
    
    ax.legend()


def _detect_gaps(datetimes, gap_factor=1.5):
    """
    Detect measurement gaps in a datetime array.

    Parameters
    ----------
    datetimes : array-like of datetime64 (or similar)
        Sorted measurement timestamps.
    gap_factor : float
        A gap is declared when the interval between consecutive
        timestamps exceeds gap_factor * median(dt).

    Returns
    -------
    typical_dt : timedelta
        Median time step.
    gap_mask : ndarray of bool, shape (len(datetimes) - 1,)
        True where the interval [t_i, t_{i+1}) is a gap.
    """
    dt = np.diff(datetimes)
    typical_dt = np.median(dt)
    gap_thresh = gap_factor * typical_dt
    gap_mask = dt > gap_thresh
    return typical_dt, gap_mask


def _contiguous_segments(datetimes, gap_factor=1.5):
    """
    Return a list of slices for contiguous (non-gap) segments.

    Each slice ``s`` satisfies: within ``datetimes[s]`` there are
    no gaps larger than ``gap_factor * median(dt)``.

    Parameters
    ----------
    datetimes : array-like of datetime64
    gap_factor : float

    Returns
    -------
    segments : list of slice
    typical_dt : timedelta
    """
    typical_dt, gap_mask = _detect_gaps(datetimes, gap_factor)
    gap_indices = np.where(gap_mask)[0]  # indices into diff array

    segments = []
    start = 0
    for gi in gap_indices:
        # segment runs from start up to and including index gi
        segments.append(slice(start, gi + 1))
        start = gi + 1
    # final segment
    segments.append(slice(start, len(datetimes)))

    return segments, typical_dt


def plot_timeseries_1d(ax,
                       datetimes,
                       samples,
                       coverage=95,
                       color='C0',
                       title=None,
                       log_yscale=False,
                       ymin=None,
                       ymax=None,
                       gap_factor=1.5,
                       ):
    '''
    Plot a 1D time series (median + credible interval band) with automatic gap masking.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    datetimes : array-like of datetime64, shape (n_meas,)
        Measurement timestamps.
    samples : array-like, shape (n_samples, n_meas)
        Posterior samples at each measurement time.
    coverage : float in (0, 100), optional
        Credible interval width in percent. Default is 95.
    color : str, optional
        Color for the line and band.
    title : str or None, optional
    log_yscale : bool, optional
    ymin, ymax : float or None, optional
        Manual y-axis limits. Default to data range.
    gap_factor : float, optional
        Gap threshold as a multiple of the median time step.
    '''
    median, CI_low, CI_high = summarize_samples(samples, coverage=coverage)
    
    segments, typical_dt = _contiguous_segments(datetimes, gap_factor)
    
    label_ci = 'Credible interval' if not coverage else f'{coverage} % credible interval'
    label_med = 'Median estimate'
    
    for seg in segments:
        t = datetimes[seg]
        ax.fill_between(t,
                        CI_low[seg],
                        CI_high[seg],
                        alpha=0.5,
                        facecolor=color,
                        label=label_ci,
                        )
        ax.plot(t, median[seg], color=color, linewidth=0.5, label=label_med)

        # Only label the first segment to avoid duplicate legend entries
        label_ci = None
        label_med = None
    
    ax.set_xlim(datetimes[0], datetimes[-1] + typical_dt)
    ymax = np.quantile(CI_high, 1) if ymax is None else ymax
    ymin = np.quantile(CI_low, 0) if ymin is None else ymin
    if log_yscale:
        ax.set_yscale('log')
    ax.set_ylim([ymin, ymax])
    ax.grid('on')
    ax.legend()
    ax.set_xlabel('Time')
    ax.set_title(title, loc='left')


def plot_timeseries_2d(ax, datetimes, d_m, Z,
                    cmap='viridis',
                    log_color_scale=True,
                    vmin_q=None, vmax_q=None,
                    vmin=None, vmax=None,
                    discrete_bounds=None,
                    cbar_extend="neither",
                    show_cbar=True,
                    cbar_label=None,
                    cbar_as_perc=False,
                    gap_factor=1.5,
                    alpha=None,  # optional per-cell alpha, shape (n_bins, n_meas)
                    ):
    '''
    Plot a 2D time-size field as a pcolormesh with automatic gap masking.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    datetimes : array-like of datetime64, shape (n_meas,)
        Measurement timestamps (columns of Z).
    d_m : array-like of float, shape (n_bins,)
        Bin-centre diameters in metres.
    Z : array-like, shape (n_bins, n_meas)
        Values to plot (e.g. size distribution).
    cmap : str, optional
    log_color_scale : bool, optional
        Use LogNorm if True, linear Normalize otherwise.
    vmin, vmax : float or None
        Manual color limits. Mutually exclusive with vmin_q / vmax_q.
    vmin_q, vmax_q : float or None
        Quantile-based color limits in [0, 1]. Defaults: 0.001, 0.999.
    show_cbar : bool, optional
    cbar_label : str or None, optional
    cbar_as_perc : bool, optional
        Format colorbar tick labels as percentages.
    gap_factor : float, optional
        Gap threshold as a multiple of the median time step.
    alpha : array-like or None, optional
        Per-cell transparency, shape (n_bins, n_meas). Values in [0, 1].
        When provided, the mesh is rendered with per-cell RGBA facecolors
        and gap columns are forced to alpha=0 (fully transparent).

    Returns
    -------
    im : matplotlib.collections.QuadMesh
    '''
    typical_dt, gap_mask = _detect_gaps(datetimes, gap_factor)

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
    
    # # Create masked array and mask the column whose cell spans the gap interval
    gap_cols = np.zeros(Z.shape[1], dtype=bool)
    gap_cols[:-1] = gap_mask  # column i corresponds to [t[i], t[i+1])
    # Mask gap columns
    gap_cols = np.zeros(Z.shape[1], dtype=bool)
    gap_cols[:-1] = gap_mask
    Z_masked = ma.array(Z, copy=True)
    Z_masked[:, gap_cols] = ma.masked
    
    if vmin is not None and vmin_q is not None:
        raise ValueError('Give only vmin or vmin_q.')
    if vmax is not None and vmax_q is not None:
        raise ValueError('Give only vmax or vmax_q.')
    
    # Determine color scaling
    data_for_limits = Z_masked.compressed() if ma.isMaskedArray(Z_masked) else Z_masked.ravel()
    if vmin is None:
        if vmin_q is None:
            vmin_q = 0.001
        vmin = np.nanquantile(data_for_limits, vmin_q)
    if vmax is None:
        if vmax_q is None:
            vmax_q = 0.999
        vmax = np.nanquantile(data_for_limits, vmax_q)
    
    if discrete_bounds is not None:
        # Discrete / binned colormap
        discrete_bounds = np.asarray(discrete_bounds)
        n_bins_color = len(discrete_bounds) - 1
    
        # Get the base colormap object
        if isinstance(cmap, str):
            cmap_base = mpl.colormaps[cmap]
        else:
            cmap_base = cmap
    
        # Resample to exactly the number of discrete bins
        cmap = cmap_base.resampled(n_bins_color)
    
        _norm = colors.BoundaryNorm(discrete_bounds, ncolors=n_bins_color)
    
        # Override vmin/vmax so downstream code stays consistent
        vmin = discrete_bounds[0]
        vmax = discrete_bounds[-1]
    
    elif log_color_scale:
        _norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    else:
        _norm = colors.Normalize(vmin=vmin, vmax=vmax)
    
    im = ax.pcolormesh(date_edges, d_m_edges, Z_masked, cmap=cmap, norm=_norm, shading='auto')
    
    # Per-cell alpha
    if alpha is not None:
        alpha = np.asarray(alpha, dtype=float)
        cmap_obj = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
        rgba = cmap_obj(_norm(Z))           # shape (n_bins, n_meas, 4)
        rgba[..., 3] = alpha                # apply per-cell alpha

        # Force gap columns to fully transparent
        rgba[:, gap_cols, 3] = 0.0

        # Also force any originally masked cells to transparent
        if ma.isMaskedArray(Z_masked):
            rgba[Z_masked.mask, 3] = 0.0

        im.set_array(None)
        im.set_facecolors(rgba.reshape(-1, 4))
    
    if show_cbar:
        cbar = ax.figure.colorbar(im,
                                  ax=ax,
                                  label=cbar_label,
                                  pad=0.02,
                                  extend=cbar_extend,
                                  )
        ax._my_colorbar = cbar # attach the colorbar for easy reference later
    
        if discrete_bounds is not None:
            # Place ticks at the bin edges
            cbar.set_ticks(discrete_bounds)
            if cbar_as_perc:
                cbar.set_ticklabels([f"{v*100:.0f} %" for v in discrete_bounds])
        else:
            if log_color_scale:
                _my_format_colorbar(cbar, vmin, vmax, log_scale=log_color_scale, sig_digits=0)
            else:
                _my_format_colorbar(cbar, vmin, vmax, log_scale=log_color_scale, sig_digits=1)
            if cbar_as_perc:
                cbar.ax.yaxis.set_major_formatter(
                    tck.FuncFormatter(lambda x, pos: f"{x*100:.0f} %")
                )
    
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(tck.FormatStrFormatter('%.0f'))
    ax.set_yticks([d_m[0] * 1e9, 20, 50, 100, 250, 500, d_m[-1] * 1e9])
    ax.set_ylabel('Particle diameter (nm)')
    ax.set_xlabel('Time')
    
    return im


def _my_format_colorbar(cbar, vmin, vmax, min_dist_frac=0.06, log_scale=True, sig_digits=0):
    if log_scale:
        range_span = np.log10(vmax) - np.log10(vmin)
        def _dist(a, b):
            return abs(np.log10(a) - np.log10(b))
    else:
        range_span = vmax - vmin
        def _dist(a, b):
            return abs(a - b)

    min_dist = min_dist_frac * range_span

    # Get default ticks and keep only those within range
    ticks = np.array(cbar.get_ticks())
    ticks = ticks[(ticks >= vmin) & (ticks <= vmax)]

    # Remove ticks that are too close to vmin or vmax
    keep = [t for t in ticks
            if _dist(t, vmin) > min_dist and _dist(t, vmax) > min_dist]

    ticks = np.unique(np.concatenate(([vmin], keep, [vmax])))
    labels = [_fmt_sig(t, sig_digits) for t in ticks]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)


def _fmt_sig(x, sig_digits=0):
    # Format to number of significant digits
    if x < 1:
        if x == 0:
            return "0"
        p = -int(math.floor(math.log10(abs(x)))) + sig_digits # decimals needed
        return f"{x:.{p}f}"
    else:
        return f"{x:.{sig_digits}f}"


def plot_system_matrix(ax, MPSS : MobilityParticleSizeSpectrometer, title=None):
    ''' Plot the system matrix of the MPSS.
    Input:
        ax - figure axis
        MPSS - The MobilityParticleSizeSpectrometer instance
        title - additional title for the figure
    '''
    if title is None:
        title = ''
    else:
        title = f', {title}'  # add a comma
    
    dmx = MPSS.d_m * 1e9          # x centers
    dmy = MPSS.d_m_data * 1e9
    
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
    
    im = ax.pcolormesh(Xedges, Yedges, MPSS.system_matrix)
    
    ax.invert_yaxis()
    ax.set_title(f'MPSS system matrix{title}')
    ax.set_xlabel('Modelled diameter (nm)')
    ax.set_ylabel('Nominal measured diameter (nm)')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_aspect('equal', adjustable='box')
    cbar = ax.figure.colorbar(im, ax=ax, label='Matrix values')


def plot_datafit(ax,
                 MPSS : MobilityParticleSizeSpectrometer,
                 output_measured,  # counts
                 psd_posterior : PSDPosterior,
                 CI_coverage=95,
                 ):
    # Data prediction
    
    n_samples = 5000
    rng = np.random.default_rng()
    
    # The forward model can only be run for the full length PSD
    reporting_range = psd_posterior.reporting_range
    if reporting_range == 'measured':
        psd_posterior.set_reporting_range('full')
    
    if psd_posterior.input_mode == 'samples':
        # Limit the max number of samples to the max number available
        n_samples = min(len(psd_posterior.post_samples), 5000)
        output_predicted_samples = np.zeros((n_samples, len(output_measured)))
        
        def _update_ion_props(ion_properties):
            if MPSS.charging_model_name == 'LYF-interp':
                MPSS.set_charger_properties(ion_properties[0], ion_properties[1])
            elif MPSS.charging_model_name == 'LYF-interp-flux':
                MPSS.set_charger_properties(ion_properties[0], ion_properties[1],
                                            ion_properties[2])
        
        sample_idxs = rng.choice(len(psd_posterior.post_samples), size=n_samples, replace=False)
        # Evaluate the posterior samples model output in order so that we minimize the number
        # of times the charger properties need to be reset
        sample_idxs.sort()
        if hasattr(psd_posterior, 'ion_property_samples'):
            ion_properties = psd_posterior.ion_property_samples[sample_idxs[0]]
            _update_ion_props(ion_properties)
        
        for i, sample_idx in enumerate(sample_idxs):
            # Check if ion properties changed, do we need to update MPSS
            if hasattr(psd_posterior, 'ion_property_samples'):
                if np.any(ion_properties != psd_posterior.ion_property_samples[sample_idx]):
                    ion_properties = psd_posterior.ion_property_samples[sample_idx]
                    _update_ion_props(ion_properties)
            
            output_predicted_samples[i] = MPSS.forward_model(
                np.log10(psd_posterior.post_samples[sample_idx])
                )
    
    elif psd_posterior.input_mode == 'gaussian-log10':
        output_predicted_samples = np.zeros((n_samples, len(output_measured)))
        for i in range(n_samples):
            output_predicted_samples[i] = MPSS.forward_model(
                np.log10(psd_posterior.get_samples(num=1).squeeze())
                )
    else:
        raise ValueError('Unknown PSDPosterior input mode.')
    
    # Put the reporting range back to what it was when calling this function
    if reporting_range == 'measured':
        psd_posterior.set_reporting_range('measured')
    
    output_predicted_median_noiseless = np.median(output_predicted_samples, axis=0)
    
    # Add counting noise
    output_predicted_samples_noisy = rng.poisson(lam=output_predicted_samples).astype(np.float64)
    # Add additive noise (electronic/environmental)
    noise_std = 2 + 0.01 * output_predicted_samples
    output_predicted_samples_noisy += noise_std * rng.normal(loc=0.0, scale=1.0,
                                                             size=output_predicted_samples.shape)
    
    # Highest density intervals
    CI_lo, CI_hi = highest_density_interval(output_predicted_samples_noisy.T, CI_coverage / 100)
    
    ax.semilogx(MPSS.d_m_data * 1e9, output_measured, 'kx', label='Observed counts')
    ax.fill_between(MPSS.d_m_data * 1e9,
                    CI_hi,
                    CI_lo,
                    alpha=0.25,
                    facecolor='C1',
                    label=f'{CI_coverage} % posterior predictive interval'
                    )
    ax.semilogx(MPSS.d_m_data * 1e9, output_predicted_median_noiseless, 'C1-',
                label='Median predicted signal (noiseless)')
    ax.legend()
    ax.grid('on')
    ax.set_title('Data fit')
    ax.set_xlabel('DMA output nominal diameter (nm)')
    ax.set_ylabel('Counts (#)')


def plot_Ntot_histogram(ax, Ntots, Ntot_true=None, xlimits=None, color='C0'):
    ''' Plot a histogram and some credible intervals of the sampled Ntot. '''
    Ntots = Ntots.squeeze()
    
    Ntot_low95, Ntot_high95 = highest_density_interval(Ntots, 0.95)
    
    if xlimits is None:
        plt_lo, plt_hi = highest_density_interval(Ntots, 0.99)
    else:
        plt_lo, plt_hi = xlimits
    
    counts, bins = np.histogram(Ntots, bins=100, range=(plt_lo, plt_hi), density=True)
    width = bins[1] - bins[0]
    ax.bar(bins[1:] - width, counts, width=width, edgecolor="white", color=color, alpha=0.8)
    ymin, ymax = ax.get_ylim()
    
    ax.vlines(x=[Ntot_low95, Ntot_high95],
               ymin=ymin, ymax=ymax * 1.15, colors='k', linestyle='--')
    
    ax.set_ylim((ymin, 1.3 * ymax))
    
    # Plot the CI ranges
    N95_diff = Ntot_high95 - Ntot_low95
    ax.plot(
        np.array([Ntot_low95 + 0.1 * N95_diff, Ntot_high95 - 0.1 * N95_diff]),
        1.2 * np.array([ymax, ymax]),
        'k--',
        linewidth=1.5
        )
    anno_args = {
        'ha': 'center',
        'va': 'center',
        'size': 10
    }
    ax.annotate('95 % CI',
                 xy=(0.5 * (Ntot_low95 + Ntot_high95), ymax * 1.23), **anno_args)
    anno_args['size'] = 16
    ax.annotate('[', xy=(Ntot_low95, 1.2 * ymax), **anno_args)
    ax.annotate(']', xy=(Ntot_high95, 1.2 * ymax), **anno_args)
    
    ax.set_title(r'Sampled $N_{tot}$ values and credible intervals')
    ax.set_xlabel(r'$N_{tot}$')
    ax.set_ylabel(r'$N_{tot}$ density')
    
    if Ntot_true is not None:
        low_y, high_y = ax.get_ylim()
        ax.plot([Ntot_true, Ntot_true], [0, 0.2 * high_y], 'k-', linewidth=5)
        ax.annotate(r'True $N_\mathrm{tot}$',
                     xy=(Ntot_true, 0.22 * high_y), ha='center', size=10)


def add_checkerboard_background(ax, check_size_px=8, light=0.92, dark=0.75, zorder=0):
    """
    Add a checkerboard background to a Matplotlib axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add the background to.
    check_size_px : int
        Size of each checker square in pixels.
    light : float
        Grey level for the light squares (0–1).
    dark : float
        Grey level for the dark squares (0–1).
    zorder : float
        Drawing order for the background axes.
    """

    # Create an inset axes that fills the entire parent axes
    ax_bg = ax.inset_axes([0, 0, 1, 1], transform=ax.transAxes, zorder=zorder)
    ax_bg.set_axis_off()

    # Store state on the axes object for the draw callback
    ax_bg._checker_params = {
        'check_size_px': check_size_px,
        'light': light,
        'dark': dark,
        'img_artist': None,
        'base_dpi': ax.get_figure().get_dpi(),  # screen DPI at creation time
    }
    def _update_checkerboard(event=None):
        """Regenerate the checkerboard image based on current pixel size of the axes."""
        params = ax_bg._checker_params
        
        # Get the bounding box of the background axes in pixels
        bbox = ax_bg.get_window_extent()
        w_px = max(int(np.ceil(bbox.width)), 1)
        h_px = max(int(np.ceil(bbox.height)), 1)
    
        # Scale checker size with DPI so squares look the same
        # physical size on screen and in saved files
        base_dpi = params['base_dpi']
        current_dpi = ax.get_figure().get_dpi()
        s = max(1, int(round(params['check_size_px'] * current_dpi / base_dpi)))
    
        if (w_px, h_px, s) == params.get('_last', (None, None, None)):
            return
        params['_last'] = (w_px, h_px, s)
        
        # Number of checker cells needed
        n_cols = max(int(np.ceil(w_px / s)), 1)
        n_rows = max(int(np.ceil(h_px / s)), 1)
        
        # Build checkerboard at cell resolution
        j = np.arange(n_cols)[None, :]
        i = np.arange(n_rows)[:, None]
        checker = ((i + j) % 2).astype(np.float64)
        
        # Map to grey levels: 0 -> light, 1 -> dark
        checker_grey = np.where(checker == 0, params['light'], params['dark'])
        # Stack to RGB
        checker_rgb = np.stack([checker_grey] * 3, axis=-1)
    
        if params['img_artist'] is None:
            params['img_artist'] = ax_bg.imshow(
                checker_rgb, origin='lower',
                aspect='auto',
                interpolation='nearest',
                extent=[0, 1, 0, 1],  # fill the axes in axes coordinates
            )
            ax_bg.set_xlim(0, 1)
            ax_bg.set_ylim(0, 1)
        else:
            params['img_artist'].set_data(checker_rgb)
            # Extent stays [0,1,0,1]; imshow stretches to fill

    # Draw once immediately (for initial render)
    _update_checkerboard()

    # Connect to resize and draw events so it updates on layout changes
    fig = ax.get_figure()
    fig.canvas.mpl_connect('resize_event', _update_checkerboard)
    fig.canvas.mpl_connect('draw_event', _update_checkerboard)

    # Make the foreground axes transparent and on top
    ax.set_zorder(zorder + 1)
    ax.patch.set_alpha(0)
