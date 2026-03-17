# -*- coding: utf-8 -*-

import yaml
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.ticker as tck
from matplotlib.tri import Triangulation
from matplotlib.gridspec import GridSpec
from joblib import dump, load
from tqdm import tqdm

from MPSS_UQ.particlesizers import MobilityParticleSizeSpectrometer, lpm_to_m3s
from MPSS_UQ.inversion import invert_psd, invert_dataset, smoothness_prior
from MPSS_UQ.analysis import (summarize_samples, total_concentration, relative_hdi_width,
                              concentration_in_range, geometric_mean_diameter, mode_diameter,
                              median_diameter, surface_area_concentration, volume_concentration,
                              condensation_sink, effective_diameter, geometric_std)
from MPSS_UQ.measurement_data import (generate_DMPS_measurement, compute_true_Ntot_in_range,
                                      MeasurementDataset)
from MPSS_UQ.plotfunctions import (plot_psd, plot_posterior_summary,
                                   highest_density_interval, plot_timeseries_2d,
                                   plot_system_matrix, plot_Ntot_histogram,
                                   plot_timeseries_1d, add_checkerboard_background,
                                   _my_format_colorbar, make_three_zone_cmap,
                                   )
from MPSS_UQ.aerosol import NORMAL_TEMPERATURE, NORMAL_PRESSURE
from visualize_LYF_model import evaluate_charging_probability_range
from compare_full_posterior_Laplace_approximation import compare_posterior_representations


''' A script to create and save all the figures in the paper. '''

FIGURES_DIR = 'paper_figures/'

SAVE_FIGURES = False
# SAVE_FIGURES = True
FIG_WIDTH_SINGLE = 6
FIG_WIDTH_DOUBLE = 2 * FIG_WIDTH_SINGLE + 0.5
FIG_HEIGHT = 4
DPI = 300
plt.rcParams['axes.xmargin'] = 0.02
plt.close('all')



# %% Fig. 0: The range of ion mobilities

# Plot the edges of the ion mobility area
def plot_mobility_area(ax):
    xs = np.array([1.05, 1.10, 1.70, 1.70, 1.05, 1.05])
    ys = np.array([1.10, 1.10, 1.70, 2.10, 2.10, 1.10])

    # Outline & fill
    line, = ax.plot(xs, ys, color='black', lw=1.5)
    ax.fill_between([1.05, 1.10, 1.70],
                    [1.10, 1.10, 1.70],
                    [2.10, 2.10, 2.10],
                    color=(0, 0, 0, 0.1))
    
    return line

fig0, ax0 = plt.subplots(1, 2, num=99, clear=True)
fig0.set_figwidth(FIG_WIDTH_DOUBLE)
fig0.set_figheight(FIG_HEIGHT)
ax0[0].set_title('a)', loc='left')
ax0[1].set_title('b)', loc='left')

ax0[0].set_xlabel(r'$Z^+$ (cm$^2$V$^{-1}$s$^{-1}$)')
ax0[0].set_ylabel(r'$Z^-$ (cm$^2$V$^{-1}$s$^{-1}$)')
ax0[0].set_title('Ion mobilities')

# Plot the edges of the ion mobility area
plot_mobility_area(ax0[0])
ax0[0].plot(np.array([1.0, 1.85]), np.array([1.0, 1.85]), '--k')
ax0[0].text(1.90, 1.82, r'$Z^- = Z^+$', rotation=45, ha='center', va='center')
ax0[0].grid('on')

# Plot measured mobility values (list from Tigges et al. (2015))
Z_pos = np.array([1.40, 1.20, 1.20, 1.15, 1.5, 1.55, 1.57, 1.6, 1.10, 1.10, 1.10, 1.15, 1.15,
                  1.65, 1.61, 1.27, 1.41, 1.44, 1.37, 1.40, 1.35, 1.40])
Z_neg = np.array([1.90, 1.35, 1.35, 1.39, 2.09, 1.69, 1.85, 1.68, 1.15, 1.30, 1.80, 1.425, 1.90,
                  2.09, 1.90, 1.61, 1.53, 1.94, 1.89, 1.90, 1.60, 1.60])

ax0[0].plot(Z_pos, Z_neg, 'o', markersize=7, color='C0', label='Measurements from\nliterature')
ax0[0].plot(1.35, 1.60, 'd', markersize=10, color='C0',
            markeredgecolor='black', markeredgewidth=1.2,
            label='Wiedensohler (1988)')
ax0[0].legend(loc='lower right')


from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
norm = Normalize(vmin=0, vmax=1)
sm = ScalarMappable(norm=norm, cmap='viridis')
cbar1 = fig0.colorbar(sm, ax=ax0[0])

ax0[1].set_title('Derived ion masses')
ax0[1].set_xlabel(r'$m^+$ (amu)')
ax0[1].set_ylabel(r'$m^-$ (amu)')
ax0[0].axis('equal')

# Constants for the mobility-to-mass mapping
a = -0.0347
b = -0.0376
c = 1.4662
def mobility_to_mass(Z):
    return np.exp((-b - np.sqrt(b**2 - 4 * a * (c - np.log(Z)))) / (2 * a))

# Derivative of the mapping
def dm_dZ(Z):
    sqrt_term = np.sqrt(b**2 - 4 * a * (c - np.log(Z)))
    m = mobility_to_mass(Z)
    return -m / (Z * sqrt_term)

Z_plus = np.linspace(1.05, 1.7, 100)
Z_minus = np.linspace(1.1, 2.1, 100)
Zp, Zm = np.meshgrid(Z_plus, Z_minus)
mask = Zp <= Zm

Zp = Zp[mask]
Zm = Zm[mask]

# Compute masses and Jacobian determinant
mp = mobility_to_mass(Zp)
mm = mobility_to_mass(Zm)
m_pos = mobility_to_mass(Z_pos)
m_neg = mobility_to_mass(Z_neg)
dmp_dZp = dm_dZ(Zp)
dmm_dZm = dm_dZ(Zm)
jacobian = np.abs(dmp_dZp * dmm_dZm)

# Compute density in mass space (inverse of Jacobian)
density = 1 / jacobian
density /= np.max(density)

# Create a triangulation for mesh-style plotting
tri = Triangulation(mp, mm)

# Plot the density using a filled triangular mesh
tpc = ax0[1].tripcolor(tri, density, shading='flat', cmap='binary', vmin=0, vmax=1)
plt.colorbar(tpc, label='Normalized mass density')
ax0[1].plot(mobility_to_mass(np.array([1.05, 1.10, 1.70, 1.70, 1.05, 1.05])),
            mobility_to_mass(np.array([1.10, 1.10, 1.70, 2.10, 2.10, 1.10])),
            color='black', lw=1.5)
ax0[1].plot(mobility_to_mass(np.array([1.05, 2.1])),
            mobility_to_mass(np.array([1.05, 2.1])),
            '--k')
ax0[1].plot(m_pos, m_neg, 'o', markersize=7, color='C1', label='Measurements')
ax0[1].plot(mobility_to_mass(1.35), mobility_to_mass(1.60), 'd', markersize=10, color='C1',
            markeredgecolor='black', markeredgewidth=1.2,
            label='Wiedensohler (1988)')

ax0[1].text(mobility_to_mass(1.90), mobility_to_mass(1.8), r'$m^- = m^+$',
            rotation=45, ha='center', va='center')

ax0[1].grid('on')
ax0[1].axis('equal')

fig0.tight_layout()

cbar1.ax.set_visible(False)  # Make the dummy colorbar invisible

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/mobilities_range.pdf', dpi=DPI)


# %% Figs. 1-3: Example of the uncertainty in charging probability

fig1, axs1 = plt.subplots(4, 2, num=1, clear=True)
fig1.set_figheight(3 * FIG_HEIGHT)
fig1.set_figwidth(FIG_WIDTH_DOUBLE)
# fig2, axs2 = plt.subplots(4, 2, num=2, clear=True)
# fig2.set_figheight(3 * FIG_HEIGHT)
# fig2.set_figwidth(FIG_WIDTH_DOUBLE)
# fig3, axs3 = plt.subplots(4, 2, num=3, clear=True)
# fig3.set_figheight(3 * FIG_HEIGHT)
# fig3.set_figwidth(FIG_WIDTH_DOUBLE)

evaluate_charging_probability_range(fig1, axs1, 'ion-mobility')
# evaluate_charging_probability_range(fig2, axs2, 'ion-ratio')
# evaluate_charging_probability_range(fig3, axs3, 'ion-mobility-ratio')

for fig in (fig1,):#, fig2, fig3):
    fig.suptitle(r'Charging probability $f_p$')
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(hspace=0.1)


# fig1.tight_layout()

if SAVE_FIGURES:
    plt.figure(1)
    plt.savefig(f'{FIGURES_DIR}/fp_range_mobility2.pdf', dpi=DPI)
    # plt.figure(2)
    # plt.savefig(f'{FIGURES_DIR}/fp_range_ratio.pdf', dpi=DPI)
    # plt.figure(3)
    # plt.savefig(f'{FIGURES_DIR}/fp_range_mobility_ratio.pdf', dpi=DPI)


# %% Ion mobilities for generating the data

true_ion_mobilities = 1e-4 * np.array([[1.15, 1.30], [1.4, 1.65], [1.10, 1.9], [1.55, 2.00]])

fig6, ax6 = plt.subplots(1, 1, num=6, clear=True)
fig6.set_figwidth(FIG_WIDTH_SINGLE)
fig6.set_figheight(FIG_HEIGHT)

ax6.set_xlabel(r'$Z^+$ (cm$^2$V$^{-1}$s$^{-1}$)')
ax6.set_ylabel(r'$Z^-$ (cm$^2$V$^{-1}$s$^{-1}$)')
ax6.set_title('Ion mobilities')

plot_mobility_area(ax6)

ax6.axis('equal')
# ax6.grid('on')

ax6.plot(1e4 * true_ion_mobilities[:, 0], 1e4 * true_ion_mobilities[:, 1],
         'o', markersize=7, color='C0', label='True ion mobilities')
ax6.plot(1.35, 1.60, 'x', markersize=8, color='k')
# ax6.legend(loc='lower right')

for i in range(true_ion_mobilities.shape[0]):
    if i in [0, 3]:
        ax6.text(1e4 * true_ion_mobilities[i, 0] * 1.01, 1e4 * true_ion_mobilities[i, 1] + 0.02,
                 f'Meas. {i+1}', ha='center', va='bottom')
    else:
        ax6.text(1e4 * true_ion_mobilities[i, 0] * 1.01, 1e4 * true_ion_mobilities[i, 1] + 0.02,
                 f'Meas. {i+1}', ha='left', va='bottom')
ax6.text(1.35 / 1.01, 1.60 + 0.02,
         'Inversion', ha='right', va='bottom')

# ax6.plot(1e4 * true_ion_mobilities[0, 0], 1e4 * true_ion_mobilities[0, 1],
#          'o', markersize=25, color='C1', markerfacecolor='none', markeredgewidth=5)
# ax6.set_xlim([0.95, 1.80])
# ax6.set_ylim([1.05, 2.15])

def add_right_labels_with_45deg(ax, points_xy, labels=None,
                                pad_frac=0.02, right_margin_frac=0.02,
                                pixel_gap=4, text_kwargs=None, line_kwargs=None):
    """
    Adds right-side labels sorted by point y (top→bottom). Each label is connected
    by a horizontal segment (from the label's left edge) and a ±45° diagonal that
    ends just short of the point (pixel_gap).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    points_xy : (N,2) array-like
        Data coordinates of points (x, y).
    labels : list of str or None
        If None, uses "(x.xx, y.yy)" format.
    pad_frac : float
        Fraction of y-range padding at top/bottom for label rows.
    right_margin_frac : float
        Fraction of x-range used as right margin and elbow→label margin.
    pixel_gap : int
        Pixel gap so the diagonal stops just before the point marker.
    text_kwargs : dict
        Extra kwargs for `ax.text` (merged with defaults).
    line_kwargs : dict
        Extra kwargs for `ax.plot` (merged with defaults).
    """
    import numpy as np

    points_xy = np.asarray(points_xy, dtype=float)
    N = points_xy.shape[0]
    if N == 0:
        return

    # Default labels as "(x.xx, y.yy)"
    if labels is None:
        labels = [f"({x:.2f}, {y:.2f})" for x, y in points_xy]
    else:
        assert len(labels) == N, "labels must match number of points"

    # Sort by point y (descending) so labels map top→bottom
    order = np.argsort(points_xy[:, 1])[::-1]
    pts_sorted = points_xy[order]
    labels_sorted = [labels[i] for i in order]

    # Evenly spaced label y-positions (top→bottom) within the y-limits, with padding
    ymin, ymax = ax.get_ylim()
    ymin *= 1.15
    ymax *= 0.91
    yr = ymax - ymin
    y_top = ymax - pad_frac * yr
    y_bot = ymin + pad_frac * yr
    y_rows = np.linspace(y_top, y_bot, N)

    # Preliminary label x position to the right of current xlim
    xmin, xmax = ax.get_xlim()
    xr = xmax - xmin
    prelim_x_label = xmax + right_margin_frac * xr

    # Elbow x positions to guarantee a 45° diagonal that hits the point exactly.
    # With labels to the right, make the elbow lie to the RIGHT of the point:
    #   x_elbow = x_point + |y_row - y_point|
    # This automatically enforces ±45°; sign (up/down) follows y_row - y_point.
    dy = y_rows - pts_sorted[:, 1]
    x_elbows = pts_sorted[:, 0] + np.abs(dy)

    # Choose final label x so that every elbow lies to its left (horizontal goes from label → elbow)
    x_label = max(prelim_x_label, np.max(x_elbows) + right_margin_frac * xr) * 1.15

    # Drawing styles
    if text_kwargs is None: text_kwargs = {}
    if line_kwargs is None: line_kwargs = {}
    text_style = dict(ha='left', va='center', color='black', fontsize=10, clip_on=False)
    text_style.update(text_kwargs)
    line_style = dict(color='black', lw=0.9, solid_capstyle='round', clip_on=False)
    line_style.update(line_kwargs)

    # Ensure a renderer for pixel-based computations
    fig = ax.figure
    fig.canvas.draw()

    # Helper: end the diagonal with a fixed pixel gap before the point
    def end_with_pixel_gap(xe, ye, xp, yp, gap_px=pixel_gap):
        trans = ax.transData
        inv = trans.inverted()
        p_elbow = trans.transform([xe, ye])
        p_point = trans.transform([xp, yp])
        v = p_point - p_elbow
        L = np.hypot(v[0], v[1])
        if L == 0:
            return xp, yp
        u = v / L
        p_end = p_point - gap_px * u
        x_end, y_end = inv.transform(p_end)
        return x_end, y_end

    # Draw labels and connectors
    text_artists = []
    for (xp, yp), yl, xe, lab in zip(pts_sorted, y_rows, x_elbows, labels_sorted):
        # Label text (anchored at LEFT edge, vertically centered = "left-middle")
        t = ax.text(x_label, yl, lab, **text_style)
        text_artists.append(t)

        # Horizontal segment: from label's left edge to elbow
        ax.plot([x_label * 0.99, xe], [yl, yl], **line_style)

        # ±45° diagonal: elbow → near-point (with small pixel gap)
        x_end, y_end = end_with_pixel_gap(xe, yl, xp, yp, pixel_gap)
        ax.plot([xe, x_end], [yl, y_end], **line_style)
    
    # Add the label
    t = ax.text(x_label*0.95, ymax * 1.05, r'Mobilities $(Z^+, Z^-)$:', **text_style)
    
    # Add an underline for the label
    # Get the bounding box of the text in pixel coordinates
    bbox = t.get_window_extent(renderer=fig.canvas.get_renderer())
    
    # Transform pixel → data coordinates
    inv = ax.transData.inverted()
    (x0, y0) = inv.transform((bbox.x0, bbox.y0))
    (x1, y1) = inv.transform((bbox.x1, bbox.y0))
    
    # Small offset below the text (in data units)
    margin = 0.005 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    
    # Draw the underline
    ax.plot([x0, x1], [y0 - margin, y0 - margin], color='black', lw=1.0)

    text_artists.append(t)

    # Optionally extend xlim to ensure labels and text are visible
    renderer = fig.canvas.get_renderer()
    max_text_w_px = 0
    for t in text_artists:
        bbox = t.get_window_extent(renderer=renderer)
        max_text_w_px = max(max_text_w_px, bbox.width)

    # Convert max text width (px) to data units to extend xlim safely
    y_mid = 0.5 * (ymin + ymax)
    x0_px = ax.transData.transform([0, y_mid])[0]
    x1_px = ax.transData.transform([1, y_mid])[0]
    px_per_data = (x1_px - x0_px) if (x1_px - x0_px) != 0 else 1.0
    extra_right = (max_text_w_px / px_per_data) + right_margin_frac * xr

    ax.set_xlim(xmin, max(ax.get_xlim()[1], x_label + extra_right))


# Points used for annotations
points_xy = np.vstack([
    1e4 * true_ion_mobilities,   # 4 measured points
    [1.35, 1.60]                 # inversion point
])

# Coordinate-only labels like "(1.35, 1.60)"
coord_labels = [f"({x:.2f}, {y:.2f})" for x, y in points_xy]

# Add right-side labels with 45° connectors (auto slope per D3)
add_right_labels_with_45deg(
    ax6,
    points_xy,
    labels=coord_labels,
    pad_frac=0.04,
    # right_margin_frac=0.05,  # spacing to the right
    pixel_gap=12,             # gap before the point marker
    text_kwargs=dict(fontsize=10),
    line_kwargs=dict(lw=1.1)
)

fig6.tight_layout()

plt.pause(0.1)

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/datagen_ion_mobilities.pdf', dpi=DPI)


# %% Fig. 3.5: Instrument matrix

with open("../examples/DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)
DMPS_prop = DMPS_prop['UEF-A20']

DMPS_prop['d_m_data'] = np.geomspace(10e-9, 700e-9, num=30)
DMPS_prop['max_charge'] = 4
DMPS_prop['charging_model'] = 'LYF-interp'
DMPS = MobilityParticleSizeSpectrometer(DMPS_prop, n_bins=500)
DMPS.set_charger_properties(1.35e-4, 1.60e-4)
DMPS.set_operating_conditions(NORMAL_TEMPERATURE, NORMAL_PRESSURE)

fig, ax = plt.subplots(1,1 , num=11, clear=True)
fig.set_figwidth(FIG_WIDTH_SINGLE)
fig.set_figheight(FIG_HEIGHT)
plot_system_matrix(ax, DMPS)

ax.axvline(
    x=700,
    color='white',
    linestyle=':',
    linewidth=1.5,
    zorder=10   # ensures it draws on top of the pcolormesh
)

fig.tight_layout()

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/example_system_matrix.png', dpi=DPI)



# %% Fig. 4: Inversion with synthetic data

# Set up the DMPS used to create measurement data
# Load a DMPS configuration from file
with open("../examples/DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)
DMPS_prop = DMPS_prop['UEF-A20']

# Mobility diameters the DMA measures
DMPS_prop['d_m_data'] = np.geomspace(10e-9, 800e-9, num=30)

# Properties of the data generating DMPS
DMPS_prop['max_charge'] = 10
DMPS_prop['charging_model'] = 'LYF-interp'

# Properties of the inversion DMPS
n_inversion_bins = 100
DMPS_prop_inv = DMPS_prop.copy()
DMPS_prop_inv['max_charge'] = 8
DMPS_prop_inv['charging_model'] = 'LYF-interp'  # LYF charging model (interpolated)
# DMPS_prop_inv['charging_model'] = 'Wiedensohler'  # Wiedensohler charging model

DMPS = MobilityParticleSizeSpectrometer(DMPS_prop_inv,
                                        n_bins=n_inversion_bins,
                                        )

# Select the ion properties
pos_mob_inv = 1.35e-4
neg_mob_inv = 1.60e-4
DMPS.set_charger_properties(pos_mob_inv, neg_mob_inv)
DMPS.set_operating_conditions(NORMAL_TEMPERATURE, NORMAL_PRESSURE)

def create_subplots(num):
    fig, axs = plt.subplots(2, 2, num=num, clear=True)
    fig.set_figwidth(FIG_WIDTH_DOUBLE)
    fig.set_figheight(FIG_HEIGHT * 2)
    return fig, axs.flatten()

def create_mobility_inset(ax, pos_mob, neg_mob, pos_mob_inv=None, neg_mob_inv=None):
    ax_in = ax.inset_axes([0.62, 0.40, 0.35, 0.50]) # [left, bottom, width, height]

    outline = plot_mobility_area(ax_in)
    ax_in.set_xticks([])
    ax_in.set_yticks([])
    ax_in.axis('equal')
    ax_in.plot(1e4 * pos_mob, 1e4 * neg_mob,
             'o', markersize=7, color='k')
    if pos_mob_inv is not None and neg_mob_inv is not None:
        ax_in.plot(1e4 * pos_mob_inv, 1e4 * neg_mob_inv,
                 'x', markersize=7, color='k')
    # ax_in.set_facecolor((1, 1, 1, 0.9))
    ax_in.set_xlabel(r'$Z^+$')
    ax_in.set_ylabel(r'$Z^-$')
    
    return ax_in, outline

def add_marginalization_centers(ax, outline):
    xs = np.asarray(outline.get_xdata())
    ys = np.asarray(outline.get_ydata())

    # Build Path for point-in-polygon test
    from matplotlib.path import Path
    poly = Path(list(zip(xs, ys)))

    xmin, xmax = xs.min(), xs.max()
    ymin, ymax = ys.min(), ys.max()
    
    dx = dy = 0.07

    # Cell centers on a regular lattice
    xc = np.arange(xmin + dx/2, xmax, dx)
    yc = np.arange(ymin + dy/2, ymax, dy)
    Xc, Yc = np.meshgrid(xc, yc)
    pts = np.column_stack([Xc.ravel(), Yc.ravel()])
    inside = poly.contains_points(pts)
    Xci, Yci = pts[inside, 0], pts[inside, 1]

    # Plot the centers
    ax.plot(Xci, Yci, 'x', color='0.3', ms=3, alpha=0.5)


fig1, axs1 = create_subplots(12)
fig1.suptitle('PSD estimates, no charging uncertainty considered')
fig2, axs2 = create_subplots(13)
fig2.suptitle('PSD estimates, ion mobility marginalized')

ylims = np.array([0, 2.5e4])

def plot_and_format_2x2(axs, i, measurement, psd_posterior, CI_coverage, ylims):
    plot_posterior_summary(axs[i], psd_posterior, CI_coverage)
    plot_psd(axs[i], measurement.d_m_true, measurement.N_true, '--', color='k', label='Truth')
    if i in [0, 1]:
        axs[i].tick_params(labelbottom=False)  # hides x-axis labels
        axs[i].set_xlabel("")
    if i in [1, 3]:
        axs[i].tick_params(labelleft=False)
        axs[i].set_ylabel("")
    if i != 0:
        axs[i].legend().remove()
    else:
        axs[i].legend(loc='upper left')
    
    title = rf"Measurement {i+1}"
    axs[i].text(0.795, 0.97, title, transform=axs[i].transAxes,
            fontsize=12,
            # fontweight='bold',
            va='top',
            ha='center')
        
    axs[i].set_yscale('linear')
    axs[i].set_xlim([psd_posterior.d_m[0] * 1e9, psd_posterior.d_m[-1] * 1e9])
    xticks = [10, 20, 50, 100, 200, 500]        
    axs[i].set_xticks(xticks)
    if i in [2, 3]:
        axs[i].set_xticklabels([str(t) for t in xticks])
    if i in [0, 1]:
        axs[i].set_xticklabels([])
    axs[i].set_ylim([ylims[0], ylims[1]])
    axs[i].set_xlim([10, 800])
    axs[i].grid('on')

def postprocess_2x2(fig):
    fig.subplots_adjust(wspace=0.02)
    fig.subplots_adjust(hspace=0.04)

hist_xmin = float('inf')
hist_xmax = -float('inf')
hist_ymax = 0
cases = []
for i in tqdm(range(4), desc='Inverting synthetic Urban scenarios'):
    
    # Set ion properties for the measurement
    pos_ion_mobility = true_ion_mobilities[i, 0]
    neg_ion_mobility = true_ion_mobilities[i, 1]
    
    # Create the measurement
    measurement = generate_DMPS_measurement(DMPS_prop,
                                            scenario='Urban',
                                            pos_ion_mobility=pos_ion_mobility,
                                            neg_ion_mobility=neg_ion_mobility,
                                            rng_seed=6,
                                            )
    
    # Compute the MAP estimate and Laplace approximation for the posterior covariance
    psd_posterior = invert_psd(DMPS, measurement)

    psd_posterior_marg = invert_psd(
        DMPS, measurement,
        marginalize_ion_mobility=True,
        # The values below are just for higher-fidelity figures for
        # publishing. Not needed for normal use.
        num_samples=100000,
        marginalization_grid='fine'
        )
    
    Ntot_true = compute_true_Ntot_in_range(measurement, psd_posterior.d_m)
    Ntot_samples = psd_posterior.propagate_to(total_concentration, num=20000)
    Ntot_samples_marg = psd_posterior_marg.propagate_to(total_concentration)
    
    cases.append(dict(non=Ntot_samples.squeeze(),
                      marg=Ntot_samples_marg.squeeze(),
                      true=Ntot_true,
                      title=f"Measurement {i+1}"
                      ))
    
    CI_coverage = 95
    
    plot_and_format_2x2(axs1, i, measurement, psd_posterior, CI_coverage, ylims)
    plot_and_format_2x2(axs2, i, measurement, psd_posterior_marg, CI_coverage, ylims)
    create_mobility_inset(axs1[i], pos_ion_mobility, neg_ion_mobility,
                          pos_mob_inv=pos_mob_inv, neg_mob_inv=neg_mob_inv)
    ax_in, outline = create_mobility_inset(axs2[i], pos_ion_mobility, neg_ion_mobility)
    add_marginalization_centers(ax_in, outline)
    postprocess_2x2(fig1)
    postprocess_2x2(fig2)

all_samples = np.concatenate([np.concatenate([c['non'], c['marg']]) for c in cases])
bins = np.histogram_bin_edges(all_samples, bins=150)

fig3, axs3 = plt.subplots(2, 2, figsize=(FIG_WIDTH_DOUBLE, 2 * FIG_HEIGHT), num=14, clear=True)
axs3 = axs3.ravel()

for i, c in enumerate(cases):
    ax = axs3[i]

    # Histogram counts (shared edges)
    c_non, edges = np.histogram(c['non'], bins=bins, density=False)
    c_mrg, _     = np.histogram(c['marg'], bins=edges, density=False)

    # Unit-peak normalization (shape-only: each curve peaks at 1)
    peak_non = c_non.max() if c_non.size and c_non.max() > 0 else 1.0
    peak_mrg = c_mrg.max() if c_mrg.size and c_mrg.max() > 0 else 1.0
    h_non = c_non / peak_non
    h_mrg = c_mrg / peak_mrg

    # # Stairs (step) plots
    # ax.stairs(h_mrg, edges, label='Marginalized',      color='C1', linewidth=1.8, linestyle='-')
    # ax.stairs(h_non, edges, label='Non-marginalized',  color='C0', linewidth=1.8)
    
    # Historgam (bar) plots
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    ax.bar(centers, h_mrg, width=widths,
           color='C0', alpha=0.4, label='Marginalized', edgecolor='C0', linewidth=0.5)
    ax.bar(centers, h_non, width=widths,
           color='C1', alpha=0.6, label='Non-marginalized', edgecolor='C1')
    
    # True N_tot reference (scale relative to unit-peak axis)
    ymax = 1.0
    ax.vlines(c['true'], 0, 0.2 * ymax, color='k', linewidth=4)
    ax.annotate(r'True $N_\mathrm{tot}$',
                xy=(c['true'], 0.22 * ymax),
                ha='center',
                va='bottom',
                fontsize=9,
                bbox=dict(
                        facecolor='white',     # background color
                        edgecolor='none',      # no border (or use a color)
                        alpha=0.7,             # transparency
                        pad=1                  # small padding around text
                        )
                )

    ax.grid(True, alpha=0.25)
    ax.text(0.88, 0.97, c['title'], transform=ax.transAxes,
            fontsize=12,
            va='top',
            ha='center')
    
    # Add the 95 % credible intervals
    low, hi = highest_density_interval(c['non'], 0.95)
    low_marg, hi_marg = highest_density_interval(c['marg'], 0.95)
    
    multiplier = (hi_marg - low_marg) / (hi - low)
    print(multiplier)
    
    ax.annotate("",
                xy=(low_marg, 1.1 * ymax), 
                xytext=(hi_marg, 1.1 * ymax),
                arrowprops=dict(arrowstyle="|-|", color="C0", lw=1.8)
                )
    ax.annotate(r'$95$ % CI', xy=(low_marg + 200, 1.11 * ymax),
                ha='left', va='bottom', fontsize=10, color='C0')
    ax.annotate("",
                xy=(low, 1.25*ymax), 
                xytext=(hi, 1.25*ymax),
                arrowprops=dict(arrowstyle="|-|", color="C1", lw=1.8)
                )
    ax.annotate(r'$95$ % CI', xy=(low - 200, 1.28 * ymax),
                ha='right', va='center', fontsize=10, color='C1')
    
    ax.set_ylim(0, 1.35)  # same y-range for all (unit peak)

# X-limits from the common bins
xlim = (bins[0], bins[-1])
for ax in axs3:
    ax.set_xlim(xlim)

# Y-limits: take global max after plotting
ymax_global = max(ax.get_ylim()[1] for ax in axs3)
for ax in axs3:
    ax.set_ylim(0, ymax_global)

# Shared labels: left column gets y-labels, bottom row gets x-labels
for i, ax in enumerate(axs3):
    # r, c = divmod(i, 2)
    if i in [2, 3]:
        ax.set_xlabel(r'$N_\mathrm{tot}$')
    if i in [0, 2]:
        ax.set_ylabel('Relative height')
    else:
        ax.tick_params(labelleft=False)
    if i in [0, 1]:
        ax.tick_params(labelbottom=False)

handles, labels = axs3[-1].get_legend_handles_labels()
fig3.legend(handles, labels,
           ncol=2,
           loc='upper center',
           frameon=False,
           bbox_to_anchor=(0.5, 0.99),
           fontsize=11)

fig3.tight_layout()
fig3.subplots_adjust(top=0.88, wspace=0.02, hspace=0.04)
fig3.suptitle('Total particle numbers: marginalized vs non-marginalized', y=0.93)

plt.pause(0.1)
plt.draw()

if SAVE_FIGURES:
    plt.figure(num=12)
    plt.savefig(f'{FIGURES_DIR}/synth_nonmarginalized.pdf', dpi=DPI)
    plt.figure(num=13)
    plt.savefig(f'{FIGURES_DIR}/synth_marginalized.pdf', dpi=DPI)
    plt.figure(num=14)
    plt.savefig(f'{FIGURES_DIR}/synth_Ntots.pdf', dpi=DPI)



#%%  Other simulated cases

with open("../examples/DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)
DMPS_prop = DMPS_prop['UEF-A20']

# Mobility diameters the DMA measures
DMPS_prop['d_m_data'] = np.geomspace(10e-9, 800e-9, num=30)

# Properties of the data generating DMPS
DMPS_prop['max_charge'] = 10
DMPS_prop['charging_model'] = 'LYF-interp'
DMPS_prop['CPC_measuring_time'] = 5
# Properties of the inversion DMPS
n_inversion_bins = 100
DMPS_prop_inv = DMPS_prop.copy()
DMPS_prop_inv['max_charge'] = 8
DMPS_prop_inv['charging_model'] = 'LYF-interp'  # LYF charging model (interpolated)

DMPS = MobilityParticleSizeSpectrometer(DMPS_prop_inv,
                                         n_bins=n_inversion_bins,
                                         )

# Select the ion properties
pos_ion_mobility = 1.35e-4
neg_ion_mobility = 1.60e-4
DMPS.set_charger_properties(pos_ion_mobility, neg_ion_mobility)
DMPS.set_operating_conditions(NORMAL_TEMPERATURE, NORMAL_PRESSURE)

scenarios = ['Rural', 'Remote continental', 'Scaled down rural', 'Trimodal nucleation event',
             'Asymmetric', 'Irregular', 'Polar', 'Polar']

# Figure: 3 rows × 2 columns (vertical grid)
fig, axs = plt.subplots(4, 2, figsize=(FIG_WIDTH_DOUBLE, 2.75 * FIG_HEIGHT), num=31, clear=True)
axs = axs.ravel()

handles_for_legend = None

# Use consistent x‑ticks across all panels
xticks = [10, 20, 50, 100, 200, 500]

# Optional: panel letters
panel_labels = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)',
                '(g)', '(h)',
                ]

for i, scenario in tqdm(enumerate(scenarios), desc='Inverting other synthetic scenarios'):
    
    measurement = generate_DMPS_measurement(
        DMPS_prop,
        scenario=scenario,
        pos_ion_mobility=pos_ion_mobility,
        neg_ion_mobility=neg_ion_mobility,
        rng_seed=i
    )
    
    psd_posterior = invert_psd(
        DMPS, measurement,
        marginalize_ion_mobility=True,
        num_samples=10000,
        # marginalization_grid='fine'
    )
    
    ax = axs[i]
    plot_posterior_summary(ax, psd_posterior, CI_coverage=95)

    mask = (measurement.d_m_true >= psd_posterior.d_m[0]) & \
           (measurement.d_m_true <= psd_posterior.d_m[-1])
    plot_psd(ax, measurement.d_m_true[mask], measurement.N_true[mask],
             linestyle='--', color='k', linewidth=1.2, label='Truth')
    
    ax.set_yscale('linear')
    ax.set_xlim([psd_posterior.d_m[0] * 1e9, psd_posterior.d_m[-1] * 1e9])
    ax.grid('on')
    
    # Scale ylim better for some cases
    ymin, ymax = ax.get_ylim()
    post_median, _, _ = psd_posterior.summary()
    post_median /= psd_posterior.binwidth
    if ymax > 4 * np.max(post_median):
        span = 4 * np.max(post_median)
        ax.set_ylim([-0.05 * span, span])
    if i == 6 or i == 7:
        span = 100
        ax.set_ylim([-0.05 * span, span])

    ax.set_xscale('log')
    ax.grid(True, alpha=0.3)
    # Rename some things
    if i == 7:
        scenario = scenario + ' (different noise realization)'
    ax.set_title(f"{panel_labels[i]} {scenario}", fontsize=12, loc='left')  # panel letter + title
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])

    if i != 0:
        ax.legend().set_visible(False)
    
    if i < 6:
        ax.set_xlabel('')
        ax.set_xticklabels([])

fig.tight_layout()
fig.suptitle('PSD estimates (charging marginalized) for different aerosol scenarios',
             y=0.96, fontsize=13)
fig.subplots_adjust(top=0.90, wspace=0.18, hspace=0.2)

if SAVE_FIGURES:
    plt.figure(num=31)
    plt.savefig(f'{FIGURES_DIR}/synth_morecases.pdf', dpi=DPI)



# %% Real data inversions

with open("../examples/DMPS_properties.yaml", "r") as f:
    DMPS_prop = yaml.safe_load(f)
DMPS_prop = DMPS_prop['UEF-A20']

raw_data = pd.read_csv('../examples/realdata_UEF_test/UEF_DMPS_lvl0.txt.gz', sep='\t')

concentrations = raw_data.filter(like="conc_").to_numpy()
d_m_data = raw_data.filter(like="dmed_").iloc[0].to_numpy()
temperatures = raw_data['t_sam'].to_numpy()
pressures = raw_data['p_sam'].to_numpy()
pressures *= 1e2  # Convert from hPa to Pa
datetimes = pd.to_datetime(raw_data["start_time"], utc=True).dt.tz_convert(None).to_numpy()

# Convert concentration into counts
sample_flow = DMPS_prop['Qa'] * lpm_to_m3s * 1e6
counts = concentrations * sample_flow * DMPS_prop['CPC_measuring_time']

dataset = MeasurementDataset(datetimes, d_m_data, counts, 'counts', temperatures, pressures)
# dataset = dataset.between_times('2024-12-10T00:00:00', '2024-12-25T23:59:59')
# dataset = dataset.between_times('2024-12-10T00:00:00', '2024-12-10T23:59:59')
DMPS_prop['d_m_data'] = d_m_data
DMPS_prop['charging_model'] = 'LYF-interp'
DMPS_prop['max_charge'] = 10
DMPS = MobilityParticleSizeSpectrometer(DMPS_prop,
                                        n_bins=70,
                                        )

# Set ion mobilities for cases where we don't use marginalization
DMPS.set_charger_properties(1.35e-4, 1.60e-4)

# Carry out the inversions for the whole dataset
psd_posteriors_marg = invert_dataset(DMPS,
                                 dataset,
                                 marginalize_ion_mobility=True,
                                 parallel=True,
                                 )

psd_posteriors_nonmarg = invert_dataset(DMPS,
                                    dataset,
                                    marginalize_ion_mobility=False,
                                    )

# # Save (and compress)
# dump(psd_posteriors_marg, "psd_posteriors_marg.joblib", compress=("gzip", 3))
# dump(psd_posteriors_nonmarg, "psd_posteriors_nonmarg.joblib", compress=("gzip", 3))

# # Load later
# psd_posteriors_marg = load("psd_posteriors_marg.joblib")
# psd_posteriors_nonmarg = load("psd_posteriors_nonmarg.joblib")


#%%
CI_coverage = 95
medians, CI_lower, CI_upper = psd_posteriors_marg.summary(coverage=CI_coverage, n_jobs=10)
medians_nonmarg, CI_lower_nonmarg, CI_upper_nonmarg = psd_posteriors_nonmarg.summary(
    coverage=CI_coverage, n_jobs=10)

#%% Check the ratio of marginalized to nonmarginalized credible interval widths and Ntot ones

width_marg = (CI_upper - CI_lower) / medians
width_nonmarg = (CI_upper_nonmarg - CI_lower_nonmarg) / medians_nonmarg
d_m = psd_posteriors_marg[0].d_m  # d_m of the stored results
binwidth = np.log10(d_m[1]) - np.log10(d_m[0])
plt.figure(), plt.loglog(medians / binwidth, (width_marg / width_nonmarg), '.k', alpha=0.01)
plt.xlabel('concentration dN/dlogdp')
plt.ylabel('marginalized CI_width / nonmarg CI_width')
plt.grid('on', which='both')
# Above, we see ratios < 1, which should be "impossible". These result probably mostly from the
# Laplace approximation sometimes inflating the upper boundary of the (nonmarginalized) credible
# intervals, which does not happen in the mixture of gaussians (not as much at least). Therefore,
# we can exclude these ratios from further analysis.
ratios = (width_marg / width_nonmarg).ravel()
ratios_clip = ratios[ratios >= 1]

plt.figure(), plt.hist(ratios, 1000)
plt.yscale('log')
plt.figure(), plt.loglog((medians / binwidth).ravel()[ratios >= 1], ratios_clip, '.k', alpha=0.01)
plt.xlabel('concentration dN/dlogdp')
plt.ylabel('marginalized CI_width / nonmarg CI_width')
plt.grid('on', which='both')

print(f'Mean W_marg / W_nonmarg: {ratios_clip.mean() : .3f}')
np.percentile(ratios_clip, (1, 99))

Ntot_marg_all = psd_posteriors_marg.propagate_to(total_concentration)
Ntot_nonmarg_all = psd_posteriors_nonmarg.propagate_to(total_concentration)

nt_mean_marg_all, nt_ci_lo_all_marg, nt_ci_hi_all_marg = summarize_samples(Ntot_marg_all, coverage=95)
nt_mean_nonmarg_all, nt_ci_lo_all_nonmarg, nt_ci_hi_all_nonmarg = summarize_samples(Ntot_nonmarg_all, coverage=95)

nt_CI_width_all_marg =  nt_ci_hi_all_marg - nt_ci_lo_all_marg
nt_CI_width_all_nonmarg =  nt_ci_hi_all_nonmarg - nt_ci_lo_all_nonmarg
nt_CI_ratio_all = nt_CI_width_all_marg / nt_CI_width_all_nonmarg
plt.figure(), plt.plot(nt_mean_marg_all, nt_CI_ratio_all, '.k', alpha=0.1)
plt.figure(), plt.hist(nt_CI_ratio_all, 20)
print(f'Mean ntot_width_marg / ntot_width_nonmarg: {nt_CI_ratio_all.mean() : .3f}')
np.percentile(nt_CI_ratio_all, (1, 99))

#%%
# fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(FIG_WIDTH_DOUBLE, 3 * FIG_HEIGHT),
#                         num=17, clear=True)
fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(FIG_WIDTH_DOUBLE, 2 * FIG_HEIGHT),
                        num=17, clear=True)

d_m = psd_posteriors_marg[0].d_m  # d_m of the stored results
binwidth = np.log10(d_m[1]) - np.log10(d_m[0])

# Subplot 1: Posterior medians
Z = medians.T / binwidth
plot_timeseries_2d(axs[0], psd_posteriors_marg.datetimes, d_m, Z,
                cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                )
axs[0].set_title(r'a) Posterior median', loc='left')


# Subplot 2: Uncertainties as relative CI width
W = relative_hdi_width(medians, CI_lower, CI_upper).T
plot_timeseries_2d(axs[1], psd_posteriors_marg.datetimes, d_m,
                   W,
                cbar_label=r'Relative width',
                cmap='inferno_r',
                cbar_as_perc=True,
                color_scale='log_3zone',
                vmax=100,
                cbar_extend='max',
                )
axs[1].set_title(
    f'b) Posterior uncertainty (relative width of {CI_coverage} % credible interval)', loc='left'
    )


# Subplot 3: Ntots
Ntots_samples = psd_posteriors_marg.propagate_to(total_concentration)
plot_timeseries_1d(axs[2], psd_posteriors_marg.datetimes, Ntots_samples,
                   coverage=CI_coverage,
                   ymin=0,
                   )
dm0 = 10**(np.log10(d_m[0]) - 0.5 * binwidth) * 1e9
dm1 = 10**(np.log10(d_m[-1]) + 0.5 * binwidth) * 1e9
axs[2].set_title(fr'c) Total particle number between [{dm0 : .1f}, {dm1 : .1f}] nm', loc='left')

# Create and hide a dummy colorbar to make the width equal to the other plots
import matplotlib.cm as cm
norm = colors.Normalize(vmin=0, vmax=1)
sm = cm.ScalarMappable(norm=norm, cmap='viridis')
cbar = fig.colorbar(sm, ax=axs[2], pad=0.02)
cbar.remove()

fig.tight_layout()
fig.subplots_adjust(right=1.07)

plt.show()

if SAVE_FIGURES:
    plt.figure(num=17)
    plt.savefig(f'{FIGURES_DIR}/realdata_whole_marg.png', dpi=DPI)


#%% An example of the transparency plot, PSD estimate with uncertainty as the alpha channel

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(FIG_WIDTH_DOUBLE, 0.75 * FIG_HEIGHT),
                        num=199, clear=True)

W_low = 0.5  # width below which estimate is ''accurate'' (alpha == 1)
W_high = 10  # width above which estimate is ''inaccurate'' (alpha == 0)
W_clipped = np.clip(W, W_low, W_high)
alpha = 1 - (W_clipped - W_low) / (W_high - W_low)

# Draw empty mesh, then set RGBA facecolors
im = plot_timeseries_2d(ax, psd_posteriors_marg.datetimes, d_m, Z,
                      cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                      alpha=alpha,
                      )
add_checkerboard_background(ax, check_size_px=10, light=0.92, dark=0.75)
ax.set_title('Posterior median with uncertainty as transparency', loc='left')
fig.tight_layout()
fig.subplots_adjust(right=1.07)

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/transparency_example_plot.png', dpi=DPI)


#%%  Shorter time plots

# day1 = ('2024-12-10T0:00:00', '2024-12-10T23:59:59')
day1 = ('2024-12-10T06', '2024-12-11T06')
day2 = ('2024-12-25T00:00:00', '2024-12-25T23:59:59')

# Reuse the same plotting script for two different days
for i, day in enumerate((day1, day2)):
# if True:
#     i=0
#     day=day1
    day_marg = psd_posteriors_marg.between_times(day[0], day[1])
    day_nom = psd_posteriors_nonmarg.between_times(day[0], day[1])
    
    medians_marg, CI_lo_marg, CI_up_marg = day_marg.summary(coverage=CI_coverage)
    medians_nom, CI_lo_nom, CI_up_nom = day_nom.summary(coverage=CI_coverage)
    
    Ntots_marg = day_marg.propagate_to(total_concentration)
    Ntots_nom = day_nom.propagate_to(total_concentration)
    Nt_marg, Nt_low_marg, Nt_high_marg = summarize_samples(Ntots_marg, coverage=95)
    Nt_nom, Nt_low_nom, Nt_high_nom = summarize_samples(Ntots_nom, coverage=95)
    
    fig = plt.figure(num=18, clear=True, figsize=(FIG_WIDTH_DOUBLE, 2 * FIG_HEIGHT))
    
    # Grid: 3 rows x 3 cols, last colum for color bars
    gs = GridSpec(nrows=3, ncols=3, figure=fig, width_ratios=[1.0, 1.0, 0.02])
    
    ax_psd_marg = fig.add_subplot(gs[0, 0])
    ax_psd_nom  = fig.add_subplot(gs[0, 1])
    ax_psd_cbar = fig.add_subplot(gs[0, 2])
    ax_w_marg   = fig.add_subplot(gs[1, 0])
    ax_w_nom    = fig.add_subplot(gs[1, 1])
    ax_w_cbar   = fig.add_subplot(gs[1, 2])
    ax_nt_marg  = fig.add_subplot(gs[2, 0])
    ax_nt_nom   = fig.add_subplot(gs[2, 1])
    ax_nt_cbar  = fig.add_subplot(gs[2, 2])
    
    Z_marg = (medians_marg.T) / binwidth
    Z_nom  = (medians_nom.T) / binwidth
    
    W_marg = relative_hdi_width(medians_marg, CI_lo_marg, CI_up_marg).T
    W_nom = relative_hdi_width(medians_nom, CI_lo_nom, CI_up_nom).T
    
    # Common colors for the side by side plots
    vmin_psd = np.nanpercentile(np.hstack([Z_marg.ravel(), Z_nom.ravel()]), 0.1)
    vmax_psd = np.nanpercentile(np.hstack([Z_marg.ravel(), Z_nom.ravel()]), 99.9)
    norm_psd = mpl.colors.LogNorm(vmin=vmin_psd, vmax=vmax_psd)
    cmap_psd = mpl.cm.viridis
    
    vmin_w = np.nanpercentile(np.hstack([W_marg.ravel(), W_nom.ravel()]), 0.1)
    vmax_w = 100 #np.nanpercentile(np.hstack([W_marg.ravel(), W_nom.ravel()]), 99.9)
    # norm_w = mpl.colors.LogNorm(vmin=vmin_w, vmax=vmax_w)
    # cmap_w = mpl.cm.inferno_r
    cmap_w = make_three_zone_cmap(vmin=vmin_w, vmid=10, vmax=vmax_w)
    norm_w = colors.LogNorm(vmin=vmin_w, vmax=vmax_w)
    
    
    plot_timeseries_2d(ax_psd_marg, day_marg.datetimes, d_m, Z_marg, show_cbar=False,
                    vmin=vmin_psd, vmax=vmax_psd,
                    )
    ax_psd_marg.set_title(r'a) Posterior median', loc='left')
    ax_psd_marg.set_xlabel('')
    plot_timeseries_2d(ax_psd_nom, day_nom.datetimes, d_m, Z_nom, show_cbar=False,
                    vmin=vmin_psd, vmax=vmax_psd,
                    )
    ax_psd_nom.set_title(r'b) Posterior median', loc='left')
    ax_psd_nom.set_xlabel('')
    ax_psd_nom.set_ylabel('')
    
    # Shared colorbar for PSD
    sm = mpl.cm.ScalarMappable(norm=norm_psd, cmap=cmap_psd)
    cbar = fig.colorbar(sm, cax=ax_psd_cbar)
    cbar.set_label(r'$\mathrm{d}N/\mathrm{d}\log d_m$  (cm$^{-3}$)')
    _my_format_colorbar(cbar, vmin_psd, vmax_psd)
    
    
    plot_timeseries_2d(ax_w_marg, day_marg.datetimes, d_m, W_marg, show_cbar=False,
                    cmap='inferno_r', vmin=vmin_w, vmax=vmax_w,
                    color_scale='log_3zone'
                    )
    ax_w_marg.set_title(fr'c) Uncertainty (relative width of {CI_coverage} % credible interval)',
                        loc='left')
    ax_w_marg.set_xlabel('')
    plot_timeseries_2d(ax_w_nom, day_nom.datetimes, d_m, W_nom, show_cbar=False,
                    cmap='inferno_r', vmin=vmin_w, vmax=vmax_w,
                    color_scale='log_3zone'
                    )
    ax_w_nom.set_title(fr'd) Uncertainty (relative width of {CI_coverage} % credible interval)',
                       loc='left')
    ax_w_nom.set_xlabel('')
    ax_w_nom.set_ylabel('')
    
    # Shared colorbar for W
    sm = mpl.cm.ScalarMappable(norm=norm_w, cmap=cmap_w)
    cbar = fig.colorbar(sm, cax=ax_w_cbar, extend='max')
    cbar.set_label(r'Relative HDI width')
    _my_format_colorbar(cbar, vmin_w, vmax_w)
    cbar.ax.yaxis.set_major_formatter(tck.FuncFormatter(lambda x, pos: f"{x*100:.0f} %"))
    
    
    ax_nt_marg.fill_between(day_marg.datetimes,
                    Nt_low_marg,
                    Nt_high_marg,
                    alpha=0.25,
                    facecolor='C0',
                    label=f'{CI_coverage} % credible interval'
                    )
    ax_nt_marg.plot(day_marg.datetimes, Nt_marg, 'C0', linewidth=1,
                label=r'Median $N_\mathrm{tot}$')
    
    ax_nt_nom.fill_between(day_nom.datetimes,
                    Nt_low_nom,
                    Nt_high_nom,
                    alpha=0.25,
                    facecolor='C1',
                    label=f'{CI_coverage} % credible interval'
                    )
    ax_nt_nom.plot(day_nom.datetimes, Nt_nom, 'C1', linewidth=1,
                   label=r'Median $N_\mathrm{tot}$')
    
    ax_nt_marg.grid(axis='y')
    ax_nt_nom.grid(axis='y')
    
    # Set ylims
    ymin1, ymax1 = ax_nt_marg.get_ylim()
    ymin2, ymax2 = ax_nt_nom.get_ylim()
    ymax = max(ymax1, ymax2)
    ax_nt_marg.set_ylim(0, ymax)
    ax_nt_nom.set_ylim(0, ymax)
    
    tmin = day_marg.datetimes[0]
    tmax = day_marg.datetimes[-1] + np.timedelta64(6, 'm')
    ax_nt_marg.set_xlim(tmin, tmax)
    ax_nt_nom.set_xlim(tmin, tmax)
    
    ax_nt_marg.set_title(r'e) Total particle number',
                       loc='left')
    ax_nt_nom.set_title(r'f) Total particle number',
                       loc='left')
    
    ax_nt_marg.set_ylabel(r'$N_\mathrm{tot}$')
    ax_nt_marg.legend()
    ax_nt_nom.legend()
    
    ax_nt_cbar.set_visible(False)
    
    # Rotate time ticks
    for ax in [ax_psd_marg, ax_psd_nom, ax_w_marg, ax_w_nom, ax_nt_marg, ax_nt_nom]:
        plt.setp(ax.get_xticklabels(), rotation=15, ha='right')
    
    fig.subplots_adjust(top=0.93, wspace=0.15, hspace=0.45, right=0.95, left=0.06)
    
    # Adjust position of colorbars
    shrink = 0.02
    left, bottom, width, height = ax_psd_cbar.get_position().bounds
    ax_psd_cbar.set_position([left - shrink, bottom, width, height])
    left, bottom, width, height = ax_w_cbar.get_position().bounds
    ax_w_cbar.set_position([left - shrink, bottom, width, height])
    
    fig.canvas.draw()   # needed so positions exist
    
    bbox_col1 = ax_psd_marg.get_position()
    bbox_col2 = ax_psd_nom.get_position()
    
    x1 = bbox_col1.x0 + 0.5 * (bbox_col1.x1 - bbox_col1.x0)
    x2 = bbox_col2.x0 + 0.5 * (bbox_col2.x1 - bbox_col2.x0)
    
    fig.text(x1, 0.98, "Marginalized inversion", ha='center', fontsize=14)
    fig.text(x2, 0.98, "Non-marginalized inversion", ha='center', fontsize=14)
    
    plt.pause(0.1)
    
    if SAVE_FIGURES:
        if i == 0:
            plt.savefig(f'{FIGURES_DIR}/day_10_comparison.png', dpi=DPI)
        elif i == 1:
            plt.savefig(f'{FIGURES_DIR}/day_25_comparison.png', dpi=DPI)
    
    # # PSD CI ratios
    # CI_width_marg =  (CI_up_marg - CI_lo_marg) / medians_marg
    # CI_width_nom =  (CI_up_nom - CI_lo_nom) / medians_nom
    # CI_ratio = CI_width_marg / CI_width_nom
    # plt.figure(), plt.semilogx(medians_marg / binwidth, CI_ratio, '.')
    # plt.grid()
    # plt.title('CI ratio vs. median')
    # plt.figure(), plt.hist(CI_ratio, 15)
    
    # # Ntot CI ratios
    # N_CI_width_marg =  Nt_high_marg - Nt_low_marg
    # N_CI_width_nom =  Nt_high_nom - Nt_low_nom
    # N_CI_ratio = N_CI_width_marg / N_CI_width_nom
    # plt.figure(), plt.plot(Nt_marg, N_CI_ratio, '.')
    # plt.figure(), plt.hist(N_CI_ratio, 15)
    # r = np.corrcoef(Nt_marg, N_CI_ratio)[0, 1]
    # print(f'Correlation between Ntot and CI_ratio: {r : .3f}')


#%% Check a single scan

single_marg = psd_posteriors_marg.between_times('2024-12-10T18:00:00', '2024-12-10T18:06:00')
single_nom = psd_posteriors_nonmarg.between_times('2024-12-10T18:00:00', '2024-12-10T18:06:00')

fig, axs = plt.subplots(2, 2, num=19, clear=True,
                 figsize=(FIG_WIDTH_DOUBLE, 1.5 * FIG_HEIGHT))
axs = axs.flatten()

# Convert numpy datetimes to Python datetimes for easier formatting
datetime_1 = single_marg.datetimes[0].astype('datetime64[s]').item()
datetime_2 = single_nom.datetimes[0].astype('datetime64[s]').item()

plot_posterior_summary(axs[0], single_marg[0], CI_coverage)
axs[0].set_yscale('linear')
axs[0].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
axs[0].grid('on')
axs[0].set_title(
    f'PSD on {datetime_1.date()} at {datetime_1.time()}',
    loc='center'
    )

plot_posterior_summary(axs[2], single_nom[0], CI_coverage, color='C1')
axs[2].set_yscale('linear')
axs[2].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
axs[2].grid('on')
# axs[2].set_title(
#     f'PSD on {datetime_2.date()} at {datetime_2.time()}',
#     loc='center'
#     )

# Set same ylims
axx = (axs[0], axs[2])
ymaxx = max(ax.get_ylim()[1] for ax in axx)
yminn = min(ax.get_ylim()[0] for ax in axx)
for ax in axx:
    ax.set_ylim(yminn, ymaxx)


Ntot_samples_marg = single_marg[0].propagate_to(total_concentration)
Ntot_samples_nom = single_nom[0].propagate_to(total_concentration)
Ntot_min = min(np.percentile(Ntot_samples_marg, 0.1), np.percentile(Ntot_samples_nom, 0.1))
Ntot_max = max(np.percentile(Ntot_samples_marg, 99.9), np.percentile(Ntot_samples_nom, 99.9))
plot_Ntot_histogram(axs[1], Ntot_samples_marg, xlimits=(Ntot_min, Ntot_max))
plot_Ntot_histogram(axs[3], Ntot_samples_nom, xlimits=(Ntot_min, Ntot_max), color='C1')

for ax in (axs[0], axs[1]):
    ax.tick_params(labelbottom=False)
    ax.set_xlabel('')
for ax in (axs[2], axs[3]):
    ax.set_title('')
for ax in (axs[0], axs[2]):
    ax.set_xticks([d_m[0] * 1e9, 20, 50, 100, 250, 500, d_m[-1] * 1e9])

axs[0].annotate(r'Marginalized',
                xy=(700, 10100),
            ha='right',
            va='bottom',
            fontsize=12,
            bbox=dict(
                    facecolor='white',
                    edgecolor='C0',
                    linewidth=2,
                    alpha=0.7,
                    pad=3,
                    )
            )
axs[2].annotate(r'Non-marginalized',
                xy=(700, 10100),
            ha='right',
            va='bottom',
            fontsize=12,
            bbox=dict(
                    facecolor='white',
                    edgecolor='C1',
                    linewidth=2,
                    alpha=0.7,
                    pad=3,
                    )
            )

fig.tight_layout()
fig.subplots_adjust(hspace=0.05)
plt.show()

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/day_10_singlescan_comparison.pdf', dpi=DPI)


#%% Calculate some derived quantities with uncertainties

psd_timerange = psd_posteriors_marg.between_times('2024-12-08', '2024-12-15')


# concentration_in_range, geometric_mean_diameter, mode_diameter,
# median_diameter, surface_area_concentration, volume_concentration,
# condensation_sink, effective_diameter, geometric_std

fig, axs = plt.subplots(nrows=5, ncols=1, num=50, clear=True,
                        figsize=(FIG_WIDTH_DOUBLE, 2.0 * FIG_HEIGHT))

d_m = psd_timerange[0].d_m

Ntot_samples = psd_timerange.propagate_to(total_concentration)
nucl_samples = psd_timerange.propagate_to(concentration_in_range, d_m, 1e-9, 25e-9)
ait_samples = psd_timerange.propagate_to(concentration_in_range, d_m, 25e-9, 100e-9)
acc_samples = psd_timerange.propagate_to(concentration_in_range, d_m, 100e-9, 1000e-9)
CS_samples = psd_timerange.propagate_to(condensation_sink, d_m)


CI_coverage = 95
medians_tr, _, _ = psd_timerange.summary(coverage=CI_coverage, n_jobs=10)
binwidth = np.log10(d_m[1]) - np.log10(d_m[0])
Z_timerange = medians_tr.T / binwidth

plot_timeseries_2d(axs[0], psd_timerange.datetimes, d_m, Z_timerange,
                   show_cbar=False
                # cbar_label=r'$\mathrm{d}N / \mathrm{d}\log d_m$ $(\mathrm{cm}^{-3})$',
                )
axs[0].set_ylabel(r'$d_\mathrm{m}$')
axs[0].set_title(r'Posterior median', loc='left')
plot_timeseries_1d(axs[1], psd_timerange.datetimes, nucl_samples, coverage=CI_coverage,
                   title='Nucleation mode (1-25 nm)'
                   )
axs[1].set_ylabel(r'$N_\mathrm{tot}$')
plot_timeseries_1d(axs[2], psd_timerange.datetimes, ait_samples, coverage=CI_coverage,
                   title='Aitken mode (25-100 nm)')
axs[2].set_ylabel(r'$N_\mathrm{tot}$')
plot_timeseries_1d(axs[3], psd_timerange.datetimes, acc_samples, coverage=CI_coverage,
                   title='Accumulation mode (100-1000 nm)')
axs[3].set_ylabel(r'$N_\mathrm{tot}$')
plot_timeseries_1d(axs[4], psd_timerange.datetimes, CS_samples, coverage=CI_coverage,
                   title='Condensation sink')
axs[4].set_ylabel(r'$s^{-1}$')

for i, ax in enumerate(axs):
    ax.set_xlabel('')
    if i != 4:
        ax.tick_params(labelbottom=False)
    if i != 1:
        ax.legend().set_visible(False)
fig.tight_layout()
fig.subplots_adjust(hspace=0.27)


# ETC

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/derived_quantities_Ntot.png', dpi=DPI)



# %%  Compare the Laplace approximation and true posterior from MCMC

fig, axs = plt.subplots(ncols=2, nrows=2, num=199, clear=True)
fig.set_figwidth(FIG_WIDTH_DOUBLE)
fig.set_figheight(2 * FIG_HEIGHT)
compare_posterior_representations(axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1])
fig.tight_layout()

if SAVE_FIGURES:
    plt.savefig(f'{FIGURES_DIR}/compare_Laplace_MCMC.pdf', dpi=DPI)