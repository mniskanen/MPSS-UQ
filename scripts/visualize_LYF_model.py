# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as resources

from tqdm import tqdm

from MPSS_UQ.chargingmodels import LYFFluxInterpolator
from MPSS_UQ.chargingmodels import LYFChargingModel, WiedensohlerChargingModel


def evaluate_charging_probability_range(case, FIG_WIDTH, FIG_HEIGHT, num=None, zoom_num=None):
    ''' Run the LYF model with different inputs to plot the range over which the charging
    probabilities vary.
    
    cases : 'ion-mobility', 'ion-ratio', 'ion-mobility-ratio'
    
    '''
    
    # Initialize models common for each case here
    fname = resources.files('MPSS_UQ.data') / 'interpolator_flux_60dm_307'
    flux_interpolator = LYFFluxInterpolator(fname)
    d_m = np.geomspace(1e-9, 2.5e-6, 500)
    charges_output = np.arange(-8, 8 + 1)
    charger_interpolator = LYFChargingModel(d_m / 2, charges_output,
                                            flux_interpolator=flux_interpolator
                                            )

    # Compute the Wiedensohler approximation
    charger_wiedensohler = WiedensohlerChargingModel(d_m / 2, charges_output)
    wiedensohler_approximation = charger_wiedensohler.charging_probability()
    
    if case == 'ion-mobility':
        
        ion_ratio = 1.0
        
        n_samples = 2000
        n_gridpoints = int(np.sqrt(n_samples))
        pos_ion_mobilities = np.linspace(1.05e-4, 1.70e-4, n_gridpoints)
        neg_ion_mobilities = np.linspace(1.10e-4, 2.10e-4, n_gridpoints)
        
        PP, NN = np.meshgrid(pos_ion_mobilities, neg_ion_mobilities)
        n_samples = np.sum(NN >= PP)
        
        charging_fractions = np.zeros((n_samples, charges_output.shape[0], d_m.shape[0]))
        # pos_mobilities = np.zeros(n_samples)
        # neg_mobilities = np.zeros(n_samples)
        # pos_masses = np.zeros(n_samples)
        # neg_masses = np.zeros(n_samples)
        
        pbar = tqdm(position=0, desc='Varying ion mobilities')
        pbar.reset(total = n_samples)
        i = 0
        for _, pos_ion_mobility in enumerate(pos_ion_mobilities):
            for _, neg_ion_mobility in enumerate(neg_ion_mobilities):
                if pos_ion_mobility > neg_ion_mobility:
                    continue
                
                # # Store values for verification
                # pos_mobilities[i] = pos_ion_mobility
                # neg_mobilities[i] = neg_ion_mobility
                # pos_masses[i] = charger_interpolator.mobility_to_mass(pos_ion_mobility)
                # neg_masses[i] = charger_interpolator.mobility_to_mass(neg_ion_mobility)
                
                # Run the interpolation model
                charging_fractions[i] = charger_interpolator.charging_probability(pos_ion_mobility,
                                                                                  neg_ion_mobility,
                                                                                  ion_ratio,
                                                                                  )
                
                i += 1
                pbar.update(1)
        
        # Test
        if zoom_num is not None:
            plot_ranges_zoom(FIG_WIDTH, 0.8*FIG_HEIGHT, zoom_num, charges_output, charging_fractions,
                             d_m, n_samples, wiedensohler_approximation)
        
        # Paper plot
        return plot_ranges(FIG_WIDTH, FIG_HEIGHT, num, charges_output, charging_fractions, d_m,
                           n_samples, wiedensohler_approximation)
    
    elif case == 'ion-ratio':
        
        pos_ion_mobility = 1.35e-4
        neg_ion_mobility = 1.60e-4
        
        n_samples = 500
        charging_fractions = np.zeros((n_samples, charges_output.shape[0], d_m.shape[0]))
        
        for i in tqdm(range(n_samples), desc='Varying ion ratio'):
            # ion_ratio = np.random.normal(loc=1.0, scale=0.2/3)
            ion_ratio = np.random.uniform(low=1.0, high=1.2)
            charging_fractions[i] = charger_interpolator.charging_probability(pos_ion_mobility,
                                                                              neg_ion_mobility,
                                                                              ion_ratio,
                                                                              )
        
        return plot_ranges(FIG_WIDTH, FIG_HEIGHT, num, charges_output, charging_fractions, d_m,
                           n_samples, wiedensohler_approximation)
    
    elif case == 'ion-mobility-ratio':
        
        n_samples_mobility = 1000
        n_gridpoints = int(np.sqrt(n_samples_mobility))
        pos_ion_mobilities = np.linspace(1.05e-4, 1.70e-4, n_gridpoints)
        neg_ion_mobilities = np.linspace(1.05e-4, 2.10e-4, n_gridpoints)
        
        PP, NN = np.meshgrid(pos_ion_mobilities, neg_ion_mobilities)
        n_samples_mobility = np.sum(NN >= PP)
        
        n_samples_ratio = 10
        n_samples = n_samples_mobility * n_samples_ratio
        charging_fractions = np.zeros((n_samples, charges_output.shape[0], d_m.shape[0]))
        # breakpoint()
        pbar = tqdm(position=0, desc='Varying ion mobilities and ratios')
        pbar.reset(total = n_samples)
        i = 0
        for _, pos_ion_mobility in enumerate(pos_ion_mobilities):
            for _, neg_ion_mobility in enumerate(neg_ion_mobilities):
                if pos_ion_mobility > neg_ion_mobility:
                    continue
                
                for _ in range(n_samples_ratio):
                    ion_ratio = np.random.normal(loc=1.0, scale=0.2/2)
                    
                    charging_fractions[i] = charger_interpolator.charging_probability(
                        pos_ion_mobility, neg_ion_mobility, ion_ratio
                        )
                    
                    i += 1
                    pbar.update(1)
        
        return plot_ranges(FIG_WIDTH, FIG_HEIGHT, num, charges_output, charging_fractions, d_m,
                           n_samples, wiedensohler_approximation)


def plot_ranges(FIG_WIDTH, FIG_HEIGHT, fignum, charges_output, charging_fractions, d_m, n_samples,
                wiedensohler_approximation):
    
    fig, axs = plt.subplots(4, 2, num=fignum, clear=True)
    axs = axs.flatten()
    fig.set_figwidth(FIG_WIDTH)
    fig.set_figheight(4 * FIG_HEIGHT)
    for charge_idx, charge in enumerate(charges_output):
        
        # Brute force it...
        if charge == 0:
            plt_ax = 0
        elif charge == -1:
            plt_ax = 2
        elif charge == 1:
            plt_ax = 3
        elif charge == -2:
            plt_ax = 4
        elif charge == 2:
            plt_ax = 5
        elif charge == -3:
            plt_ax = 6
        elif charge == 3:
            plt_ax = 7
        else:
            continue
        
        if charge < 0:
            facecolor = 'b'
        elif charge == 0:
            facecolor = 'g'
        elif charge > 0:
            facecolor = 'r'
        
        pr_plot = np.c_[np.min(charging_fractions[:, charge_idx, :], axis=0),
                        np.max(charging_fractions[:, charge_idx, :], axis=0)].T
        
        axs[plt_ax].fill_between(d_m * 1e9,
                                 pr_plot[1],
                                 pr_plot[0],
                                 alpha=0.15,
                                 facecolor=facecolor,
                                 label='95 % credible interval'
                                 )
        n_jump = int(np.ceil(n_samples / 50))
        axs[plt_ax].semilogx(d_m * 1e9, charging_fractions[::n_jump, charge_idx, :].T,
                             color=facecolor,
                             linewidth=1,
                             alpha=0.15
                             )
        
        # Plot the Wiedensohler approximation only up to 1e-6 (it's not valid above that)
        upper = np.where(d_m > 1e-6)[0][0]  # First d_m value > 1e-6 m
        # For charges -1, 0, 1 the model is valid for >= 1 nm
        # For charges -2, 2 the model is valid for >= 20 nm
        # For charges >= +-3 the (Gunn) model is valid for >= 50 nm
        # TODO: make this check a part of the Wiedensohler charging model so it outputs NaN
        #       or zero or something like that for values outside the valid range
        if np.abs(charge) <= 1:
            lower = np.where(d_m > 1e-9)[0][0]
        elif np.abs(charge) == 2:
            lower = np.where(d_m > 20e-9)[0][0]
        else:
            lower = np.where(d_m > 50e-9)[0][0]
        axs[plt_ax].plot(d_m[lower:upper] * 1e9,
                         wiedensohler_approximation[charge_idx, lower:upper],
                         'k--', label='Wiedensohler'
                         )
        
        axs[plt_ax].set_xscale('log')
        axs[plt_ax].set_xlim([0.85 * d_m[0] * 1e9, d_m[-1] * 1e9 / 0.85])
        if charge > 0:
            axs[plt_ax].set_title(f'Charge = +{charge}')
        else:
            axs[plt_ax].set_title(f'Charge = {charge}')
        axs[plt_ax].set_xlabel('Particle mobility diameter (nm)')
        axs[plt_ax].set_ylabel('Charging probability')
        axs[plt_ax].grid('on', which='major')
    
    fig.tight_layout()
    
    # Make the legend in axs[1] which is otherwise invisible
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='g',
                        label=r'Range of neutral ion $C_p$ (LYF)')
                        # label=r'Range of charging prob. for charge = 0 (LYF)')
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='b',
                        label=r'Range of negative ion $C_p$ (LYF)')
                        # label=r'Range of charging prob. for charge = -1 (LYF)')
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='r',
                        label=r'Range of positive ion $C_p$ (LYF)')
                        # label=r'Range of charging prob. for charge = 1 (LYF)')
    axs[1].plot(1, 1, 'k--', label='Wiedensohler approximation')
    axs[1].axis('off')
    axs[1].legend(loc='upper right')
    
    # Move axs[0] towards the right
    pos = axs[0].get_position()
    pos.x0 += 0.10
    pos.x1 += 0.10
    axs[0].set_position(pos)
    
    plt.pause(0.1)
    
    return fig, axs

def plot_ranges_zoom(FIG_WIDTH, FIG_HEIGHT, fignum, charges_output, charging_fractions, d_m,
                     n_samples, wiedensohler_approximation):
    
    fig, axs = plt.subplots(3, 2, num=fignum, clear=True)
    axs = axs.flatten()
    fig.set_figwidth(FIG_WIDTH)
    fig.set_figheight(3 * FIG_HEIGHT)
    
    for charge_idx, charge in enumerate(charges_output):
        
        # Brute force it...
        if charge == 0:
            plt_ax = 0
        elif charge == -1:
            plt_ax = 2
        elif charge == 1:
            plt_ax = 3
        elif charge == -2:
            plt_ax = 4
        elif charge == 2:
            plt_ax = 5
        else:
            continue
        
        if charge < 0:
            facecolor = 'b'
        elif charge == 0:
            facecolor = 'g'
        elif charge > 0:
            facecolor = 'r'
        
        pr_plot = np.c_[np.min(charging_fractions[:, charge_idx, :], axis=0),
                        np.max(charging_fractions[:, charge_idx, :], axis=0)].T
        
        axs[plt_ax].fill_between(d_m * 1e9,
                                 pr_plot[1],
                                 pr_plot[0],
                                 alpha=0.15,
                                 facecolor=facecolor,
                                 label='95 % credible interval'
                                 )
        n_jump = int(np.ceil(n_samples / 50))
        axs[plt_ax].semilogx(d_m * 1e9, charging_fractions[::n_jump, charge_idx, :].T,
                             color=facecolor,
                             linewidth=1,
                             alpha=0.15
                             )
        
        # Plot the Wiedensohler approximation only up to 1e-6 (it's not valid above that)
        upper = np.where(d_m > 1e-6)[0][0]  # First d_m value > 1e-6 m
        # For charges -1, 0, 1 the model is valid for >= 1 nm
        # For charges -2, 2 the model is valid for >= 20 nm
        # For charges >= +-3 the (Gunn) model is valid for >= 50 nm
        # TODO: make this check a part of the Wiedensohler charging model so it outputs NaN
        #       or zero or something like that for values outside the valid range
        if np.abs(charge) <= 1:
            lower = np.where(d_m > 1e-9)[0][0]
        elif np.abs(charge) == 2:
            lower = np.where(d_m > 20e-9)[0][0]
        else:
            lower = np.where(d_m > 50e-9)[0][0]
        axs[plt_ax].plot(d_m[lower:upper] * 1e9,
                         wiedensohler_approximation[charge_idx, lower:upper],
                         'k--', label='Wiedensohler'
                         )
        
        axs[plt_ax].set_xscale('log')
        # axs[plt_ax].set_xlim([0.85 * d_m[0] * 1e9, d_m[-1] * 1e9 / 0.85])
        axs[plt_ax].set_xlim([0.85 * d_m[0] * 1e9, 40])
        # breakpoint()
        _, ymax = axs[plt_ax].get_ylim()
        ymin = np.min(pr_plot[0, :np.where(d_m > 40e-9)[0][0]])
        axs[plt_ax].set_ylim([np.max((1e-10, ymin)), ymax])
        axs[plt_ax].set_yscale('log')
        if charge > 0:
            axs[plt_ax].set_title(f'Charge = +{charge}')
        else:
            axs[plt_ax].set_title(f'Charge = {charge}')
        axs[plt_ax].set_xlabel('Particle mobility diameter (nm)')
        axs[plt_ax].set_ylabel('Charging probability')
        axs[plt_ax].grid('on', which='major')
    
    fig.tight_layout()
    
    # Make the legend in axs[1] which is otherwise invisible
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='g',
                        label=r'Range of neutral ion $C_p$ (LYF)')
                        # label=r'Range of charging prob. for charge = 0 (LYF)')
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='b',
                        label=r'Range of negative ion $C_p$ (LYF)')
                        # label=r'Range of charging prob. for charge = -1 (LYF)')
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='r',
                        label=r'Range of positive ion $C_p$ (LYF)')
                        # label=r'Range of charging prob. for charge = 1 (LYF)')
    axs[1].plot(1, 1, 'k--', label='Wiedensohler approximation')
    axs[1].axis('off')
    axs[1].legend(loc='upper right')
    
    # Move axs[0] towards the right
    pos = axs[0].get_position()
    pos.x0 += 0.10
    pos.x1 += 0.10
    axs[0].set_position(pos)
    
    plt.pause(0.1)
    
    return fig, axs