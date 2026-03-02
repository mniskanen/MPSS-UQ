# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as resources

from tqdm import tqdm

from MPSS_UQ.chargingmodels import LYFFluxInterpolator, LYFInterpolator
from MPSS_UQ.chargingmodels import LYFChargingModel, WiedensohlerChargingModel


def evaluate_charging_probability_range(fig, axs, vary_parameter):
    ''' Run the LYF model with different inputs to plot the range over which the charging
    probabilities vary.
    
    vary_parameters : 'ion-mobility', 'ion-ratio', 'ion-mobility-ratio'
    
    '''
    
    # Initialize models common for each case here
    fname = resources.files('MPSS_UQ.data') / 'LYF_interpolator_data.npz'
    flux_interpolator = LYFFluxInterpolator(fname)
    d_m = np.geomspace(1e-9, 2.5e-6, 200)
    # d_m = np.geomspace(1e-9, 1e-6, 300)
    # d_m = np.geomspace(1e-9, 2.5e-6, 30)
    charges_output = np.arange(-3, 3 + 1)
    
    # if vary_parameter == 'ion-mobility':
    charger_interpolator = LYFInterpolator(d_m / 2, charges_output, fname)
    
    # else:
    #     charger_interpolator = LYFChargingModel(d_m / 2, charges_output,
    #                                             flux_interpolator=flux_interpolator
    #                                             )

    # Compute the Wiedensohler approximation
    charger_wiedensohler = WiedensohlerChargingModel(d_m / 2, charges_output)
    wiedensohler_approximation = charger_wiedensohler.charging_probability()
    
    if vary_parameter == 'ion-mobility':
        
        ion_ratio = 1.0
        
        n_samples = 5000
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
                                                                                  # ion_ratio,
                                                                                  )
                
                i += 1
                pbar.update(1)
        
        # Paper plot
        plot_ranges(fig, axs, charges_output, charging_fractions, d_m,
                    n_samples, wiedensohler_approximation)
    
    elif vary_parameter == 'ion-ratio':
        
        pos_ion_mobility = 1.35e-4
        neg_ion_mobility = 1.60e-4
        
        n_samples = 2000
        charging_fractions = np.zeros((n_samples, charges_output.shape[0], d_m.shape[0]))
        
        for i in tqdm(range(n_samples), desc='Varying ion ratio'):
            # ion_ratio = np.random.normal(loc=1.0, scale=0.2/3)
            ion_ratio = np.random.uniform(low=1.0, high=1.2)
            
            charging_fractions[i] = charger_interpolator.charging_probability(pos_ion_mobility,
                                                                              neg_ion_mobility,
                                                                              ion_ratio,
                                                                              )
        
        plot_ranges(fig, axs, charges_output, charging_fractions, d_m,
                    n_samples, wiedensohler_approximation)
    
    elif vary_parameter == 'ion-mobility-ratio':
        
        n_samples_mobility = 1000
        n_gridpoints = int(np.sqrt(n_samples_mobility))
        pos_ion_mobilities = np.linspace(1.05e-4, 1.70e-4, n_gridpoints)
        neg_ion_mobilities = np.linspace(1.10e-4, 2.10e-4, n_gridpoints)
        
        PP, NN = np.meshgrid(pos_ion_mobilities, neg_ion_mobilities)
        n_samples_mobility = np.sum(NN >= PP)
        
        n_samples_ratio = 10
        n_samples = n_samples_mobility * n_samples_ratio
        charging_fractions = np.zeros((n_samples, charges_output.shape[0], d_m.shape[0]))
        
        pbar = tqdm(position=0, desc='Varying ion mobilities and ratios')
        pbar.reset(total = n_samples)
        i = 0
        for _, pos_ion_mobility in enumerate(pos_ion_mobilities):
            for _, neg_ion_mobility in enumerate(neg_ion_mobilities):
                if pos_ion_mobility > neg_ion_mobility:
                    continue
                
                for _ in range(n_samples_ratio):
                    # ion_ratio = np.random.normal(loc=1.0, scale=0.2/2)
                    ion_ratio = np.random.uniform(low=1.0, high=1.2)
                    
                    charging_fractions[i] = charger_interpolator.charging_probability(
                        pos_ion_mobility, neg_ion_mobility, ion_ratio
                        )
                    
                    i += 1
                    pbar.update(1)
        
        plot_ranges(fig, axs, charges_output, charging_fractions, d_m,
                    n_samples, wiedensohler_approximation)
    
    # return charging_fractions
    # breakpoint()
    # plt.figure(num=77, clear=True)
    # for i in range(5, 12):
    #     fmin = charging_fractions[:, i, :].min(axis=0)
    #     fmax = charging_fractions[:, i, :].max(axis=0)
    #     rel_unc = (fmax - fmin) / ((fmax + fmin) * 0.5) * 100
    #     plt.semilogx(d_m, rel_unc)
    
    calculate_rel_diff_wiedensohler(d_m, charges_output, charging_fractions,
                                    wiedensohler_approximation)
    

def calculate_rel_diff_wiedensohler(d_m, charges_output, charging_fractions,
                                    wiedensohler_approximation):
    
    n_charges = charges_output.shape[0]
    rel_errors = np.nan * np.ones((n_charges, charging_fractions.shape[0], d_m.shape[0]))
    mean_abs_rel_error = np.zeros((n_charges))
    p99_abs_rel_error = np.zeros_like(mean_abs_rel_error)
    
    for i, charge in enumerate(charges_output):
        # Compare to the Wiedensohler approximation only up to 1e-6 (it's not valid above that)
        upper = np.where(d_m >= 1e-6)[0][0]  # First d_m value > 1e-6 m
        # For charges -1, 0, 1 the model is valid for >= 1 nm
        # For charges -2, 2 the model is valid for >= 20 nm
        # For charges >= +-3 the (Gunn) model is valid for >= 50 nm
        if np.abs(charge) <= 1:
            lower = np.where(d_m >= 1e-9)[0][0]
        elif np.abs(charge) == 2:
            lower = np.where(d_m >= 20e-9)[0][0]
        else:
            lower = np.where(d_m >= 50e-9)[0][0]
        
        rel_errors[i, :, lower:upper] = (wiedensohler_approximation[i, lower:upper]
                         - charging_fractions[:, i, lower:upper]) \
            / (np.abs(wiedensohler_approximation[i, lower:upper]) + 1e-9) * 100
        mean_abs_rel_error[i] = np.mean(np.abs(rel_errors[i, :, lower:upper]))
        p99_abs_rel_error[i] = np.quantile(np.abs(rel_errors[i, :, lower:upper]), 0.99)
        mean_abs_rel_error
        p99_abs_rel_error
    # plt.figure(), plt.semilogx(d_m, rel_errors.max(axis=1)[5:12].T)
    # plt.figure(), plt.semilogx(d_m, rel_errors[5].T, 'k', alpha=0.01), plt.grid()
    # breakpoint()


def plot_ranges(fig, axs, charges_output, charging_fractions, d_m, n_samples,
                wiedensohler_approximation):
    
    axs = axs.flatten()    
    
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
            facecolor = 'C0'
        elif charge == 0:
            facecolor = 'C2'
        elif charge > 0:
            facecolor = 'C3'
        
        pr_plot = np.c_[np.min(charging_fractions[:, charge_idx, :], axis=0),
                        np.max(charging_fractions[:, charge_idx, :], axis=0)].T
        
        axs[plt_ax].fill_between(d_m * 1e9,
                                 pr_plot[1],
                                 pr_plot[0],
                                 alpha=0.20,
                                 facecolor=facecolor,
                                 label='95 % credible interval'
                                 )
        rng = np.random.default_rng(seed=1)
        indexes = rng.choice(n_samples, 50)
        axs[plt_ax].semilogx(d_m * 1e9,
                             charging_fractions[indexes, charge_idx, :].T,
                             color=facecolor,
                             linewidth=0.5,
                             alpha=0.5
                             )
        
        # Plot the Wiedensohler approximation only up to 1e-6 (it's not valid above that)
        upper = np.where(d_m >= 1e-6)[0][0]  # First d_m value > 1e-6 m
        # For charges -1, 0, 1 the model is valid for >= 1 nm
        # For charges -2, 2 the model is valid for >= 20 nm
        # For charges >= +-3 the (Gunn) model is valid for >= 50 nm
        if np.abs(charge) <= 1:
            lower = np.where(d_m >= 1e-9)[0][0]
        elif np.abs(charge) == 2:
            lower = np.where(d_m >= 20e-9)[0][0]
        else:
            lower = np.where(d_m >= 50e-9)[0][0]
        axs[plt_ax].plot(d_m[lower:upper] * 1e9,
                         wiedensohler_approximation[charge_idx, lower:upper],
                         'k--', label='Wiedensohler'
                         )
        
        axs[plt_ax].set_xscale('log')
        # axs[plt_ax].set_yscale('log')
        # axs[plt_ax].set_xlim([0.85 * d_m[0] * 1e9, d_m[-1] * 1e9 / 0.85])
        if charge > 0:
            title = f'Charge = +{charge}'
        else:
            title = f'Charge = {charge}'
        text_v = 0.95
        if plt_ax == 0:
            text_v = 0.75
        axs[plt_ax].text(0.04, text_v, title, transform=axs[plt_ax].transAxes,
                fontsize=12,
                va='top',
                ha='left',
                bbox=dict(
                        boxstyle='round,pad=0.3',   # shape and padding
                        facecolor='white',          # light background
                        edgecolor='lightgray',      # light border
                        linewidth=1,
                        # alpha=0.8                   # transparency
                    )
                )

        if plt_ax == 6 or plt_ax == 7:
            axs[plt_ax].set_xlabel('Particle mobility diameter (nm)')
        else:
            axs[plt_ax].tick_params(labelbottom=False)  # hides x-axis labels
            axs[plt_ax].set_xlabel("")  # ensures no xlabel is set
        
        if plt_ax in [0, 2, 4, 6]:
            axs[plt_ax].set_ylabel('Probability')
        else:
            axs[plt_ax].tick_params(labelleft=False)
        
        axs[plt_ax].grid('on', which='major')
        
        ymin, ymax = axs[plt_ax].get_ylim()
        axs[plt_ax].set_ylim(0, ymax)
        axs[plt_ax].set_xlim([d_m[0] * 1e9, d_m[-1] * 1e9])
    
    # Set the same ylimits for +- same charge
    for idx in [2, 4, 6]:
        y0, y1 = axs[idx].get_ylim()
        _, y2 = axs[idx+1].get_ylim()
        y_max = max(y1, y2)
        axs[idx].set_ylim([y0, y_max])
        axs[idx+1].set_ylim([y0, y_max])
    
    # Make the legend in axs[1] which is otherwise invisible
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='C2',
                        label=r'Range of neutral ion $f_p$ (LYF)')
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='C0',
                        label=r'Range of negative ion $f_p$ (LYF)')
    axs[1].fill_between(d_m[0:2], pr_plot[1, 0:2], pr_plot[0, 0:2],
                        alpha=0.5, facecolor='C3',
                        label=r'Range of positive ion $f_p$ (LYF)')
    axs[1].plot(1, 1, 'k--', label='Wiedensohler approximation')
    axs[1].axis('off')
    axs[1].legend(loc='center')
    
    plt.pause(0.1)
