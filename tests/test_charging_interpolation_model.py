# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import importlib.resources as resources

from time import perf_counter
from tqdm import tqdm

from MPSS_UQ.chargingmodels import LYFChargingModel, LYFInterpolator, LYFFluxInterpolator


def test_interpolation_accuracy():
    
    n_bins = 35
    
    d_m = np.geomspace(1e-9, 2.5e-6, n_bins)
    charges_output = np.arange(-8, 8 + 1)
    n_charges = charges_output.shape[0]
    
    # Initialize the different charger models
    fname = resources.files('MPSS_UQ.data') / 'LYF_interpolator_data.npz'
    flux_interpolator = LYFFluxInterpolator(fname)
    charger_flux_interpolator = LYFChargingModel(d_m / 2,
                                                 charges_output,
                                                 flux_interpolator=flux_interpolator
                                                 )
    charger_interpolator = LYFInterpolator(d_m / 2,
                                           charges_output,
                                           fname,
                                           )
    charger_direct = LYFChargingModel(d_m / 2,
                                      charges_output,
                                      max_modelled_charge=25
                                      )
    
    n_tests = 10
    time_direct = 0
    time_interp_1 = 0
    time_interp_2 = 0
    rel_errors_1 = np.zeros(n_tests * charges_output.shape[0] * d_m.shape[0])
    rel_errors_2 = np.zeros_like(rel_errors_1)
    for test in tqdm(range(n_tests)):
        
        # Choose test parameters
        while True:
            pos_ion_mobility = np.random.uniform(low=1.05e-4, high=1.70e-4)
            neg_ion_mobility = np.random.uniform(low=1.10e-4, high=2.10e-4)
            ion_ratio = 1#np.random.normal(loc=1.0, scale=0.2/2)
            
            if pos_ion_mobility < neg_ion_mobility:
                break
        
        t1 = perf_counter()
        
        # Run the flux interpolator
        cp_interp_1 = charger_flux_interpolator.charging_probability(pos_ion_mobility,
                                                                     neg_ion_mobility,
                                                                     ion_ratio,
                                                                     )
        t2 = perf_counter()
        
        # Run the charging probability interpolator
        cp_interp_2 = charger_interpolator.charging_probability(pos_ion_mobility,
                                                                neg_ion_mobility,
                                                                )
        t3 = perf_counter()        
        
        # Run the direct model
        cp_direct = charger_direct.charging_probability(pos_ion_mobility,
                                                        neg_ion_mobility,
                                                        ion_ratio,
                                                        )
        t4 = perf_counter()
        
        time_interp_1 += t2 - t1
        time_interp_2 += t3 - t2
        time_direct += t4 - t3
        
        # Plots to compare charged fractions
        plt.figure(num=3), plt.clf()
        plt.title('Steady-state charge distribution, flux interpolator')
        plt.xlabel('Particle diameter (m)')
        for idx, k in enumerate(charges_output):
            if k < 0:
                label = 'neg. charges, direct' if k == -1 else None
                plt.loglog(d_m, cp_direct[idx], 'r-', label=label)
                label = 'neg. charges, interp' if k == -1 else None
                plt.loglog(d_m, cp_interp_1[idx], 'k--', label=label)
            elif k == 0:
                plt.loglog(d_m, cp_direct[idx], 'm-', label='0, direct')
                plt.loglog(d_m, cp_interp_1[idx], 'k--', label='0, interp')
            elif k > 0:
                label = 'pos. charges, direct' if k == 1 else None
                plt.loglog(d_m, cp_direct[idx], 'b-', label=label)
                label = 'pos. charges, interp' if k == 1 else None
                plt.loglog(d_m, cp_interp_1[idx], 'k--', label=label)

        plt.legend()
        plt.pause(0.1)
        
        plt.figure(num=4), plt.clf()
        plt.title('Steady-state charge distribution, charge prob. interpolator')
        plt.xlabel('Particle diameter (m)')
        for idx, k in enumerate(charges_output):
            if k < 0:
                label = 'neg. charges, direct' if k == -1 else None
                plt.loglog(d_m, cp_direct[idx], 'r-', label=label)
                label = 'neg. charges, interp' if k == -1 else None
                plt.loglog(d_m, cp_interp_2[idx], 'k--', label=label)
            elif k == 0:
                plt.loglog(d_m, cp_direct[idx], 'm-', label='0, direct')
                plt.loglog(d_m, cp_interp_2[idx], 'k--', label='0, interp')
            elif k > 0:
                label = 'pos. charges, direct' if k == 1 else None
                plt.loglog(d_m, cp_direct[idx], 'b-', label=label)
                label = 'pos. charges, interp' if k == 1 else None
                plt.loglog(d_m, cp_interp_2[idx], 'k--', label=label)

        plt.legend()
        plt.pause(0.1)
        
        # Calculate the relative errors
        eps = 1e-9
        for i in range(n_charges):
            denom = np.abs(cp_direct[i]) + eps
            offset = (test * n_charges + i) * n_bins

            rel_errors_1[offset : offset + n_bins] = np.abs(cp_direct[i] - cp_interp_1[i]) / denom
            rel_errors_2[offset : offset + n_bins] = np.abs(cp_direct[i] - cp_interp_2[i]) / denom
    
    p99_rel_error_1 = np.quantile(rel_errors_1, 0.99)   # 99th percentile
    p99_rel_error_2 = np.quantile(rel_errors_2, 0.99)
    avg_rel_error_1 = np.mean(rel_errors_1)
    avg_rel_error_2 = np.mean(rel_errors_2)
    print(f'\nAverage relative error (flux interp): {avg_rel_error_1 * 100 : .2g} %')
    print(f'Average relative error (cp interp): {avg_rel_error_2 * 100 : .2g} %')
    print(f'99th perc. relative error (flux interp): {p99_rel_error_1 * 100 : .2g} %')
    print(f'99th perc. relative error (cp interp): {p99_rel_error_2 * 100 : .2g} %\n')
    
    assert avg_rel_error_1 < 0.05, \
        'Average mean absolute error between direct and flux-interpolated solutions too large'
    assert avg_rel_error_2 < 0.05, \
        'Average mean absolute error between direct and cp-interpolated solutions too large'
    
    print(
        f'\nTimings for direct model: {time_direct : .2e} s, ' + 
        f'and flux interp model: {time_interp_1 : .2e} s, i.e., ' +
        f'and charge prob. interp model: {time_interp_2 : .2e} s, i.e., ' +
        f'the charge prob. interp model was {time_direct / time_interp_2 : .0f} times faster ' +
        'than the direct one.'
        )
    

if __name__ == '__main__':
    test_interpolation_accuracy()
