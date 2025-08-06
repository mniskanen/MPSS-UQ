# -*- coding: utf-8 -*-

import time
import numpy as np
import matplotlib.pyplot as plt

from MPSS_UQ.chargingmodels import (LYFChargingModel, WiedensohlerChargingModel,
                                          LYFFluxInterpolator)
from MPSS_UQ.definitions import INTERPOLATOR_DIR

particle_radius = np.logspace(np.log10(5e-10), np.log10(1.25e-6), num=32)

charges_output = np.array([-5, -2, -1, 0, 1, 2, 5])

W_cp = WiedensohlerChargingModel(particle_radius, charges_output).charging_probability()

flux_interpolator = LYFFluxInterpolator(f'{INTERPOLATOR_DIR}/interpolator_flux_60dm_307')
LYF_charging_model = LYFChargingModel(particle_radius, charges_output,
                                      flux_interpolator=flux_interpolator
                                      )
t1 = time.perf_counter()
LYF_cp = LYF_charging_model.charging_probability(1.2e-4,
                                                 1.35e-4,
                                                 1.0,
                                                 )

t2 = time.perf_counter()
total_time = t2 - t1
print(f'\nComputation took {total_time : .3f} seconds')

#%%
# Plot

flux_coeffs = LYF_charging_model.average_flux_coefficients

plt.figure(num=1, clear=True)
for idx, k in enumerate(LYF_charging_model.particle_charges):
    if k in charges_output:
        # Negative ions
        plt.subplot(2,2,1)
        plt.loglog(particle_radius, flux_coeffs[0, idx].T, label=k)
        plt.axis([2e-10, 1e-5, 1e-15, 1e-9])
        plt.title('Flux coefficients, negative ions')
        plt.xlabel('Particle radius (m)')
        plt.legend()
        
        # Positive ions
        plt.subplot(2,2,2)
        plt.loglog(particle_radius, flux_coeffs[1, idx].T, label=k)
        plt.axis([2e-10, 1e-5, 1e-15, 1e-9])
        plt.title('Flux coefficients, positive ions')
        plt.xlabel('Particle radius (m)')
        plt.legend()

# Charged fractions
plt.subplot(2,2,3)
plt.title('Steady-state charge distribution')
plt.xlabel('Particle radius (m)')
for idx, k in enumerate(charges_output):
    if k < 0:
        label = None
        if k == -1:
            label = 'neg. charges'
        plt.loglog(particle_radius, LYF_cp[idx], 'r-', label=label)
        plt.loglog(particle_radius, W_cp[idx], 'gd', markerfacecolor='none')
    elif k == 0:
        plt.loglog(particle_radius, LYF_cp[idx], 'k-', label='0')
        plt.loglog(particle_radius, W_cp[idx], 'gd', markerfacecolor='none',
                   label='Wiedensohler')
    elif k > 0:
        label = None
        if k == 1:
            label = 'pos. charges'
        plt.loglog(particle_radius, LYF_cp[idx], 'b-', label=label)
        plt.loglog(particle_radius, W_cp[idx], 'gd', markerfacecolor='none')

plt.axis([2e-10, 1e-5, 1e-4, 1.15])
# plt.axis([2e-10, 1e-5, 1e-16, 1.15])
plt.legend()