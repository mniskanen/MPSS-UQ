# -*- coding: utf-8 -*-

import importlib.resources as resources

from MPSS_UQ.chargingmodels import LYFInterpolator, LYFFluxInterpolator


''' A script to create the LYF model interpolators. '''

savefolder = resources.files('MPSS_UQ.data')

charging_model = LYFInterpolator()
charging_model.construct_interpolators()
charging_model.save_interpolators(savefolder / 'charging_prob_interpolator_data')

charging_model_flux = LYFFluxInterpolator()
charging_model_flux.construct_interpolators()
charging_model_flux.save_interpolators(savefolder / 'ion_flux_interpolator_data')