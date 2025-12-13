# -*- coding: utf-8 -*-

import importlib.resources as resources

from MPSS_UQ.chargingmodels import LYFInterpolatorPrecomputation


''' A script to create the LYF model interpolators. '''

savefolder = resources.files('MPSS_UQ.data')

LYF_interp = LYFInterpolatorPrecomputation()
LYF_interp.compute()
LYF_interp.save_precomputed_data(savefolder / 'LYF_interpolator_data')
