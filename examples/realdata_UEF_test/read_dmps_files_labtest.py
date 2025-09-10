# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 04:40:30 2025

@author: arttuy, Arttu Ylisirniö, University of Eastern Finland
Small modifications by Matti Niskanen, UEF
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime


def load_and_process_data(filestart,folder_path, start_date, end_date):
    
    def extract_date(filename):
        filedate = filename[-13:-5]

        return datetime.strptime(filedate,'%Y%m%d').date()
    
    all_files = [f for f in os.listdir(folder_path) if f.startswith(filestart) and  f.endswith(".scan")]
    
    filtered_files = [
        f for f in all_files
        if start_date <= str(extract_date(f)) <= end_date
        ]

    dataframes = []

    for file in filtered_files:
        
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path, sep='\t', header=1)
        
        #f1 = cpc flow status = aerosol flow status
        #f2 = liquid lvl status 0,1
        #f3 = sheat flow status
      
        df_out = pd.DataFrame()
        
        df_out[['start_time','end_time','p_sam', 't_sam', 'f1' ,'f2' ,'f3',
                't_sat','t_con', 'rhsh', 'qsam', 'qsmerr',
                'qsh', 'qsherr', 'nbn'
                ]] = df[['start_time','end_time','pressure(kpa)','temperature(c)','cpc_flow_status','cpc_butanol_status','sheat_flow_status',
                         'cpc_saturator_temp','cpc_condenser_temp','sheat flow_rh','aerosol_flow(lpm)','aerosol_flow_std',
                         'sheath_flow_(lpm)','sheat_flow_std','number_of_bins']].copy()
                         
        dp_columns = [f"dp{i}" for i in range(1, 31)]
        df_out[dp_columns] = df[dp_columns].copy()
        
        
        #rename dp to dmed_
        df_out.rename(columns={col: f'dmed_{col[2:]}' for col in df_out.columns if 'dp' in col}, inplace=True)
        
        conc_columns = [f"conc{i}" for i in range(1, 31)]
        df_out[conc_columns] = df[conc_columns].copy()
        
        #rename conc to conc_
        df_out.rename(columns={col: f'conc_{col[4:]}' for col in df_out.columns if col.startswith('conc')}, inplace=True)
        
        #At some point aerosol flow tolerance was too strict, so redoing f1
        df_out.loc[(df_out['qsam'] > 0.8) & (df_out['qsam'] < 1.25), ['f1']] = [1]
            
        df_out["numflag"] = np.where((df_out["f1"] == 1) & (df_out["f2"] == 1) & (df_out["f3"] == 1), 0, 0.999)
        
        df_out['start_time'] = pd.to_datetime(df_out['start_time']).dt.tz_convert('UTC') #   tz=pytz.timezone('Etc/GMT-2'))
        df_out['end_time'] = pd.to_datetime(df_out['end_time']).dt.tz_convert('UTC')

        #Convert C to K
        df_out[['t_sam','t_sat','t_con']] = df_out[['t_sam','t_sat','t_con']]+273.15        
        
        # Create derr columns for nasa-ames files
        new_cols = {
            col.replace('dmed_', 'derr_'): df_out[col] * 0.03
            for col in df_out.columns if col.startswith('dmed_')
        }
        
        df_new = pd.DataFrame(new_cols)
        df_out = pd.concat([df_out, df_new], axis=1)

        dataframes.append(df_out)

    # 
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.set_index('start_time', inplace = True)

        return combined_df
    else:
        return pd.DataFrame()
