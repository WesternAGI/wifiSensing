

import os
import pandas as pd
import numpy as np
import math

N_of_samples = 200

def get_csi(directory, cls, verbose=True) :

    cls_df = pd.DataFrame()
    for file in os.listdir(directory):
        if file.endswith(".csv") and cls in file:
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, on_bad_lines='warn', engine='python')

            csi_rows = [] 
            for one_row in df['CSI_DATA'].iloc[40:-40]: # Ignore first few and last few seconds data
                try :
                    one_row = one_row.strip("[]")
                    csi_row = [int(x) for x in one_row.split(" ") if x != '']
                    csi_rows.append(csi_row)
                except :
                    if verbose : print(f"Error in file: {file}, row: {one_row}")
                    continue

            df = pd.DataFrame(csi_rows)
            cls_df = pd.concat([cls_df, df], axis=0)
    
    return cls_df 
    


def csi_to_amplitude_phase(df):
    total_amps, total_phases = [], []
    for i, value in enumerate(df.values):
        imaginary, real, amplitudes, phases = [], [], [], []
        csi_one_row_lst = value.tolist()
        [imaginary.append(csi_one_row_lst[item]) if item%2==0 else real.append(csi_one_row_lst[item]) for item in range(len(csi_one_row_lst))]
        val = int(len(csi_one_row_lst)//2)
        
        for k in range(val):
            amplitudes.append(round(math.sqrt(float(imaginary[k])** 2 + float(real[k])** 2),4))
            phases.append(round(math.atan2(float(imaginary[k]), float(real[k])),4))
        total_amps.append(np.array(amplitudes))
        total_phases.append(np.array(phases))
    
    total_amps_df = pd.DataFrame(total_amps)
    total_phases_df = pd.DataFrame(total_phases)
    
    return total_amps_df, total_phases_df


def filter_df(df):
    ## Here, based on sig_mode, 802.11a/g/n received. Here we receive both 802.11a/g and 802.11n
    ## So, either 52 or 56 total sub-carrier would be useful. The first 4 and the last 4 are rejected as null guard.
    df1 = df.iloc[:,5:32]  # 6:32 for 802.11ag 4:32 for 802.11n
    df2 = df.iloc[:,33:60] # 33:59 for 802.11ag 33:61 for 802.11n
    df = pd.concat([df1, df2], axis=1)
    return df


def select_data_portion(df,select_size):
    selected_df_list = []
    for item in range(0,len(df)-select_size, select_size):
        selected_df = df.iloc[item:item+select_size].to_numpy().flatten()
        selected_df_list.append(selected_df)
    selected_df = pd.DataFrame(selected_df_list)
    return selected_df


# Moving average of the data
def moving_average(df, window_size):
    """"
    Compute the moving average with a window of size specified
    """

    rolling_mean = df.rolling(window=window_size).mean()
    downsampled = rolling_mean.iloc[window_size::window_size, :]
    return downsampled

