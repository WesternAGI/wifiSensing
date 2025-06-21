import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import math, os
import glob
import random
from sklearn.decomposition import PCA
import serial
import time
import threading
from sklearn.preprocessing import StandardScaler

import pickle
import numpy as np
import pandas as pd
from utils import csi_to_amplitude_phase, N_of_samples, filter_df, select_data_portion

# Load the trained model from the notebook
# Available models: human_identification_svc.pkl, pipe_final_knn.pkl, pipe_final_svm.pkl
# MODEL_PATH = "../model/pipe_final_knn.pkl"  # Use KNN model for binary classification
MODEL_PATH = "../model/human_identification_svc.pkl"  # Alternative: SVC model
DATAFILE_PATH = "tempfile/testData_doorclosed_myslef_working.csv"
try:
    loaded_pipe = pickle.load(open(MODEL_PATH, "rb"))
    print(f"Successfully loaded model from {MODEL_PATH}")
    print(f"Model pipeline steps: {loaded_pipe.named_steps.keys()}")
    print(f"Model classes: {loaded_pipe.classes_}")
except FileNotFoundError:
    print(f"Error: Model file {MODEL_PATH} not found!")
    print("Available models in model/:")
    for file in os.listdir("../model/"):
        if file.endswith(".pkl"):
            print(f"  - {file}")
    exit(1)

# Class labels mapping (based on notebook training data)
# From notebook: classes = ['empty', 'working'] with labels assigned as i (index)
class_labels = {
    0: "empty",
    1: "working"
}

column_names=["type","role","mac","rssi","rate","sig_mode","mcs","bandwidth","smoothing","not_sounding","aggregation","stbc","fec_coding","sgi","noise_floor","ampdu_cnt","channel","secondary_channel","local_timestamp","ant","sig_len","rx_state","real_time_set","real_timestamp","len","CSI_DATA"]

# Define the batch size (should match training: N_of_samples = 100)
batch_size = 500  # Increase batch size to ensure we have enough data after filtering
bandWidth = 1
verbose = True  # Add verbose flag for debugging

def testData():
    last_processed_row = 0
    prediction_count = 0
    
    print("Starting real-time CSI data processing and prediction...")
    print(f"N_of_samples size: {N_of_samples}")
    print(f"Batch size: {batch_size}")
    print(f"Expected input shape for model: {loaded_pipe.named_steps}")
    
    while True: 
        try:
            # Read the CSV file
            tempDF = pd.read_csv(DATAFILE_PATH)
            tempDF.columns = column_names
            tempDF.dropna(inplace=True)
            
            # Filter by bandwidth if needed
            df = tempDF   
            df.reset_index(inplace=True)
        
            # Check if there are new rows to process
            if len(df) > (last_processed_row + batch_size):
                print(f"\nProcessing new data batch #{prediction_count + 1}")
                
                # Get the new rows since the last processed row
                new_rows = df.iloc[last_processed_row:]
                csi_rows_raw = []

                # Get the most recent batch_size rows
                filtered_df = new_rows[(len(new_rows)-batch_size):]
                print(f"Length of filtered data: {len(filtered_df)}")
                
                # Parse CSI data from string format
                for one_row in filtered_df['CSI_DATA']:
                    try:
                        one_row = one_row.strip("[]")
                        csi_row_raw = [int(x) for x in one_row.split(" ") if x != '']
                        csi_rows_raw.append(csi_row_raw)
                    except Exception as e:
                        if verbose: 
                            print(f"Error during reading CSI row: {e}")
                        continue 

                if len(csi_rows_raw) == 0:
                    print("No valid CSI data found in batch")
                    time.sleep(5)
                    continue

                # Convert to DataFrame and process
                csi_df = pd.DataFrame(csi_rows_raw)
                print(f"Raw CSI data shape: {csi_df.shape}")
                csi_df.dropna(inplace=True)
                
                # Apply the same preprocessing pipeline as in training
                act_amp_df, act_phase_df = csi_to_amplitude_phase(csi_df)
                print(f"Amplitude data shape: {act_amp_df.shape}")
                
                # Filter subcarriers (remove guard bands)
                act_amp_df, act_phase_df = filter_df(act_amp_df), filter_df(act_phase_df)
                print(f"Filtered amplitude data shape: {act_amp_df.shape}")
                
                # Check if we have enough data for segmentation
                if len(act_amp_df) < N_of_samples:
                    print(f"Insufficient data for segmentation: need {N_of_samples}, got {len(act_amp_df)}")
                    print("Increasing batch size or waiting for more data...")
                    time.sleep(5)
                    continue
                
                # Select data portions (segment into fixed-size windows)
                X1, X2 = select_data_portion(act_amp_df, N_of_samples), select_data_portion(act_phase_df, N_of_samples)
                
                if len(X1) == 0 or len(X2) == 0:
                    print("No segments created after data portion selection")
                    print(f"act_amp_df shape: {act_amp_df.shape}, N_of_samples: {N_of_samples}")
                    time.sleep(5)
                    continue
                
                # Concatenate amplitude and phase features
                X = pd.concat([X1, X2], axis=1)
                X = X.fillna(X.mean())  # Fill NaN values with mean (same as training)
                
                print(f"Feature matrix shape: {X.shape}")
                
                # Make predictions for each segment
                predictions = loaded_pipe.predict(X)
                prediction_probabilities = None

                print(f"Predictions: {predictions}")
                
                # Get prediction probabilities if available
                if hasattr(loaded_pipe, 'predict_proba'):
                    try:
                        prediction_probabilities = loaded_pipe.predict_proba(X)
                    except:
                        pass
                
                # Display results
                print(f"\n=== PREDICTION RESULTS ===")
                print(f"Number of segments processed: {len(predictions)}")
                
                for i, pred in enumerate(predictions):
                    class_name = class_labels.get(pred, f"Unknown class {pred}")
                    prob_str = ""
                    if prediction_probabilities is not None:
                        prob_str = f" (confidence: {max(prediction_probabilities[i]):.3f})"
                    print(f"Segment {i+1}: {class_name}{prob_str}")
                
                # Overall prediction (majority vote)
                unique, counts = np.unique(predictions, return_counts=True)
                majority_pred = unique[np.argmax(counts)]
                majority_class = class_labels.get(majority_pred, f"Unknown class {majority_pred}")
                confidence = max(counts) / len(predictions)
                
                print(f"\nOVERALL PREDICTION: {majority_class}")
                print(f"Confidence: {confidence:.3f} ({max(counts)}/{len(predictions)} segments)")
                print("=" * 30)
                
                last_processed_row = last_processed_row + len(csi_df)
                prediction_count += 1
                time.sleep(5)

            else:
                if verbose:
                    print(f"Waiting for more data... (current: {len(df)}, need: {last_processed_row + batch_size})")
                time.sleep(2)
                continue
                
        except FileNotFoundError:
            print("Error: scripts/tempfile/tempData.csv not found. Make sure CSI data is being written to this file.")
            time.sleep(5)
        except Exception as e:
            print(f"Unexpected error in data processing: {e}")
            time.sleep(5)

if __name__ == "__main__":
    print("WiFi Sensing Real-time Prediction System")
    print("========================================")
    
    # Start the real-time processing thread
    t2 = threading.Thread(target=testData)
    t2.daemon = True  # Make thread daemon so it stops when main program stops
    t2.start()
    
    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down real-time prediction system...")