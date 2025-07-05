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
import torch
import torch.nn as nn
from torch import optim
from collections import Counter

import pickle
import numpy as np
import pandas as pd
from utils import csi_to_amplitude_phase, N_of_samples, filter_df, select_data_portion

# Define the PyTorch model architecture (must match training)
class WiFiSensingNet(nn.Module):
    def __init__(self, input_size=10800, num_classes=2):
        super(WiFiSensingNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.bn3 = nn.BatchNorm1d(512)
        self.fc4 = nn.Linear(512, 256)
        self.bn4 = nn.BatchNorm1d(256)
        self.fc5 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = self.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        x = self.relu(self.bn4(self.fc4(x)))
        x = self.dropout(x)
        x = self.fc5(x)
        return x

# Load all available models
MODEL_DIR = "../model/"
DATAFILE_PATH = "tempfile/testdata.csv"

# Initialize model containers
pkl_models = {}
pth_models = {}

# Load all .pkl models (scikit-learn pipelines)
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pkl"):
        try:
            model = pickle.load(open(os.path.join(MODEL_DIR, file), "rb"))
            pkl_models[file] = model
            print(f"Loaded scikit-learn model: {file}")
            if hasattr(model, 'named_steps'):
                print(f"  Pipeline steps: {list(model.named_steps.keys())}")
            if hasattr(model, 'classes_'):
                print(f"  Classes: {model.classes_}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

# Load all .pth models (PyTorch models)
for file in os.listdir(MODEL_DIR):
    if file.endswith(".pth"):
        try:
            model = WiFiSensingNet()
            model.load_state_dict(torch.load(os.path.join(MODEL_DIR, file), map_location=torch.device('cpu')))
            model.eval()
            pth_models[file] = model
            print(f"Loaded PyTorch model: {file}")
        except Exception as e:
            print(f"Error loading {file}: {e}")

if not pkl_models and not pth_models:
    print("No valid models found in the model directory!")
    exit(1)

# Class labels mapping (based on notebook training data)
# From notebook: classes = ['empty', 'working'] with labels assigned as i (index)
class_labels = {
    0: "empty",
    1: "working"
}

# Function to get predictions from all models
def get_all_predictions(X):
    all_predictions = {}
    
    # Get predictions from scikit-learn models
    for model_name, model in pkl_models.items():
        try:
            preds = model.predict(X)
            all_predictions[model_name] = preds
        except Exception as e:
            print(f"Error getting predictions from {model_name}: {e}")
    
    # Get predictions from PyTorch models
    if pth_models:
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            for model_name, model in pth_models.items():
                try:
                    outputs = model(X_tensor)
                    _, preds = torch.max(outputs, 1)
                    all_predictions[model_name] = preds.numpy()
                except Exception as e:
                    print(f"Error getting predictions from {model_name}: {e}")
    
    return all_predictions

# Function to get combined prediction using voting
def get_combined_prediction(predictions):
    if not predictions:
        return None, 0.0
    
    # Transpose to get predictions per sample
    sample_predictions = list(zip(*predictions.values()))
    final_predictions = []
    
    for sample_preds in sample_predictions:
        # Count votes for each class
        vote_counts = Counter(sample_preds)
        # Get the most common prediction
        most_common = vote_counts.most_common(1)[0]
        final_predictions.append(most_common[0])
    
    # Calculate confidence as percentage of models that agreed with the majority
    confidence = sum(1 for pred in sample_predictions[0] if pred == final_predictions[0]) / len(sample_predictions[0])
    
    return final_predictions, confidence

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
                
                # Make predictions using all models
                all_predictions = get_all_predictions(X)
                
                # Get combined prediction using voting
                combined_preds, combined_confidence = get_combined_prediction(all_predictions)
                
                # Display results
                print(f"\n=== PREDICTION RESULTS ===")
                print(f"Number of segments processed: {len(X)}")
                print(f"Number of models used: {len(all_predictions)}")
                
                # Print predictions from each model
                print("\nIndividual Model Predictions:")
                for model_name, preds in all_predictions.items():
                    model_votes = Counter(preds)
                    total = len(preds)
                    votes_str = ", ".join([f"{class_labels.get(k, k)}: {v}/{total}" for k, v in model_votes.items()])
                    print(f"{model_name}: {votes_str}")
                
                # Print combined prediction
                if combined_preds is not None:
                    combined_votes = Counter(combined_preds)
                    majority_pred = combined_votes.most_common(1)[0][0]
                    majority_class = class_labels.get(majority_pred, f"Unknown class {majority_pred}")
                    
                    print(f"\nCOMBINED PREDICTION: {majority_class}")
                    print(f"Confidence: {combined_confidence:.3f} (agreement among models)")
                
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
            print(f"Error: {DATAFILE_PATH} not found. Make sure CSI data is being written to this file.")
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