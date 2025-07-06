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
DATAFILE_PATH = "tempfile/empty1.csv"

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
    all_logits = {}
    
    # Get predictions from scikit-learn models
    for model_name, model in pkl_models.items():
        try:
            # Get class predictions
            preds = model.predict(X)
            all_predictions[model_name] = preds
            
            # Get logits or probabilities
            if hasattr(model, 'decision_function'):  # For SVMs
                logits = model.decision_function(X)
                # For binary classification, decision_function returns 1D array, convert to 2D
                if len(logits.shape) == 1:
                    # For binary classification, create a 2D array with both class scores
                    logits = np.column_stack([-logits, logits])
                all_logits[model_name] = logits
            elif hasattr(model, 'predict_proba'):  # For models with probability estimates
                probs = model.predict_proba(X)
                all_logits[model_name] = probs
            else:  # For models without probability estimates
                all_logits[model_name] = f"No probability estimates available for {model_name}"
                
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
                    all_logits[model_name] = outputs.numpy()  # Raw logits from PyTorch model
                except Exception as e:
                    print(f"Error getting predictions from {model_name}: {e}")
    
    return all_predictions, all_logits

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
    
    # Get the first available model for input shape info
    if pkl_models:
        model_name, model = next(iter(pkl_models.items()))
        print(f"Using model '{model_name}' for inference")
        if hasattr(model, 'named_steps'):
            print(f"  Pipeline steps: {list(model.named_steps.keys())}")
    else:
        print("Warning: No scikit-learn models available for inference")
    
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
                
                # Get predictions and logits from all models
                all_predictions, all_logits = get_all_predictions(X)
                
                # Get combined prediction using voting
                final_prediction, combined_confidence = get_combined_prediction(all_predictions)
                
                # Display results
                print("\n=== PREDICTION RESULTS ===")
                print(f"Number of segments processed: {len(X)}")
                print(f"Number of models used: {len(all_predictions)}")
                
                print("\nIndividual Model Predictions and Logits:")
                for model_name, preds in all_predictions.items():
                    pred_counts = Counter(preds)
                    pred_str = ", ".join([f"{class_labels[p]}: {c}/{len(preds)}" for p, c in pred_counts.items()])
                    print(f"\n{model_name}:")
                    print(f"  Predictions: {pred_str}")
                    
                    # Print logits or probabilities if available
                    if model_name in all_logits and all_logits[model_name] is not None:
                        logits = all_logits[model_name]
                        if isinstance(logits, str):
                            print(f"  {logits}")
                        else:
                            print("  Logits/Probabilities (per sample):")
                            for i, sample_logits in enumerate(logits):
                                if len(sample_logits) <= 10:  # Only print full logits if there aren't too many classes
                                    logit_str = ", ".join([f"{class_labels.get(j, j)}: {v:.4f}" for j, v in enumerate(sample_logits)])
                                else:
                                    # For many classes, just show the top 3
                                    top_indices = np.argsort(sample_logits)[-3:][::-1]  # Get indices of top 3 logits
                                    logit_str = ", ".join([f"{class_labels.get(j, j)}: {sample_logits[j]:.4f}" for j in top_indices])
                                    logit_str += f" ... (and {len(sample_logits)-3} more classes)"
                                print(f"    Sample {i+1}: {logit_str}")
                    else:
                        print("  No logits/probabilities available")
                
                # Print combined prediction
                if final_prediction is not None:
                    combined_votes = Counter(final_prediction)
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