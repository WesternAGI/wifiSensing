"""
preprocess_csi.py
Loads raw CSI CSVs, cleans, normalizes, and saves processed data.
"""
import argparse
import pandas as pd
import numpy as np

def preprocess_csi(input_file, output_file):
    df = pd.read_csv(input_file)
    # Example preprocessing: Remove NaNs, normalize CSI_DATA
    df = df.dropna()
    if 'CSI_DATA' in df.columns:
        df['CSI_DATA'] = df['CSI_DATA'].apply(lambda x: np.array([int(i) for i in str(x).strip('[]').split() if i != '']))
    df.to_csv(output_file, index=False)
    print(f"Preprocessed CSI data saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw CSI data.")
    parser.add_argument('--input', required=True, help='Input raw CSV file')
    parser.add_argument('--output', required=True, help='Output processed CSV file')
    args = parser.parse_args()
    preprocess_csi(args.input, args.output)
