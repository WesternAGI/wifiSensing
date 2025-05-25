"""
visualize_csi.py
Visualizes CSI data, amplitude/phase, and activity segments.
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_amplitude(df):
    if 'CSI_DATA' in df.columns:
        csi_matrix = np.stack(df['CSI_DATA'].apply(lambda x: np.array([int(i) for i in str(x).strip('[]').split() if i != ''])))
        plt.imshow(np.abs(csi_matrix), aspect='auto', cmap='viridis')
        plt.colorbar(label='Amplitude')
        plt.title('CSI Amplitude')
        plt.xlabel('Subcarrier')
        plt.ylabel('Packet Index')
        plt.show()

def main(input_file, plot_type):
    df = pd.read_csv(input_file)
    if plot_type == 'amplitude':
        plot_amplitude(df)
    else:
        print(f"Plot type {plot_type} not implemented.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize CSI data.")
    parser.add_argument('--input', required=True, help='Input processed CSV file')
    parser.add_argument('--plot-type', default='amplitude', help='Type of plot: amplitude (default)')
    args = parser.parse_args()
    main(args.input, args.plot_type)
