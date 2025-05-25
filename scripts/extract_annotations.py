"""
extract_annotations.py
Extracts and manages activity/label annotations from raw or labeled CSI data.
"""
import argparse
import pandas as pd

def extract_annotations(input_file, output_file):
    # Example: Extract 'type' and 'mac' columns as annotations
    df = pd.read_csv(input_file)
    annotations = df[['type', 'mac']].drop_duplicates()
    annotations.to_csv(output_file, index=False)
    print(f"Annotations extracted and saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract annotations from CSI data.")
    parser.add_argument('--input', required=True, help='Input CSV file')
    parser.add_argument('--output', required=True, help='Output annotation CSV file')
    args = parser.parse_args()
    extract_annotations(args.input, args.output)
