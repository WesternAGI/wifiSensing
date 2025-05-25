"""
train_model.py
Loads processed data, trains ML models, saves models.
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle

def train_knn(input_file, output_file):
    df = pd.read_csv(input_file)
    X = np.stack(df['CSI_DATA'].apply(lambda x: np.array([int(i) for i in str(x).strip('[]').split() if i != ''])))
    y = df['type'] if 'type' in df.columns else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    with open(output_file, 'wb') as f:
        pickle.dump(clf, f)
    print(f"KNN model trained and saved to {output_file}")

def main(input_file, output_file):
    train_knn(input_file, output_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ML model on CSI data.")
    parser.add_argument('--data', required=True, help='Processed CSV file')
    parser.add_argument('--output', required=True, help='Output model file (pickle)')
    args = parser.parse_args()
    main(args.data, args.output)
