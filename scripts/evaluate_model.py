"""
evaluate_model.py
Loads models and test data, evaluates and prints reports.
"""
import argparse
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model_file, test_file):
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    df = pd.read_csv(test_file)
    X = np.stack(df['CSI_DATA'].apply(lambda x: np.array([int(i) for i in str(x).strip('[]').split() if i != ''])))
    y = df['type'] if 'type' in df.columns else None
    y_pred = model.predict(X)
    print(confusion_matrix(y, y_pred))
    print(classification_report(y, y_pred))

def main(model_file, test_file):
    evaluate_model(model_file, test_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ML model on CSI data.")
    parser.add_argument('--model', required=True, help='Trained model file (pickle)')
    parser.add_argument('--test', required=True, help='Test CSV file')
    args = parser.parse_args()
    main(args.model, args.test)
