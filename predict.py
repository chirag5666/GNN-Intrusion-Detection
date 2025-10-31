# predict.py
import sys
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

def load_model_and_artifacts(model_path='final_model.h5', scaler_path='scaler.pkl', feat_path='feature_cols.txt'):
    model = tf.keras.models.load_model(model_path)
    scaler = joblib.load(scaler_path)
    feat_cols = pd.read_csv(feat_path, header=None).iloc[:,0].tolist()
    return model, scaler, feat_cols

def preprocess_input(df, feat_cols, scaler):
    X = df[feat_cols].fillna(0)
    Xs = scaler.transform(X)
    timesteps = Xs.shape[1]
    Xs = Xs.reshape((Xs.shape[0], timesteps, 1)).astype('float32')
    return Xs

def predict_csv(csv_path):
    model, scaler, feat_cols = load_model_and_artifacts()
    df = pd.read_csv(csv_path)
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    X = preprocess_input(df, feat_cols, scaler)
    preds = model.predict(X, batch_size=256)
    labels = (preds.flatten() >= 0.5).astype(int)
    out = pd.DataFrame({'pred_prob': preds.flatten(), 'pred_label': labels})
    out.to_csv('predictions.csv', index=False)
    print("Saved predictions to predictions.csv (0=normal,1=attack)")

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <csv_to_predict>")
        sys.exit(1)
    predict_csv(sys.argv[1])
