# train.py
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import joblib
import matplotlib.pyplot as plt

def load_and_preprocess(csv_path, nrows=None, binary=True):
    df = pd.read_csv(csv_path, nrows=nrows)
    # Drop obvious non-feature cols if present
    for c in ['id','attack_cat']:
        if c in df.columns: df = df.drop(columns=[c])
    # target: 'label' (0 normal, 1 attack) in UNSW-NB15
    if 'label' not in df.columns:
        raise ValueError("CSV must contain 'label' column")
    y = df['label'].astype(int).values
    X = df.drop(columns=['label'])
    # Keep numeric columns only (fast, robust)
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    X = X[numeric_cols].fillna(0)
    # Quick encode small categorical if exist (rare after numeric filter)
    # scale numeric
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    return Xs, y, scaler, numeric_cols

def build_model(input_timesteps, input_features):
    # we'll treat features as a 1D "time-series" of length = input_timesteps, features=1
    inp = layers.Input(shape=(input_timesteps, 1))
    x = layers.Conv1D(filters=64, kernel_size=3, padding='same', activation='relu')(inp)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Conv1D(filters=128, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool1D(pool_size=2)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy',
                                                                         tf.keras.metrics.Precision(name='precision'),
                                                                         tf.keras.metrics.Recall(name='recall')])
    return model

def main(args):
    print("Loading and preprocessing...")
    X, y, scaler, feature_cols = load_and_preprocess(args.csv, nrows=args.nrows)
    # reshape: (samples, timesteps, features=1)
    timesteps = X.shape[1]
    X_reshaped = X.reshape((X.shape[0], timesteps, 1)).astype('float32')
    # train/test split (time-aware not possible here) — stratify to maintain classes
    X_train, X_temp, y_train, y_temp = train_test_split(X_reshaped, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # class weights
    cw = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = {i: cw[i] for i in range(len(cw))}
    print("Class weights:", class_weights)

    model = build_model(input_timesteps=timesteps, input_features=1)
    model.summary()

    # callbacks
    es = callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True)
    chk = callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
    red = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

    history = model.fit(X_train, y_train, epochs=args.epochs, batch_size=args.batch_size,
                        validation_data=(X_val, y_val), class_weight=class_weights,
                        callbacks=[es, chk, red], verbose=2)

    # save final artifacts
    print("Saving artifacts...")
    model.save('final_model.h5')
    joblib.dump(scaler, 'scaler.pkl')
    # save feature list
    pd.Series(feature_cols).to_csv('feature_cols.txt', index=False, header=False)

    # evaluate
    print("Evaluating on test set...")
    res = model.evaluate(X_test, y_test, verbose=0)
    print("Test results (loss, accuracy, precision, recall):", res)

    # plot train history quickly
    plt.figure(figsize=(8,4))
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend()
    plt.savefig('training_loss.png')
    print("Done. Artifacts: best_model.h5 final_model.h5 scaler.pkl feature_cols.txt training_loss.png")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, default='UNSW_NB15.csv', help='path to UNSW CSV')
    parser.add_argument('--nrows', type=int, default=None, help='optional: limit rows for speed (e.g., 50000)')
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch_size', type=int, default=256)
    args = parser.parse_args()
    main(args)
