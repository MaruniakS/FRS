import argparse
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import preprocessing
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

from constants import FEATURES

from numpy.random import seed
seed(1)
tf.random.set_seed(2)

N_CLASSES = 3
BATCH_SIZE = 1
EPOCHS = 10
SEQUENCE_LENGTH = 10
TRAIN_SIZE = 0.7
TEST_SIZE = 0.1
SEED_SPLIT1 = 0
SEED_SPLIT2 = 1

def normalize_weights(shap_weights, min_val=1, max_val=10):
    """
    Normalize SHAP weights to a meaningful range [min_val, max_val]
    
    Parameters:
    - shap_weights: dict → Original SHAP weights
    - min_val: float → Minimum normalized weight
    - max_val: float → Maximum normalized weight
    
    Returns:
    - Normalized weights dictionary
    """
    normalized_shap = {}
    
    for label, feature_weights in shap_weights.items():
        values = np.array(list(feature_weights.values()))
        
        # Normalize weights to the range [min_val, max_val]
        min_w, max_w = values.min(), values.max()
        norm_values = min_val + ((values - min_w) / (max_w - min_w)) * (max_val - min_val)
        
        # Reconstruct dictionary
        normalized_shap[label] = {feat: norm  for feat, norm in zip(feature_weights.keys(), norm_values)}
    
    return normalized_shap

# Load SHAP weights
with open("test_shap_values.json", "r") as f:
# with open("research_shap_values.json", "r") as f:
# with open("raw_shap_values.json", "r") as f:
    SHAP_WEIGHTS = json.load(f)
    # shap_w = json.load(f)
    # SHAP_WEIGHTS = normalize_weights(shap_w, min_val=1, max_val=10)   
    # print(SHAP_WEIGHTS) 


def train_valid_test_split(X, y, train_size=TRAIN_SIZE, test_size=TEST_SIZE):
    X_train, X_rem, y_train, y_rem = train_test_split(X, y, train_size=train_size, random_state=SEED_SPLIT1)
    X_valid, X_test, y_valid, y_test = train_test_split(X_rem, y_rem, test_size=test_size/(1 - train_size), random_state=SEED_SPLIT2)
    return X_train, X_valid, X_test, y_train, y_valid, y_test

def apply_shap_weights(X, label):
    """
    Apply SHAP-based feature weighting using feature names, ensuring only specified
    features are modified while others remain unchanged.

    Parameters:
    - X: np.ndarray of shape (num_sequences, num_features) → Feature sequences
    - label: Class label as an integer or string
    
    Returns:
    - X_weighted: np.ndarray with adjusted feature values
    """
    X_weighted = X.copy()  # Preserve original structure

    str_label = str(label)  # Convert label to string for dictionary lookup
    if str_label in SHAP_WEIGHTS:
        shap_weights = SHAP_WEIGHTS[str_label]  # Retrieve weights for this label

        # Iterate over the known feature names and apply weights correctly
        for feature_name, weight in shap_weights.items():
            if feature_name in FEATURES:  # Ensure the feature exists in the dataset
                feature_idx = FEATURES.index(feature_name)  # Get index
                X_weighted[:, feature_idx] *= weight  # Apply weight to correct column     
    return X_weighted

def get_sequences(df, sequence_length, overlapping_sequences=False):
    Xs, ys = {}, {}
    X = df[FEATURES]
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    
    step = 1 if overlapping_sequences else sequence_length
    
    for i in range(0, len(X_scaled) - sequence_length, step):
        y = np.zeros(N_CLASSES)
        label_index = max(df.label[i:i + sequence_length])
        y[label_index] = 1

        if label_index not in Xs:
            Xs[label_index] = []
            ys[label_index] = []

        X_weighted = apply_shap_weights(X_scaled[i:i + sequence_length], label_index)
        Xs[label_index].append(X_weighted)
        ys[label_index].append(y)
    
    return Xs, ys

def get_model(n_features, n_sequence):
    model = keras.Sequential([
        layers.Conv1D(filters=32, kernel_size=2, input_shape=(n_sequence, n_features), padding='same'),
        layers.MaxPooling1D(pool_size=2, strides=2),
        layers.LSTM(100),
        layers.Dropout(0.2),
        layers.Dense(N_CLASSES, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer=Adam(0.0067), metrics=['categorical_accuracy'])
    return model

def get_classification_report(model, xs, ys):
    predictions = np.argmax(model.predict(np.array(xs)), axis=-1)
    labels = [np.argmax(y) for y in ys]
    cm = tf.math.confusion_matrix(labels=labels, predictions=predictions)
    print('Confusion matrix')
    print(cm)
    print(classification_report(labels, predictions))

def main():
    df = pd.read_csv(args.features).fillna(0)
    df_final_test = pd.read_csv(args.separated_features).fillna(0)
    
    if args.only_anomalous:
        df = df[df.label != -1]
    
    Xs, ys = get_sequences(df, args.sequence_length, overlapping_sequences=args.overlapping_sequences)
    Xs_f, ys_f = get_sequences(df_final_test, args.sequence_length, overlapping_sequences=False)
    
    model = get_model(len(FEATURES), args.sequence_length)
    print(model.summary())
    
    x_train, x_valid, x_test, y_train, y_valid, y_test = [], [], [], [], [], []
    xs_ftest, ys_ftest = [], []
    for key in Xs:
        split = train_valid_test_split(Xs[key], ys[key], args.train_size, args.test_size)
        x_temp_train, x_temp_valid, x_temp_test, y_temp_train, y_temp_valid, y_temp_test = split
        x_train.extend(x_temp_train)
        x_valid.extend(x_temp_valid)
        x_test.extend(x_temp_test)
        y_train.extend(y_temp_train)
        y_valid.extend(y_temp_valid)
        y_test.extend(y_temp_test)
        xs_ftest.extend(Xs_f[key])
        ys_ftest.extend(ys_f[key])
        print(f'class={key} => {len(x_train)=}, {len(x_valid)=}, {len(x_test)=}, {len(xs_ftest)=}')
    
    xy_train = list(zip(x_train, y_train))
    np.random.shuffle(xy_train)
    x_train, y_train = zip(*xy_train)
    
    model.fit(
        np.asarray(x_train), np.asarray(y_train), batch_size=args.batch_size, epochs=args.epochs,
        validation_data=(np.array(x_valid), np.array(y_valid))
    )
    
    model.save(args.output)
    print("========== VALIDATION REPORT ===============")
    get_classification_report(model, x_valid, y_valid)
    if args.run_test_events_seen:
        print("========== TEST (SPLIT SET) REPORT ===============")
        get_classification_report(model, x_test, y_test)
    if args.run_test_events_not_seen:
        print("========== TEST (UNSEEN FEATURES) REPORT ===============")
        get_classification_report(model, xs_ftest, ys_ftest)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('features')
    parser.add_argument('separated_features')
    parser.add_argument('output')
    parser.add_argument('--run-test-events-seen', action='store_true')
    parser.add_argument('--run-test-events-not-seen', action='store_true')
    parser.add_argument('--sequence-length', type=int, default=SEQUENCE_LENGTH)
    parser.add_argument('--train-size', type=float, default=TRAIN_SIZE)
    parser.add_argument('--test-size', type=float, default=TEST_SIZE)
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    parser.add_argument('--overlapping-sequences', action='store_true')
    parser.add_argument('--only-anomalous', action='store_true')
    args = parser.parse_args()
    main()
