import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

def prepare_anomaly_data(df, label_column='attack_types_encoded', benign_label=0, sample_size=50000):
    benign_data = df[df[label_column] == benign_label].sample(n=sample_size, random_state=42)
    attack_data = df[df[label_column] != benign_label]
    return benign_data, attack_data

def split_data(df, label_column='Label', test_size=0.3, random_state=42, stratify=True):
    """
    Split data into train and test sets.

    Parameters:
        df (pd.DataFrame): full dataset
        label_column (str): target column name
        test_size (float): fraction for testing
        random_state (int): reproducibility seed
        stratify (bool): preserve class balance if True

    Returns:
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[label_column])
    y = df[label_column]

    stratify_param = y if stratify else None
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_param)

def encode_labels(df, label_name) -> tuple[LabelEncoder, pd.DataFrame]:
    le = LabelEncoder()
    df[f'{label_name}_encoded'] = le.fit_transform(df['attack_types'])

    for i, class_name in enumerate(le.classes_):
        print(f"{i}: {class_name}")

    return le, df


def train_model(X, y, model_type='xgb', random_state=42, **kwargs):
    """
    Train a machine learning model using the selected algorithm.

    Parameters:
        X (pd.DataFrame): features
        y (pd.Series): labels
        model_type (str): 'xgb', 'rf', 'lr' or 'svm'
        random_state (int): reproducibility seed
        kwargs: extra model parameters

    Returns:
        trained model
    """
    num_classes = len(set(y)) if y is not None else 1

    if model_type == 'xgb':
        if num_classes > 2:
            kwargs.setdefault('objective', 'multi:softmax')
            kwargs.setdefault('num_class', num_classes)
            kwargs.setdefault('eval_metric', 'mlogloss')
        else:
            kwargs.setdefault('objective', 'binary:logistic')
            kwargs.setdefault('eval_metric', 'logloss')
        model = XGBClassifier(random_state=random_state, **kwargs)
        model.fit(X, y)
        return model

    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state, **kwargs)
        model.fit(X, y)
        return model

    elif model_type == 'lr':
        if num_classes > 2:
            kwargs.setdefault('multi_class', 'multinomial')
            kwargs.setdefault('solver', 'lbfgs')
        model = LogisticRegression(random_state=random_state, max_iter=1000, **kwargs)
        model.fit(X, y)
        return model

    elif model_type == 'svm':
        model = SVC(kernel='linear', probability=False, random_state=random_state, **kwargs)
        model.fit(X, y)
        return model

    elif model_type == 'isof':
        model = IsolationForest(contamination=kwargs.get('contamination', 0.01), random_state=random_state)
        model.fit(X)
        return model

    elif model_type == 'ae':
        input_dim = X.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(64, activation="relu")(input_layer)
        encoded = Dense(32, activation="relu")(encoded)
        encoded = Dense(16, activation="relu")(encoded)

        decoded = Dense(32, activation="relu")(encoded)
        decoded = Dense(64, activation="relu")(decoded)
        output_layer = Dense(input_dim, activation="linear")(decoded)

        autoencoder = Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer="adam", loss="mse")
        autoencoder.fit(X, X, epochs=kwargs.get('epochs', 10), batch_size=kwargs.get('batch_size', 128),
                        shuffle=True, validation_split=0.1, verbose=0)
        return autoencoder

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


# def evaluate_model(model, X_test, y_test):
#     y_pred = model.predict(X_test)
#     print("=== Classification Report ===")
#     print(classification_report(y_test, y_pred))
#
#     print("=== Confusion Matrix ===")
#     print(confusion_matrix(y_test, y_pred))
#
#     if hasattr(model, "predict_proba"):
#         try:
#             y_proba = model.predict_proba(X_test)
#             classes = model.classes_ if hasattr(model, "classes_") else sorted(set(y_test))
#
#             if len(classes) > 2:
#                 y_bin = label_binarize(y_test, classes=classes)
#                 auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
#             else:
#                 auc = roc_auc_score(y_test, y_proba[:, 1])
#
#             print(f"ROC AUC: {auc:.4f}")
#         except Exception as e:
#             print(f"Could not calculate ROC AUC: {e}")
#     else:
#         print("Model does not support probability prediction.")


def evaluate_model(model, X_test, y_test, class_names=None, show_heatmap=False):
    if isinstance(model, IsolationForest):
        y_pred = model.predict(X_test)
        y_pred = np.where(y_pred == 1, 0, 1)  # 0 = benign, 1 = anomaly

    elif isinstance(model, Model):  # Keras autoencoder
        reconstructions = model.predict(X_test)
        mse = np.mean(np.power(X_test - reconstructions, 2), axis=1)
        threshold = np.mean(mse) + 3 * np.std(mse)
        y_pred = [1 if e > threshold else 0 for e in mse]

    else:
        y_pred = model.predict(X_test)

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=class_names if class_names else None))

    print("=== Confusion Matrix ===")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    if show_heatmap:
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()

    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X_test)
            classes = model.classes_ if hasattr(model, "classes_") else sorted(set(y_test))

            if len(classes) > 2:
                y_bin = label_binarize(y_test, classes=classes)
                auc = roc_auc_score(y_bin, y_proba, average='macro', multi_class='ovr')
            else:
                auc = roc_auc_score(y_test, y_proba[:, 1])

            print(f"ROC AUC: {auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
    else:
        print("Model does not support probability prediction.")


def save_model(model, name="model_v1.pkl"):
    """
    Save the trained model to a file.

    Parameters:
        model: trained classifier
        path (str): folder to save the model
        name (str): filename

    Returns:
        full path to the saved file
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    full_path = os.path.join(MODEL_DIR, name)
    joblib.dump(model, full_path)
    print(f"Model saved to: {full_path}")
    return full_path
