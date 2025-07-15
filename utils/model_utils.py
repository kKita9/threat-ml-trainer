import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

MODEL_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

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


def train_model(X, y, model_type='xgb', random_state=42, **kwargs):
    """
    Train a machine learning model using the selected algorithm.

    Parameters:
        X (pd.DataFrame): features
        y (pd.Series): labels
        model_type (str): 'xgb' or 'rf'
        random_state (int): reproducibility seed
        kwargs: extra model parameters

    Returns:
        trained model
    """
    if model_type == 'xgb':
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                              random_state=random_state, **kwargs)
    elif model_type == 'rf':
        model = RandomForestClassifier(random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    model.fit(X, y)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model on test data and print key metrics.

    Parameters:
        model: trained classifier
        X_test (pd.DataFrame): test features
        y_test (pd.Series): test labels
    """
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    try:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except Exception:
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
