import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


def split_data(df, target_col="HeartDisease"):
    """Split dataset into train/test."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_neural_net(X_train, y_train, hidden_layer_sizes=(32, 16), max_iter=500, random_state=42):
    """Train a simple feedforward neural network."""
    model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation="relu", solver="adam", max_iter=max_iter, random_state=random_state)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, name="Model"):
    """Prints accuracy and classification report."""
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Results:")
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))
    return acc
