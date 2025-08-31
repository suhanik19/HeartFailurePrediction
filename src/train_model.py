# src/train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def train_logistic_regression(df, target_col="DEATH_EVENT"):
    """
    Train a Logistic Regression model on the dataset.
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Target column name
    Returns:
        model, X_test, y_test
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred))

    return model, X_test, y_test
