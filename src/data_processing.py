import pandas as pd

def preprocess_data(df):
    """
    Preprocess the Heart Disease dataset:
    - One-hot encode categorical variables
    - Ensure all features are numeric
    """
    categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]
    
    # One-hot encode categoricals
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded
