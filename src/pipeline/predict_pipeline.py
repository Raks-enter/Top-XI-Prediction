import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def clean_numeric_columns(self, df, cols):
        for col in cols:
            # Convert to string, remove all non-digit and non-dot characters (like +, *, etc)
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            # Replace empty strings with '0'
            df[col] = df[col].replace('', '0')
            # Convert to numeric, coerce errors and fill NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    def predict(self, input_df: pd.DataFrame):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            df = input_df.copy()

            # Columns that need cleaning because they might contain '+', '*', or other chars
            cols_to_clean = ['HS', 'Mat', 'Inns', 'Runs', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']

            # Clean all these columns
            df = self.clean_numeric_columns(df, cols_to_clean)

            # Now feature engineering
            df['Boundary_Runs'] = df['4s'] * 4 + df['6s'] * 6
            df['Dot_Ball_Percentage'] = 1 - (df['Runs'] / df['BF']).clip(upper=1)
            df['Dot_Ball_Percentage'] = df['Dot_Ball_Percentage'].fillna(0).clip(lower=0)
            df['Consistency'] = (df['Ave'] / df['Mat']).replace([np.inf, -np.inf], 0).fillna(0)

            # Drop columns not needed for prediction
            features = df.drop(columns=['Player', 'Unnamed: 0', 'Span', 'NO'], errors='ignore')

            # Preprocessing
            data_scaled = preprocessor.transform(features)

            # Predict PPS
            preds = model.predict(data_scaled)

            # Add predictions to original dataframe
            df['Predicted_PPS'] = preds

            # Get Top 11 players based on prediction
            top_11 = df.sort_values(by='Predicted_PPS', ascending=False).head(11)

            return top_11[['Player', 'Predicted_PPS']]

        except Exception as e:
            raise CustomException(e, sys)
