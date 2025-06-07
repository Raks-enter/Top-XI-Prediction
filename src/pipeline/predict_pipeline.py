import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, input_df: pd.DataFrame):
        try:
            model_path = 'artifact/model.pkl'
            preprocessor_path = 'artifact/preprocessor.pkl'

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            # Feature Engineering (must match what you did during training)
            df = input_df.copy()
            df['HS'] = df['HS'].astype(str).str.replace('*', '', regex=False).astype(float)
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

            # Get Top 11 players
            top_11 = df.sort_values(by='Predicted_PPS', ascending=False).head(11)

            return top_11[['Player', 'Predicted_PPS']]

        except Exception as e:
            raise CustomException(e, sys)
