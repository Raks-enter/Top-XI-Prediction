import pandas as pd
import numpy as np
import pickle

from src.components.data_transformation import DataTransformation
from src.utils import load_object  # You need to define this utility if not already

def predict_top_11(input_csv_path):
    # Load new match player stats
    df = pd.read_csv(input_csv_path)

    # Feature engineering (match your training logic)
    df['HS'] = df['HS'].astype(str).str.replace('*', '', regex=False).astype(float)
    df['Boundary_Runs'] = df['4s'] * 4 + df['6s'] * 6
    df['Dot_Ball_Percentage'] = 1 - (df['Runs'] / df['BF']).clip(upper=1)
    df['Dot_Ball_Percentage'] = df['Dot_Ball_Percentage'].fillna(0).clip(lower=0)
    df['Consistency'] = (df['Ave'] / df['Mat']).replace([np.inf, -np.inf], 0).fillna(0)

    # Drop unused columns
    df_input = df.drop(columns=['Player', 'Unnamed: 0', 'Span', 'NO'], errors='ignore')

    # Load preprocessor and model
    preprocessor = load_object('artifact/preprocessor.pkl')
    model = load_object('artifact/model.pkl')

    # Transform and predict
    X = preprocessor.transform(df_input)
    predictions = model.predict(X)

    # Add predictions to dataframe
    df['Predicted_PPS'] = predictions

    # Get top 11
    top_11 = df.sort_values(by='Predicted_PPS', ascending=False).head(11)

    return top_11[['Player', 'Predicted_PPS']]

# Example usage:
if __name__ == "__main__":
    top_players = predict_top_11("data/new_match_players.csv")  # update path
    print("🏏 Top 11 Players:")
    print(top_players)
