import pandas as pd
import numpy as np
import os
import sys
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.exception import CustomException
from src.logger import logging
import pickle

class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        # Define which columns to scale
        numerical_cols = [
            'Mat', 'Inns', 'Runs', 'HS', 'Ave', 'BF', 'SR',
            '100', '50', '0', '4s', '6s',
            'Boundary_Runs', 'Dot_Ball_Percentage', 'Consistency'
        ]

        pipeline = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        preprocessor = ColumnTransformer([
            ('num_pipeline', pipeline, numerical_cols)
        ])
        return preprocessor

    def clean_numeric_columns(self, df, columns):
        for col in columns:
            df[col] = df[col].astype(str).str.replace(r'[^\d.]', '', regex=True)
            df[col] = df[col].replace('', '0')
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        return df

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            # Clean numeric columns to remove non-digit characters
            numeric_cols_to_clean = ['Mat', 'Inns', 'Runs', 'HS', 'Ave', 'BF', 'SR', '100', '50', '0', '4s', '6s']
            train_df = self.clean_numeric_columns(train_df, numeric_cols_to_clean)
            test_df = self.clean_numeric_columns(test_df, numeric_cols_to_clean)

            # Add engineered features to both train and test
            for df in [train_df, test_df]:
                df['Boundary_Runs'] = df['4s'] * 4 + df['6s'] * 6
                df['Dot_Ball_Percentage'] = 1 - (df['Runs'] / df['BF']).clip(upper=1)
                df['Dot_Ball_Percentage'] = df['Dot_Ball_Percentage'].fillna(0).clip(lower=0)
                df['Consistency'] = (df['Ave'] / df['Mat']).replace([np.inf, -np.inf], 0).fillna(0)

            # Define or create target column PPS
            target_column = 'PPS'
            if target_column not in train_df.columns:
                train_df['PPS'] = (
                    train_df['Runs'] * 0.4 +
                    train_df['Ave'] * 0.3 +
                    train_df['SR'] * 0.2 +
                    train_df['100'] * 5 +
                    train_df['50'] * 2.5 -
                    train_df['0'] * 2
                )
                test_df['PPS'] = (
                    test_df['Runs'] * 0.4 +
                    test_df['Ave'] * 0.3 +
                    test_df['SR'] * 0.2 +
                    test_df['100'] * 5 +
                    test_df['50'] * 2.5 -
                    test_df['0'] * 2
                )

            # Store player names (for future use)
            player_names_test = test_df['Player']

            # Drop unnecessary columns
            columns_to_drop = ['Unnamed: 0', 'Span', 'NO', 'Player']
            train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], inplace=True)
            test_df.drop(columns=[col for col in columns_to_drop if col in test_df.columns], inplace=True)

            # Separate input and target features
            input_features_train = train_df.drop(columns=['PPS'])
            target_feature_train = train_df['PPS']

            input_features_test = test_df.drop(columns=['PPS'])
            target_feature_test = test_df['PPS']

            # Transform data
            preprocessing_obj = self.get_data_transformer_object()
            input_features_train_scaled = preprocessing_obj.fit_transform(input_features_train)
            input_features_test_scaled = preprocessing_obj.transform(input_features_test)

            # Save preprocessor
            with open(self.config.preprocessor_obj_file_path, 'wb') as f:
                pickle.dump(preprocessing_obj, f)

            return (
                np.c_[input_features_train_scaled, target_feature_train.to_numpy()],
                np.c_[input_features_test_scaled, target_feature_test.to_numpy()],
                self.config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
